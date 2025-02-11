from typing import Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from functools import partial
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from yatdense import YatDense
from yatembed import YatEmbed

@dataclass(frozen=True)
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    num_embeds: int = 768
    dropout_rate: float = 0.1
    use_bias: bool = True
    dtype: Any = jnp.float32
    epsilon: float = 1/137
    head_dim: int = field(init=False)
    
    def __post_init__(self):
        # Compute and cache head dimension
        object.__setattr__(self, 'head_dim', self.num_embeds // self.num_heads)
        assert self.num_embeds % self.num_heads == 0, "num_embeds must be divisible by num_heads"

class AttentionOutput(NamedTuple):
    output: jnp.ndarray
    attention_weights: Optional[jnp.ndarray] = None

class SelfAttention(nn.Module):
    num_heads: int
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1
    deterministic: Optional[bool] = None
    use_proj_bias: bool = True
    epsilon: float = 1/137
    return_attention: bool = False
    alpha_init: Any = lambda key, shape, dtype: jnp.ones(shape, dtype)  # Initialize alpha to 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, deterministic: Optional[bool] = None) -> jnp.ndarray | AttentionOutput:
        B, T, C = x.shape
        head_dim = C // self.num_heads
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)

        # Pre-compute static values
        alpha = self.param(
            'alpha',
            self.alpha_init,
            (1,),  # Single scalar parameter
            self.dtype)

        scale = jnp.log1p(head_dim).astype(self.dtype)
        inv_scale = (head_dim / scale) ** alpha

        # QKV projection with shape optimization
        qkv = YatDense(
            3 * C, 
            use_bias=self.use_proj_bias, 
            dtype=self.dtype, 
            name='c_attn'
        )(x)
        
        # More efficient reshape and split
        qkv = qkv.reshape(B, T, 3, self.num_heads, head_dim)
        q, k, v = jnp.moveaxis(qkv, 2, 0)  # Faster than array_split
        
        # Rearrange to [B, h, T, d] all at once
        q = jnp.transpose(q, (0, 3, 1, 2))
        k = jnp.transpose(k, (0, 3, 1, 2))
        v = jnp.transpose(v, (0, 3, 1, 2))

        # Optimized attention computation
        dot_product = jnp.einsum('bhid,bhjd->bhij', q, k, optimize='optimal')
        scaled_dot_product = dot_product 
        squared_dot_product = jnp.square(scaled_dot_product)
        
        # Vectorized distance computation
        q_norm = jnp.sum(jnp.square(q), axis=-1, keepdims=True)
        k_norm = jnp.sum(jnp.square(k), axis=-1, keepdims=True).transpose(0, 1, 3, 2)
        squared_dist = q_norm + k_norm - 2.0 * scaled_dot_product
        
        # Attention scores with improved numerical stability
        attn = squared_dot_product / (squared_dist + self.epsilon)
        attn = attn * inv_scale
        attn = jnp.where(mask, attn, jnp.finfo(self.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1).astype(self.dtype)
        
        if not deterministic:
            attn = nn.Dropout(self.dropout_rate)(attn, deterministic=deterministic)

        # Optimized output computation
        output = jnp.einsum('bhij,bhjd->bhid', attn, v, optimize='optimal')
        output = output.transpose(0, 2, 1, 3).reshape(B, T, C)
        output = YatDense(
            C, 
            use_bias=self.use_proj_bias, 
            dtype=self.dtype, 
            name='c_proj'
        )(output)
        
        if not deterministic:
            output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
            
        return AttentionOutput(output, attn) if self.return_attention else output

class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: Optional[bool] = None) -> jnp.ndarray:
        _, _, C = x.shape
        intermediate_dim = 4 * C
        
        x = YatDense(
            intermediate_dim, 
            dtype=self.config.dtype, 
            use_bias=self.config.use_bias, 
            name='c_fc'
        )(x)
        x = YatDense(
            C, 
            dtype=self.config.dtype, 
            use_bias=self.config.use_bias, 
            name='c_proj'
        )(x)
        
        if not deterministic:
            x = nn.Dropout(self.config.dropout_rate)(x, deterministic)
        return x

class Block(nn.Module):
    config: GPTConfig
    return_attention: bool = False

    def setup(self):
        self.attn = SelfAttention(
            num_heads=self.config.num_heads,
            dtype=self.config.dtype,
            dropout_rate=self.config.dropout_rate,
            epsilon=self.config.epsilon,
            return_attention=self.return_attention
        )
        self.mlp = MLP(self.config)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, 
                 deterministic: Optional[bool] = None) -> jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]:
        attn_output = self.attn(x, mask, deterministic)
        
        if self.return_attention:
            x = x + attn_output.output
            attention_weights = attn_output.attention_weights
        else:
            x = x + attn_output
            attention_weights = None
            
        x = x + self.mlp(x, deterministic)
        return (x, attention_weights) if self.return_attention else x

class GPT(nn.Module):
    config: GPTConfig
    return_attention: bool = False

    @nn.compact
    def __call__(self, idx: jnp.ndarray, deterministic: Optional[bool] = None) -> jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]:
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Optimize position and attention mask computation
        pos = jnp.arange(T, dtype=jnp.int32)[None]
        attn_mask = nn.make_causal_mask(idx, dtype=bool)

        # Embedding layers
        wte = YatEmbed(
            self.config.vocab_size, 
            self.config.num_embeds, 
            dtype=self.config.dtype, 
            name='wte'
        )
        wpe = YatEmbed(
            self.config.block_size, 
            self.config.num_embeds, 
            dtype=self.config.dtype, 
            name='wpe'
        )

        # Compute embeddings
        x = wte(idx) + wpe(pos)
        if not deterministic:
            x = nn.Dropout(self.config.dropout_rate)(x, deterministic)

        attention_weights = []
        # Process through transformer blocks
        for i in range(self.config.num_layers):
            block_output = Block(
                self.config, 
                return_attention=self.return_attention, 
                name=f'block_{i}'
            )(x, attn_mask, deterministic)
            
            if self.return_attention:
                x, block_attention = block_output
                attention_weights.append(block_attention)
            else:
                x = block_output

        # Final projection to vocabulary
        logits = wte.attend(x)
        return (logits, jnp.stack(attention_weights)) if self.return_attention else logits

    def init(self, rng: Any) -> FrozenDict:
        """Initialize model parameters using JAX JIT."""
        tokens = jnp.zeros((2, self.config.block_size), dtype=jnp.int32)
        return jax.jit(
            super().init, 
            static_argnums=(2,)
        )(rng, tokens, True)

def convert_hf_params(
    hf_params: FrozenDict, 
    num_heads: int, 
    num_embeds: int
) -> FrozenDict:
    """Convert Hugging Face parameters to model format with improved efficiency."""
    params = unfreeze(hf_params['transformer'])
    
    # Handle transformer blocks
    if 'h' in params:
        block_params = params.pop('h')
        params.update({f'block_{k}': v for k, v in block_params.items()})

    # Flatten and transform parameters
    flat_params = flatten_dict(params, sep='.')
    for k, v in flat_params.items():
        if k.endswith('attn.c_attn.kernel'):
            flat_params[k] = v.T
        elif k.endswith('attn.c_proj.kernel'):
            flat_params[k] = v.T
        elif k.split('.')[1] == 'mlp' and k.endswith('kernel'):
            flat_params[k] = v.T

    # Reconstruct parameter dictionary
    return freeze(unflatten_dict(
        {f'params.{k}': v for k, v in flat_params.items()}, 
        sep='.'
    ))

def get_pretrained_params(model_type: str) -> Tuple[GPTConfig, FrozenDict]:
    """Load pretrained parameters from Hugging Face models."""
    valid_models = {
        'gpt2':        {'layers': 12, 'heads': 12, 'embeds': 768},   # 124M params
        'gpt2-medium': {'layers': 24, 'heads': 16, 'embeds': 1024},  # 350M params
        'gpt2-large':  {'layers': 36, 'heads': 20, 'embeds': 1280},  # 774M params
        'gpt2-xl':     {'layers': 48, 'heads': 25, 'embeds': 1600},  # 1558M params
    }
    
    assert model_type in valid_models, f"Model type must be one of: {list(valid_models.keys())}"
    
    from transformers import FlaxGPT2LMHeadModel
    print(f"Loading weights from pretrained GPT: {model_type}")

    model_config = valid_models[model_type]
    config = GPTConfig(
        num_layers=model_config['layers'],
        num_heads=model_config['heads'],
        num_embeds=model_config['embeds']
    )

    model_hf = FlaxGPT2LMHeadModel.from_pretrained(model_type)
    params = convert_hf_params(
        model_hf.params['transformer'],
        config.num_heads,
        config.num_embeds
    )
    
    return config, params