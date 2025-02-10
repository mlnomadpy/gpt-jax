from typing import Any, Optional, Tuple
from dataclasses import dataclass
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

def create_rotary_embedding(dim: int, max_seq_len: int, base: int = 10000):
    """Creates rotary positional embeddings."""
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2) / dim))
    position = jnp.arange(max_seq_len)
    sinusoid_input = jnp.einsum('i,j->ij', position, inv_freq)
    sin = jnp.sin(sinusoid_input)
    cos = jnp.cos(sinusoid_input)
    return sin, cos

def apply_rotary_embedding(x: jnp.ndarray, sin: jnp.ndarray, cos: jnp.ndarray) -> jnp.ndarray:
    """Applies rotary positional embedding to input tensor."""
    sin = sin[:x.shape[1]]  # Take only what we need
    cos = cos[:x.shape[1]]
    dim = x.shape[-1]
    sin = jnp.broadcast_to(sin[:, None, :], x.shape[:-1] + (dim//2,))
    cos = jnp.broadcast_to(cos[:, None, :], x.shape[:-1] + (dim//2,))
    x1, x2 = x[..., :dim//2], x[..., dim//2:]
    return jnp.concatenate([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], axis=-1)

class SelfAttention(nn.Module):
    num_heads: int
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1
    use_proj_bias: bool = True
    epsilon: float = 1e-6

    def setup(self):
        head_dim = self.num_heads * 64  # Standard head dimension
        self.sin, self.cos = create_rotary_embedding(head_dim, 2048)  # Support up to 2048 tokens

    @nn.compact
    def __call__(self, x, mask, deterministic=None):
        B, T, C = x.shape
        head_dim = C // self.num_heads

        # QKV projection
        qkv = YatDense(3 * C, use_bias=self.use_proj_bias, dtype=self.dtype, name='c_attn')(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, head_dim)
        q, k, v = jnp.moveaxis(qkv, 2, 0)  # Split into q, k, v and move batch dims

        # Apply rotary embeddings
        q = apply_rotary_embedding(q, self.sin, self.cos)
        k = apply_rotary_embedding(k, self.sin, self.cos)

        # Prepare mask
        if mask is None:
            mask = nn.make_causal_mask(jnp.ones((B, T)), dtype=bool)
        mask = mask[:, None, :, :]  # Add head dimension

        # Scaled dot-product attention
        scale = jnp.sqrt(head_dim).astype(self.dtype)
        attn = (q @ jnp.swapaxes(k, -2, -1)) / scale
        
        # Apply mask and softmax
        attn = jnp.where(mask, attn, jnp.finfo(self.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.dropout_rate)(attn, deterministic)

        # Compute output
        output = attn @ v
        output = output.reshape(B, T, C)
        output = YatDense(C, use_bias=self.use_proj_bias, dtype=self.dtype, name='c_proj')(output)
        output = nn.Dropout(self.dropout_rate)(output, deterministic)
        
        return output

class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, deterministic=None):
        hidden_dim = 4 * self.config.num_embeds
        gate = YatDense(hidden_dim, dtype=self.config.dtype, use_bias=self.config.use_bias, name='c_fc')(x)
        x = YatDense(hidden_dim, dtype=self.config.dtype, use_bias=self.config.use_bias, name='c_gate')(x)
        x = nn.gelu(x) * gate  # SwiGLU activation
        x = YatDense(self.config.num_embeds, dtype=self.config.dtype, use_bias=self.config.use_bias, name='c_proj')(x)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic)
        return x

class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        self.norm1 = nn.LayerNorm(dtype=self.config.dtype, use_bias=False)
        self.attn = SelfAttention(
            num_heads=self.config.num_heads,
            dtype=self.config.dtype,
            dropout_rate=self.config.dropout_rate,
            use_proj_bias=self.config.use_bias
        )
        self.norm2 = nn.LayerNorm(dtype=self.config.dtype, use_bias=False)
        self.mlp = MLP(self.config)

    def __call__(self, x, mask=None, deterministic=None):
        x = x + self.attn(self.norm1(x), mask, deterministic)
        x = x + self.mlp(self.norm2(x), deterministic)
        return x

class GPT(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, idx, deterministic=None):
        B, T = idx.shape
        assert T <= self.config.block_size, "Input too long!"

        pos = jnp.arange(0, T)[None]
        attn_mask = nn.make_causal_mask(idx, dtype=bool)

        wte = self.param('wte', nn.initializers.normal(stddev=0.02), 
                        (self.config.vocab_size, self.config.num_embeds))
        wpe = self.param('wpe', nn.initializers.normal(stddev=0.02), 
                        (self.config.block_size, self.config.num_embeds))

        token_embed = jnp.take(wte, idx, axis=0)
        pos_embed = jnp.take(wpe, pos, axis=0)
        x = nn.Dropout(self.config.dropout_rate)(token_embed + pos_embed, deterministic)

        for i in range(self.config.num_layers):
            x = Block(self.config, name=str(i))(x, attn_mask, deterministic=deterministic)

        x = nn.LayerNorm(dtype=self.config.dtype, use_bias=False, name='ln_f')(x)
        logits = x @ wte.T  # Weight tying
        return logits

    def init(self, rng):
        tokens = jnp.zeros((2, self.config.block_size), dtype=jnp.uint16)
        params = jax.jit(super().init, static_argnums=(2,))(rng, tokens, True)
        return params



def convert_hf_params(hf_params: FrozenDict, num_heads, num_embeds) -> FrozenDict:
    params = unfreeze(hf_params['transformer'])
    for k, v in params.pop('h', {}).items():
        params[k] = v

    params = flatten_dict(params, sep='.')
    for k in params.keys():
        #if k.endswith('attn.c_attn.bias'):
        #    params[k] = params[k].reshape(num_heads, -1)
        if k.endswith('attn.c_attn.kernel'):
            #params[k] = params[k].reshape(num_embeds, num_heads, -1) 
            params[k] = params[k].T
        elif k.endswith('attn.c_proj.kernel'):
            #params[k] = params[k].reshape(num_heads, -1, num_embeds)
            params[k] = params[k].T
        elif k.split('.')[1] == 'mlp' and k.endswith('kernel'):
            params[k] = params[k].T

    params = unflatten_dict({f'params.{k}': v for k, v in params.items()}, sep='.')
    return freeze(params)


def get_pretrained_params(model_type: str) -> Tuple[GPTConfig, FrozenDict]:
    """
    returns config and pretrained parameters from huggingface gpt models 
    """
    assert model_type in ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
    # only dropout can be overridden see more notes below
    from transformers import FlaxGPT2LMHeadModel
    print("loading weights from pretrained gpt: %s" % model_type)

    config = {
        'gpt2':         GPTConfig(num_layers=12, num_heads=12, num_embeds=768),  # 124M params
        'gpt2-medium':  GPTConfig(num_layers=24, num_heads=16, num_embeds=1024), # 350M params
        'gpt2-large':   GPTConfig(num_layers=36, num_heads=20, num_embeds=1280), # 774M params
        'gpt2-xl':      GPTConfig(num_layers=48, num_heads=25, num_embeds=1600), # 1558M params
    }[model_type]

    model_hf = FlaxGPT2LMHeadModel.from_pretrained(model_type)
    hf_params = model_hf.params['transformer']
    params = convert_hf_params(hf_params, config.num_heads, config.num_embeds)
    return config, params
