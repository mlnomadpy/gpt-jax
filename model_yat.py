from typing import Any, Optional, Tuple
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from yatdense import YatDense

@dataclass(frozen=True)
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    num_embeds: int = 768
    dropout_rate: float = 0.1
    use_bias: bool = True
    dtype: Optional[str] = None



class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, deterministic=None):
        B, T, C = x.shape
        x = YatDense(4 * C, dtype=self.config.dtype, use_bias=self.config.use_bias, name='c_fc')(x)
        x = YatDense(C, dtype=self.config.dtype, use_bias=self.config.use_bias, name='c_proj')(x)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic)
        return x



class SelfAttention(nn.Module):
    num_heads: int
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1
    deterministic: Optional[bool] = None
    use_proj_bias: bool = True

    @nn.compact
    def __call__(self, x, mask, deterministic=None):
        B, T, C = x.shape
        print(f"SelfAttention input shape: {x.shape}")
        
        assert C % self.num_heads == 0
        head_dim = C // self.num_heads
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)

        print(f"Before YatDense - input shape: {x.shape}, target features: {3 * C}")
        qkv = YatDense(3 * C, use_bias=self.use_proj_bias, dtype=self.dtype, name='c_attn')(x)
        print(f"After YatDense qkv shape: {qkv.shape}")
        
        qkv = qkv.reshape(B, T, 3 * self.num_heads, head_dim)
        print(f"After reshape qkv shape: {qkv.shape}")
        
        q, k, v = jnp.array_split(qkv, 3, axis=2)
        print(f"Split shapes - q: {q.shape}, k: {k.shape}, v: {v.shape}")
        
        scale = 1.0 / jnp.sqrt(head_dim).astype(self.dtype)
        attn = jnp.einsum('...qhd,...khd->...hqk', q, k) * scale
        print(f"Attention weights shape: {attn.shape}")
        
        attn = jnp.where(mask, attn, jnp.finfo(self.dtype).min)
        attn = jax.nn.softmax(attn).astype(self.dtype)
        attn = nn.Dropout(self.dropout_rate)(attn, deterministic=deterministic)

        x = jnp.einsum('...hqk,...khd->...qhd', attn, v).reshape(B, T, C)
        print(f"Before final YatDense shape: {x.shape}")
        
        x = YatDense(C, use_bias=self.use_proj_bias, dtype=self.dtype, name='c_proj')(x)
        print(f"Final output shape: {x.shape}")

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return x


class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias)
        self.attn = SelfAttention(self.config.num_heads,
                                 self.config.dtype,
                                 dropout_rate=self.config.dropout_rate)
        self.ln_2 = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias)
        self.mlp = MLP(self.config)

    def __call__(self, x, mask=None, deterministic=None):
        print(f"\nBlock input shape: {x.shape}")
        ln1_out = self.ln_1(x)
        print(f"After LayerNorm 1 shape: {ln1_out.shape}")
        
        attn_out = self.attn(ln1_out, mask, deterministic)
        print(f"After attention shape: {attn_out.shape}")
        
        x = x + attn_out
        
        ln2_out = self.ln_2(x)
        print(f"After LayerNorm 2 shape: {ln2_out.shape}")
        
        mlp_out = self.mlp(ln2_out, deterministic)
        print(f"After MLP shape: {mlp_out.shape}")
        
        x = x + mlp_out
        print(f"Block output shape: {x.shape}")
        return x


class GPT(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, idx, deterministic=None):
        B, T = idx.shape
        print(f"\nGPT input shape: {idx.shape}")
        
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

        pos = jnp.arange(0, T)[None]
        attn_mask = nn.make_causal_mask(idx, dtype=bool)
        print(f"Attention mask shape: {attn_mask.shape}")

        wte = nn.Embed(self.config.vocab_size, self.config.num_embeds, dtype=self.config.dtype, name='wte')
        wpe = nn.Embed(self.config.block_size, self.config.num_embeds, dtype=self.config.dtype, name='wpe')

        token_embed = wte(idx)
        print(f"Token embedding shape: {token_embed.shape}")
        
        pos_embed = wpe(pos)
        print(f"Position embedding shape: {pos_embed.shape}")
        
        x = nn.Dropout(self.config.dropout_rate)(token_embed + pos_embed, deterministic)
        print(f"After embedding + dropout shape: {x.shape}")

        for i in range(self.config.num_layers):
            print(f"\nProcessing layer {i}")
            x = Block(self.config, name=str(i))(x, attn_mask, deterministic=deterministic)

        x = nn.LayerNorm(1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias, name='ln_f')(x)
        print(f"After final LayerNorm shape: {x.shape}")
        
        logits = wte.attend(x)
        print(f"Final logits shape: {logits.shape}")
        return logits


    def init(self, rng):
        """
        by jitting init, traced values instead of concrete values are used
        which saves memory (since un-jitted model may not fit in memory)
        """
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
