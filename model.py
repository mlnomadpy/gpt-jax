from typing import Any, Optional, Tuple
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

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


class SelfAttention(nn.Module):
    num_heads: int
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1
    use_proj_bias: bool = True

    @nn.compact
    def __call__(self, x, mask, deterministic=None):
        B, T, C = x.shape
        head_dim = C // self.num_heads

        qkv = nn.Dense(3 * C, use_bias=self.use_proj_bias, dtype=self.dtype, name='c_attn')(x)
        qkv = qkv.reshape(B, T, self.num_heads, 3, head_dim)
        q, k, v = jnp.moveaxis(qkv, 3, 0)  # Shape: (3, B, T, num_heads, head_dim)

        scale = 1.0 / jnp.sqrt(head_dim)
        attn_weights = jax.lax.batch_matmul(q, k.transpose(0, 1, 3, 2)) * scale  # (B, num_heads, T, T)

        # Fix: Ensure mask is reshaped *before* applying it
        mask = mask[:, :, None, :]  # Explicitly reshape to (B, num_heads, T, T) for TPU efficiency
        attn_weights = jnp.where(mask, attn_weights, -1e9)  # Apply mask safely

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(rate=self.dropout_rate)(attn_weights, deterministic=deterministic)

        attn_output = jax.lax.batch_matmul(attn_weights, v)
        attn_output = attn_output.reshape(B, T, C)

        attn_output = nn.Dense(C, use_bias=self.use_proj_bias, dtype=self.dtype, name='c_proj')(attn_output)
        return nn.Dropout(rate=self.dropout_rate)(attn_output, deterministic=deterministic)



class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, deterministic=None):
        x = nn.Dense(4 * self.config.num_embeds, dtype=self.config.dtype, use_bias=self.config.use_bias, name='c_fc')(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(self.config.num_embeds, dtype=self.config.dtype, use_bias=self.config.use_bias, name='c_proj')(x)
        return nn.Dropout(self.config.dropout_rate)(x, deterministic=deterministic)


class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias)
        self.attn = SelfAttention(self.config.num_heads, self.config.dtype, dropout_rate=self.config.dropout_rate)
        self.ln_2 = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias)
        self.mlp = MLP(self.config)

    def __call__(self, x, mask, deterministic=None):
        x = x + self.attn(self.ln_1(x), mask, deterministic)
        x = x + self.mlp(self.ln_2(x), deterministic)
        return x


class GPT(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, idx, deterministic=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        pos = jnp.arange(T)[None, :]
        mask = nn.make_causal_mask(idx, dtype=bool)

        wte = nn.Embed(self.config.vocab_size, self.config.num_embeds, dtype=self.config.dtype, name='wte')
        wpe = nn.Embed(self.config.block_size, self.config.num_embeds, dtype=self.config.dtype, name='wpe')

        x = wte(idx) + wpe(pos)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic)

        for i in range(self.config.num_layers):
            x = Block(self.config, name=f'block_{i}')(x, mask, deterministic=deterministic)

        x = nn.LayerNorm(1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias, name='ln_f')(x)
        logits = wte.attend(x)
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
    params.update(params.pop('h', {}))  # Flatten 'h' into main dict
    params = flatten_dict(params, sep='.')

    for k, v in params.items():
        if 'attn.c_attn.kernel' in k or 'attn.c_proj.kernel' in k or ('mlp' in k and 'kernel' in k):
            params[k] = v.T  # Transpose weights for Flax format

    return freeze(unflatten_dict({f'params.{k}': v for k, v in params.items()}, sep='.'))


def get_pretrained_params(model_type: str) -> Tuple[GPTConfig, FrozenDict]:
    assert model_type in ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
    
    from transformers import FlaxGPT2LMHeadModel
    print(f"Loading pretrained GPT-2 model: {model_type}")

    config_map = {
        'gpt2': GPTConfig(num_layers=12, num_heads=12, num_embeds=768),
        'gpt2-medium': GPTConfig(num_layers=24, num_heads=16, num_embeds=1024),
        'gpt2-large': GPTConfig(num_layers=36, num_heads=20, num_embeds=1280),
        'gpt2-xl': GPTConfig(num_layers=48, num_heads=25, num_embeds=1600),
    }
    config = config_map[model_type]

    model_hf = FlaxGPT2LMHeadModel.from_pretrained(model_type)
    params = convert_hf_params(model_hf.params['transformer'], config.num_heads, config.num_embeds)

    return config, params
