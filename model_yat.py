from typing import Any, Optional, Tuple
from dataclasses import dataclass
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
    dtype: Optional[str] = None

class SelfAttention(nn.Module):
    num_heads: int
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1
    deterministic: Optional[bool] = None
    use_proj_bias: bool = True
    epsilon: float = 1/137

    @nn.compact
    def __call__(self, x, mask, deterministic=None):
        B, T, C = x.shape
        assert C % self.num_heads == 0
        head_dim = C // self.num_heads
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)

        # [B, T, 3*C] -> [B, T, 3*h, d]
        qkv = YatDense(3 * C, use_bias=self.use_proj_bias, dtype=self.dtype, name='c_attn')(x)
        qkv = qkv.reshape(B, T, 3 * self.num_heads, head_dim)
        q, k, v = jnp.array_split(qkv, 3, axis=2)  # Each has shape [B, T, h, d]

        # Rearrange to [B, h, T, d]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Compute attention with squared dot product and euclidean distance
        scale = 1.0 / jnp.sqrt(head_dim).astype(self.dtype)
        # Compute dot product attention
        dot_product = jnp.einsum('bhid,bhjd->bhij', q, k)
        squared_dot_product = jnp.square(dot_product * scale)
        
        # Compute Euclidean distance squared between q and k vectors
        # Using: ||a-b||² = ||a||² + ||b||² - 2(a·b)
        # Computing in a TPU-friendly way
        q_norm = jnp.sum(jnp.square(q), axis=-1)[..., None]  # [B, h, T, 1]
        k_norm = jnp.sum(jnp.square(k), axis=-1)[..., None, :]  # [B, h, 1, T]
        squared_dist = q_norm + k_norm - 2.0 * dot_product * scale
        
        # Final attention scores
        attn = squared_dot_product / (squared_dist + self.epsilon)

        # Apply mask and softmax
        attn = jnp.where(mask, attn, jnp.finfo(self.dtype).min)
        attn = jax.nn.softmax(attn).astype(self.dtype)
        attn = nn.Dropout(self.dropout_rate)(attn, deterministic=deterministic)

        # Apply attention to values and reshape
        x = jnp.einsum('bhij,bhjd->bhid', attn, v)
        x = jnp.transpose(x, (0, 2, 1, 3)).reshape(B, T, C)
        x = YatDense(C, use_bias=self.use_proj_bias, dtype=self.dtype, name='c_proj')(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return x
        
class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, deterministic=None):
        B, T, C = x.shape
        x = YatDense(4 * C, dtype=self.config.dtype, use_bias=self.config.use_bias, name='c_fc')(x)
        x = YatDense(C, dtype=self.config.dtype, use_bias=self.config.use_bias, name='c_proj')(x)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic)
        return x


class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        self.attn = SelfAttention(self.config.num_heads,
                                  self.config.dtype,
                                  dropout_rate=self.config.dropout_rate)
        self.mlp = MLP(self.config)

    def __call__(self, x, mask=None, deterministic=None):
        x = x + self.attn(x, mask, deterministic)
        x = x + self.mlp(x, deterministic)
        return x


class GPT(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, idx, deterministic=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

        pos = jnp.arange(0, T)[None]
        attn_mask = nn.make_causal_mask(idx, dtype=bool)

        wte = YatEmbed(self.config.vocab_size, self.config.num_embeds, dtype=self.config.dtype, name='wte')
        wpe = YatEmbed(self.config.block_size, self.config.num_embeds, dtype=self.config.dtype, name='wpe')

        token_embed = wte(idx)      # [B, T, num_embeds]
        pos_embed = wpe(pos)        # [1, T, num_embeds]
        x = nn.Dropout(self.config.dropout_rate)(token_embed + pos_embed, deterministic)

        for i in range(self.config.num_layers):
            x = Block(self.config, name=str(i))(x, attn_mask, deterministic=deterministic)

        # x = nn.LayerNorm(1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias, name='ln_f')(x)
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
