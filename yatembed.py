from typing import (
  Any,
)
from collections.abc import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import eval_shape, lax
from jax.core import ShapedArray

import opt_einsum

from flax.core import meta
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen import module
from flax.linen.module import Module, compact
from flax.typing import (
  Array,
  PRNGKey as PRNGKey,
  Dtype,
  Shape as Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
  ConvGeneralDilatedT,
  PaddingLike,
  LaxPadding,
)
from flax import linen as nn

class YatEmbed(Module):
    """Embedding Module.

    Attributes:
    num_embeddings: number of embeddings / vocab size.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: same as embedding).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    embedding_init: embedding initializer.
    """

    num_embeddings: int
    features: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    embedding_init: Initializer = initializers.xavier_normal()

    def setup(self):
        self.embedding = self.param(
            'embedding',
            self.embedding_init,
            (self.num_embeddings, self.features),
            self.param_dtype,
        )

    def __call__(self, inputs: Array) -> Array:
        """Embeds the inputs along the last dimension.

        Args:
            inputs: input data, all dimensions are considered batch dimensions.
            Values in the input array must be integers.

        Returns:
            Output which is embedded input data.  The output shape follows the input,
            with an additional ``features`` dimension appended.
        """
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError('Input type must be an integer or unsigned integer.')
        # Use take because fancy indexing numpy arrays with JAX indices does not
        # work correctly.
        (embedding,) = promote_dtype(
            self.embedding, dtype=self.dtype, inexact=False
        )
        if self.num_embeddings == 1:
            return jnp.where(
            jnp.broadcast_to(inputs[..., None], inputs.shape + (self.features,))
            == 0,
            embedding,
            jnp.nan,
            )
        return jnp.take(embedding, inputs, axis=0)

    def attend(self, query: Array) -> Array:
        """Attend over the embedding using a query array with squared Euclidean distance transformation.
        
        Args:
            query: array with last dimension equal the feature depth ``features`` of the
                embedding.
        Returns:
            An array with final dim ``num_embeddings`` corresponding to the transformed
            similarity between query vectors and embeddings, using squared Euclidean
            distance-based attention.
        """
        query, embedding = promote_dtype(query, self.embedding, dtype=self.dtype)
        
        # Compute dot product between query and embedding
        y = jnp.dot(query, embedding.T)
        
        # Compute squared Euclidean distances components
        query_squared_sum = jnp.sum(query**2, axis=-1, keepdims=True)
        embedding_squared_sum = jnp.sum(embedding**2, axis=-1)
        distances = query_squared_sum + embedding_squared_sum - 2 * y
        
        epsilon = 1/137
        y = y**2 / (distances + epsilon)
        
        return y 