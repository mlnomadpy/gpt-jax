import jax
import jax.numpy as jnp
from typing import Union
import chex
import functools
import operator
from typing import Optional, Union
from functools import partial

from optax import projections



def canonicalize_axis(axis, ndim):
  """Vendored version of :func:`numpy.lib.array_utils.normalize_axis_index`.
  """
  if 0 <= (axis := operator.index(axis)) < ndim:
    return axis
  elif -ndim <= axis < 0:
    return axis + ndim
  else:
    raise ValueError(f'axis {axis} is out of bounds for array of '
                     f'dimension {ndim}')


def canonicalize_axes(axes, ndim) -> tuple[int, ...]:
  """Vendored version of :func:`numpy.lib.array_utils.normalize_axis_tuple`.
  """
  return tuple(canonicalize_axis(x, ndim) for x in axes)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def softer_max(
    x: jax.Array,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[jax.Array, None] = None,
    initial: jax.Array = -jnp.inf
) -> jax.Array:
    # Implementation of the softer_max function (same as provided earlier)
    assert jnp.all(jnp.greater_equal(x, 0)), "Input array must be non-negative"
    
    if where is not None:
        x_safe = jnp.where(where, x, 0.0)
        unnormalized = jnp.where(where, 1 + x_safe, 0.0)
    else:
        unnormalized = 1 + x

    sum_unnormalized = jnp.sum(unnormalized, axis, keepdims=True)
    result = unnormalized / sum_unnormalized
    
    if where is not None:
        result = jnp.where(where, result, 0.0)
    return result

@softer_max.defjvp
def _softer_max_jvp(axis, primals, tangents):
    (x,), (dx,) = primals, tangents

    # Compute softer_max
    y = softer_max(x, axis)

    # Compute sum of (1 + x)
    sum_unnormalized = jnp.sum(1 + x, axis=axis, keepdims=True)

    # Compute Jacobian (derivative)
    grad = y * (dx - jnp.sum(y * dx, axis=axis, keepdims=True))

    return y, grad
