@partial(jax.custom_jvp, nondiff_argnums=(1,))
def softer_max(
    x: ArrayLike,
    axis: int | tuple[int, ...] | None = -1,
    where: ArrayLike | None = None,
    initial: ArrayLike | None = -jnp.inf) -> jax.Array:
    """
    A modified softmax that uses (1 + x) instead of exp(x).
    
    Args:
        x: Input array (must be non-negative)
        axis: Axis along which to compute softer_max
        where: Optional boolean mask
        initial: Value to use for masked elements
    
    Returns:
        Array with softer_max applied
    """
    # Ensure x is non-negative
    assert jnp.all(x >= 0), "Input array must be non-negative"

    # Safe handling of masked elements
    x_safe = x if where is None else jnp.where(where, x, 0.0)

    # Compute unnormalized values (1 + x instead of exp(x))
    unnormalized = 1 + x_safe

    # Compute sum
    sum_unnormalized = jnp.sum(unnormalized, axis=axis, keepdims=True)
    
    # Normalize
    result = unnormalized / sum_unnormalized

    # Apply mask if provided
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
