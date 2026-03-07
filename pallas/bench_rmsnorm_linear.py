# Fused RMSNorm + Linear — TPU benchmark
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def _rmsnorm_linear_kernel(x_ref, w_ref, out_ref):
    x = x_ref[...].astype(jnp.float32)
    d = x.shape[-1]
    rms = jnp.sum(x * x, axis=-1, keepdims=True) / d
    x_norm = (x * jax.lax.rsqrt(rms + 1e-6)).astype(jnp.bfloat16)
    out_ref[...] = jnp.dot(x_norm, w_ref[...],
                           preferred_element_type=jnp.float32).astype(jnp.bfloat16)

def rmsnorm_linear(x, w, *, block_m=1024, block_n=3072):
    M, D = x.shape
    _, N = w.shape
    return pl.pallas_call(
        _rmsnorm_linear_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        grid=(M // block_m, N // block_n),
        in_specs=[
            pl.BlockSpec((block_m, D), lambda i, j: (i, 0)),
            pl.BlockSpec((D, block_n), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=['parallel', 'parallel'],
        ),
    )(x, w)

# Test shapes matching one microbatch attention projection
M = cfg.microbatch_size * cfg.seq_len   # 4 * 2048 = 8192
D, F = cfg.n_embd, cfg.mlp_dim          # 1024, 3072
x = jax.random.normal(jax.random.key(0), (M, D), dtype=jnp.bfloat16)
W = jax.random.normal(jax.random.key(1), (D, F), dtype=jnp.bfloat16)

# Baseline: separate rms_norm + matmul
@jax.jit
def unfused(x, w):
    return rms_norm(x.astype(jnp.float32)).astype(jnp.bfloat16) @ w

fused_jit = jax.jit(rmsnorm_linear)

flops = 2 * M * D * F  # matmul FLOPs only
hbm = (M * D + D * F + M * F) * 2  # read x, read W, write out (bf16)

print("--- Unfused (rms_norm then matmul) ---")
benchmark(unfused, x, W, flop_count=flops, hbm_bytes=hbm, label="unfused rms_norm+linear")

print("\n--- Fused Pallas kernel (1024x3072) ---")
benchmark(fused_jit, x, W, flop_count=flops, hbm_bytes=hbm, label="fused rms_norm+linear")

# Correctness check
ref = unfused(x, W)
fused = fused_jit(x, W)
abs_err = jnp.max(jnp.abs(ref - fused)).item()
mean_err = jnp.mean(jnp.abs(ref - fused)).item()
print(f"\nCorrectness: max abs err={abs_err:.2e}, mean abs err={mean_err:.2e}")
