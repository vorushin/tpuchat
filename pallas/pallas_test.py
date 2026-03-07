"""Local CPU correctness tests for Pallas kernels (interpret mode).

Usage:
    uv run python pallas/pallas_test.py                  # run all tests
    uv run python pallas/pallas_test.py rmsnorm_linear    # run one test
"""

import sys

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# Force CPU backend for local testing
jax.config.update("jax_platform_name", "cpu")

# Test dimensions — small but preserving model ratios
D = 128    # n_embd
N = 2      # n_head
H = 64     # head_dim (D / N)
F = 384    # mlp_dim (~3x D, as in SwiGLU)
B = 2      # batch
T = 32     # seq_len


def rms_norm(x):
    """Reference RMSNorm (no learnable params)."""
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)


# === Fused RMSNorm + Linear kernel ===

def _rmsnorm_linear_kernel(x_ref, w_ref, out_ref):
    """Kernel body: RMSNorm(x) @ W for one output tile.

    x_ref:   (block_m, D)   — full D needed for norm reduction
    w_ref:   (D, block_n)   — weight slice for this output tile
    out_ref: (block_m, block_n)
    """
    x = x_ref[...].astype(jnp.float32)
    d = x.shape[-1]
    rms = jnp.sum(x * x, axis=-1, keepdims=True) / d
    x_norm = (x * jax.lax.rsqrt(rms + 1e-6)).astype(jnp.bfloat16)
    out_ref[...] = jnp.dot(x_norm, w_ref[...],
                           preferred_element_type=jnp.float32).astype(jnp.bfloat16)


def rmsnorm_linear(x, w, *, block_m=128, block_n=128, interpret=False):
    """Fused RMSNorm(x) @ W.

    x: (M, D) bfloat16 — flattened (B*T, D)
    w: (D, N) bfloat16 — weight matrix
    returns: (M, N) bfloat16
    """
    M, D = x.shape
    _, out_N = w.shape
    return pl.pallas_call(
        _rmsnorm_linear_kernel,
        out_shape=jax.ShapeDtypeStruct((M, out_N), jnp.bfloat16),
        grid=(M // block_m, out_N // block_n),
        in_specs=[
            pl.BlockSpec((block_m, D), lambda i, j: (i, 0)),
            pl.BlockSpec((D, block_n), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        interpret=interpret,
    )(x, w)


# === Tests ===

def test_rmsnorm_linear():
    """RMSNorm + matmul fusion: rms_norm(x) @ W."""
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    M = B * T
    x = jax.random.normal(k1, (M, D), dtype=jnp.bfloat16)
    w = jax.random.normal(k2, (D, F), dtype=jnp.bfloat16)

    # Reference: float32 norm -> bf16 -> matmul with f32 accumulator (matches kernel/TPU)
    x_norm = rms_norm(x.astype(jnp.float32)).astype(jnp.bfloat16)
    ref = jnp.dot(x_norm, w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)

    # Fused kernel (block_m=16 divides M=64, block_n=128 divides F=384)
    fused = rmsnorm_linear(x, w, block_m=16, block_n=128, interpret=True)

    return compare(ref, fused)


def test_relu2_down():
    """ReLU^2 + down-projection fusion: (relu(x)^2) @ W_down."""
    key = jax.random.key(1)
    k1, k2 = jax.random.split(key)
    M = B * T
    x = jax.random.normal(k1, (M, F), dtype=jnp.bfloat16)
    w_down = jax.random.normal(k2, (F, D), dtype=jnp.bfloat16)

    ref = (jax.nn.relu(x) ** 2) @ w_down

    # Fused kernel — NOT IMPLEMENTED
    return None, "NOT IMPLEMENTED"


def compare(ref, actual, name=""):
    """Compare two arrays, return (passed, status_string)."""
    abs_err = jnp.max(jnp.abs(ref - actual)).item()
    denom = jnp.maximum(jnp.abs(ref), 1e-8)
    rel_err = jnp.max(jnp.abs(ref - actual) / denom).item()
    passed = abs_err < 1e-2  # bfloat16-friendly tolerance
    status = "PASS" if passed else "FAIL"
    return passed, f"{status}  shape={list(ref.shape)}  abs_err={abs_err:.2e}  rel_err={rel_err:.2e}"


ALL_TESTS = {
    "rmsnorm_linear": test_rmsnorm_linear,
    "relu2_down": test_relu2_down,
}


def main():
    if len(sys.argv) > 1:
        names = sys.argv[1:]
        tests = {n: ALL_TESTS[n] for n in names if n in ALL_TESTS}
        for n in names:
            if n not in ALL_TESTS:
                print(f"Unknown test: {n}")
                print(f"Available: {', '.join(ALL_TESTS)}")
                sys.exit(1)
    else:
        tests = ALL_TESTS

    print(f"=== Pallas Kernel Tests (CPU interpret mode) ===")
    print(f"Test dims: B={B}, T={T}, D={D}, N={N}, H={H}, F={F}\n")

    all_passed = True
    for name, fn in tests.items():
        passed, status = fn()
        if passed is None:
            print(f"  {name:<25s} {status}")
        else:
            print(f"  {name:<25s} {status}")
            if not passed:
                all_passed = False

    print()
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
