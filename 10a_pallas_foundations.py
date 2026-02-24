# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# <a href="https://colab.research.google.com/github/vorushin/tpuchat/blob/master/10a_pallas_foundations.ipynb?flush_caches=true" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 10a — Pallas Foundations
#
# **Pallas** is JAX's kernel language for writing custom operations that run on
# TPU (and GPU). Think of it as "NumPy inside a tile" — you write a function
# that operates on small blocks of data, and `pallas_call` maps that function
# across a grid of tiles covering the full arrays.
#
# This notebook contains **6 progressive puzzles** that build your Pallas
# intuition from scratch. Every puzzle runs on **CPU** via `interpret=True` —
# no TPU needed. Fill in the kernel skeletons and run the check cells.
#
# **Prerequisites**: solid JAX/NumPy. No prior Pallas required.
#
# **Key Pallas docs**: https://docs.jax.dev/en/latest/pallas/index.html

# %% [markdown]
# ## Setup

# %%
# !pip install -q jax jaxtyping

# %%
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
print(f"JAX {jax.__version__}")


# %%
def check(kernel_fn, spec_fn, inputs, *, grid=(), in_specs=None, out_specs=None,
          out_shape=None, scratch_shapes=(), atol=1e-3, rtol=1e-3, **kwargs):
    """Run a Pallas kernel in interpret mode and compare against a reference spec.

    Args:
        kernel_fn: The Pallas kernel to test.
        spec_fn: Reference function computing the expected output in pure JAX.
        inputs: Tuple of input arrays.
        grid: Pallas grid tuple.
        in_specs: List of BlockSpec for inputs (None = no blocking).
        out_specs: BlockSpec for output (None = no blocking).
        out_shape: jax.ShapeDtypeStruct for the output.
        scratch_shapes: Scratch memory specs (empty by default).
        atol, rtol: Tolerance for comparison.
        **kwargs: Extra args to pl.pallas_call.
    """
    expected = spec_fn(*inputs)
    if out_shape is None:
        out_shape = jax.ShapeDtypeStruct(expected.shape, expected.dtype)

    # Handle default specs
    if in_specs is None:
        in_specs = [pl.BlockSpec(memory_space=pl.ANY)] * len(inputs)
    if out_specs is None:
        out_specs = pl.BlockSpec(memory_space=pl.ANY)

    actual = pl.pallas_call(
        kernel_fn,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=scratch_shapes,
        interpret=True,
        **kwargs,
    )(*inputs)

    if jnp.allclose(actual, expected, atol=atol, rtol=rtol):
        print(f"PASSED ✓  (shape={actual.shape}, dtype={actual.dtype})")
    else:
        max_err = float(jnp.max(jnp.abs(actual - expected)))
        print(f"FAILED ✗  max error: {max_err:.6f}")
        n = min(4, expected.shape[0])
        print(f"  Expected (first {n}):\n{expected[:n]}")
        print(f"  Got      (first {n}):\n{actual[:n]}")


# %% [markdown]
# ---
# ## Puzzle 1: Hello Pallas — Constant Add
#
# **Goal**: Write a kernel that adds 10 to every element.
#
# ### Theory
#
# A Pallas kernel is a Python function that receives **Ref** objects — typed
# pointers to blocks of memory. You read from a Ref with `ref[...]` (loads the
# entire block) and write with `ref[...] = value`.
#
# `pallas_call` invokes your kernel once for each point in a **grid**. With an
# empty grid `()`, the kernel runs exactly once and sees the full arrays.
#
# ```
# ┌────────────────────────┐
# │  x_ref  →  [ read ]   │
# │                ↓       │
# │           x + 10.0     │
# │                ↓       │
# │  o_ref  ←  [ write ]  │
# └────────────────────────┘
# ```

# %%
N = 32

# --- Reference (spec) ---
def add10_spec(x):
    """x: (N,) → x + 10"""
    return x + 10.0

# --- Kernel skeleton ---
def add10_kernel(x_ref, o_ref):
    # x_ref: Ref to input block (shape (N,))
    # o_ref: Ref to output block (shape (N,))
    pass  # YOUR CODE HERE


# %%
x = jax.random.uniform(jax.random.key(0), (N,))
check(add10_kernel, add10_spec, (x,))

# %% [markdown]
# <details><summary>💡 Hint</summary>
#
# Read the entire input with `x_ref[...]`, add 10, write to `o_ref[...] = ...`
# </details>

# %% [markdown]
# ---
# ## Puzzle 2: Tiled Vector Add
#
# **Goal**: Add two vectors using a 1D grid with block tiling.
#
# ### Theory
#
# When arrays are large, we split them into **blocks** and process each block
# in a separate kernel invocation. The **grid** defines how many blocks there
# are, and **BlockSpec** tells Pallas how to slice each array.
#
# ```python
# BlockSpec(block_shape, index_map)
# ```
#
# - `block_shape`: shape of the tile each invocation sees
# - `index_map`: function from grid indices → tile indices
#
# For a 1D grid: `BlockSpec((bm,), lambda i: (i,))` means "invocation `i`
# sees slice `[i*bm : (i+1)*bm]`".
#
# ```
# Array:  [████████ ████████ ████████ ████████]
#          block 0   block 1   block 2   block 3
#          grid i=0  grid i=1  grid i=2  grid i=3
# ```
#
# Inside the kernel, `pl.program_id(axis)` returns the current grid index.
# But with `BlockSpec`, the Refs already point to the right block — so
# often you don't need `program_id` at all for simple element-wise ops!

# %%
N0 = 256
# Tile sizes follow Pallas convention: bm, bk, bn
# (see https://docs.jax.dev/en/latest/pallas/tpu/matmul.html)
bm = 64

# --- Reference ---
def vadd_spec(x, y):
    """x, y: (N0,) → x + y"""
    return x + y

# --- Kernel skeleton ---
def vadd_kernel(x_ref, y_ref, o_ref):
    # Each invocation sees a (bm,) slice thanks to BlockSpec
    pass  # YOUR CODE HERE


# %%
x = jax.random.uniform(jax.random.key(1), (N0,))
y = jax.random.uniform(jax.random.key(2), (N0,))

check(vadd_kernel, vadd_spec, (x, y),
      grid=(N0 // bm,),
      in_specs=[
          pl.BlockSpec((bm,), lambda i: (i,)),
          pl.BlockSpec((bm,), lambda i: (i,)),
      ],
      out_specs=pl.BlockSpec((bm,), lambda i: (i,)))

# %% [markdown]
# <details><summary>💡 Hint</summary>
#
# The BlockSpecs handle all the slicing. Your kernel just needs:
# `o_ref[...] = x_ref[...] + y_ref[...]`
# </details>

# %% [markdown]
# ---
# ## Puzzle 3: 2D Element-wise with 2D Grid
#
# **Goal**: Multiply every element of a 2D matrix by 2, using a 2D grid of
# blocks.
#
# ### Theory
#
# Grids can be multi-dimensional. A `grid=(4, 4)` creates 16 invocations,
# each indexed by `(i, j)`. Use `pl.program_id(0)` for `i` and
# `pl.program_id(1)` for `j`.
#
# BlockSpecs for 2D: `BlockSpec((bm, bn), lambda i, j: (i, j))`
# means "tile `(i,j)` is the block at rows `[i*bm:(i+1)*bm]`,
# cols `[j*bn:(j+1)*bn]`".
#
# ```
# Matrix (128×128):
# ┌────┬────┬────┬────┐
# │0,0 │0,1 │0,2 │0,3 │  ← row blocks
# ├────┼────┼────┼────┤
# │1,0 │1,1 │1,2 │1,3 │
# ├────┼────┼────┼────┤
# │2,0 │2,1 │2,2 │2,3 │
# ├────┼────┼────┼────┤
# │3,0 │3,1 │3,2 │3,3 │
# └────┴────┴────┴────┘
#        32×32 each
# ```

# %%
M, N1 = 128, 128
bm2, bn2 = 32, 32

# --- Reference ---
def mul2d_spec(x):
    """x: (M, N1) → x * 2"""
    return x * 2.0

# --- Kernel skeleton ---
def mul2d_kernel(x_ref, o_ref):
    pass  # YOUR CODE HERE


# %%
x = jax.random.uniform(jax.random.key(3), (M, N1))
check(mul2d_kernel, mul2d_spec, (x,),
      grid=(M // bm2, N1 // bn2),
      in_specs=[pl.BlockSpec((bm2, bn2), lambda i, j: (i, j))],
      out_specs=pl.BlockSpec((bm2, bn2), lambda i, j: (i, j)))

# %% [markdown]
# <details><summary>💡 Hint</summary>
#
# Same as Puzzle 2 — `o_ref[...] = x_ref[...] * 2.0`. The 2D BlockSpec
# handles the tiling.
# </details>

# %% [markdown]
# ---
# ## Puzzle 4: Outer Product (Broadcasting Inside Kernels)
#
# **Goal**: Compute the outer product `a[:, None] * b[None, :]` for two
# vectors, producing a 2D matrix.
#
# ### Theory
#
# Inputs and output can have **different shapes**. Here:
# - `a`: shape `(M,)` → BlockSpec tiles along dim 0
# - `b`: shape `(N,)` → BlockSpec tiles along dim 0 (it's 1D)
# - `out`: shape `(M, N)` → BlockSpec tiles along both dims
#
# The index maps must line up correctly:
# - For `a`: grid `(i, j)` → tile `(i,)` (only depends on row)
# - For `b`: grid `(i, j)` → tile `(j,)` (only depends on col)
# - For `out`: grid `(i, j)` → tile `(i, j)`
#
# Inside the kernel, `a_ref` has shape `(bm,)` and `b_ref` has shape `(bn,)`.
# You need to broadcast them: `a_ref[...][:, None] * b_ref[...][None, :]`
# produces shape `(bm, bn)`.

# %%
M4, N4 = 128, 64
bm4, bn4 = 32, 32

# --- Reference ---
def outer_spec(a, b):
    """a: (M4,), b: (N4,) → (M4, N4)"""
    return a[:, None] * b[None, :]

# --- Kernel skeleton ---
def outer_kernel(a_ref, b_ref, o_ref):
    # a_ref: (bm4,) — one column-block of a
    # b_ref: (bn4,) — one row-block of b
    # o_ref: (bm4, bn4) — output tile
    pass  # YOUR CODE HERE


# %%
a = jax.random.uniform(jax.random.key(4), (M4,))
b = jax.random.uniform(jax.random.key(5), (N4,))

check(outer_kernel, outer_spec, (a, b),
      grid=(M4 // bm4, N4 // bn4),
      in_specs=[
          pl.BlockSpec((bm4,), lambda i, j: (i,)),
          pl.BlockSpec((bn4,), lambda i, j: (j,)),
      ],
      out_specs=pl.BlockSpec((bm4, bn4), lambda i, j: (i, j)),
      out_shape=jax.ShapeDtypeStruct((M4, N4), jnp.float32))

# %% [markdown]
# <details><summary>💡 Hint</summary>
#
# ```python
# o_ref[...] = a_ref[...][:, None] * b_ref[...][None, :]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 5: Reduction — Row Sum with Accumulation
#
# **Goal**: Sum each row of a matrix. The K dimension is tiled, so the
# kernel must **accumulate** partial sums across multiple invocations.
#
# ### Theory
#
# Matmul and many other operations have a **reduction dimension** (K) that
# gets summed over. In Pallas, we tile K and iterate:
#
# 1. Each grid point `(i, k)` processes row-block `i`, K-block `k`
# 2. On the first K-block (`k == 0`): **zero** the output
# 3. On every K-block: **accumulate** the partial sum
#
# ```
# x: (ROWS, COLS)
#     ┌──────┬──────┬──────┬──────┐
# r=0 │ k=0  │ k=1  │ k=2  │ k=3  │  → sum → out[0:bm5]
#     ├──────┼──────┼──────┼──────┤
# r=1 │ k=0  │ k=1  │ k=2  │ k=3  │  → sum → out[bm5:2*bm5]
#     └──────┴──────┴──────┴──────┘
# ```
#
# **Important**: `pl.program_id(axis)` tells you which K-block you're
# processing. You need `k_i == 0` to zero and `k_i == tiles_k - 1` is the
# last step (though for a simple sum, just accumulating on every step works).

# %%
ROWS, COLS = 16, 256
bm5, bk5 = 16, 64  # row block, K block
tiles_k = COLS // bk5

# --- Reference ---
def rowsum_spec(x):
    """x: (ROWS, COLS) → (ROWS,)"""
    return x.sum(axis=1)

# --- Kernel skeleton ---
def rowsum_kernel(x_ref, o_ref):
    # x_ref: (bm5, bk5) — one tile of x
    # o_ref: (bm5,) — accumulator for this row block
    # Grid: (ROWS // bm5, COLS // bk5) — iterates (row_block, k_block)
    k_i = pl.program_id(1)
    pass  # YOUR CODE HERE
    # 1. On first k tile (k_i == 0), zero the output
    # 2. Accumulate: o_ref[...] += partial row sums of x_ref


# %%
x = jax.random.uniform(jax.random.key(6), (ROWS, COLS))
check(rowsum_kernel, rowsum_spec, (x,),
      grid=(ROWS // bm5, tiles_k),
      in_specs=[pl.BlockSpec((bm5, bk5), lambda i, k: (i, k))],
      out_specs=pl.BlockSpec((bm5,), lambda i, k: (i,)),
      out_shape=jax.ShapeDtypeStruct((ROWS,), jnp.float32))

# %% [markdown]
# <details><summary>💡 Hint</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     o_ref[...] = jnp.zeros(bm5, dtype=jnp.float32)
#
# o_ref[...] += x_ref[...].sum(axis=1)
# ```
#
# `@pl.when(cond)` conditionally executes a block — like an `if` for kernels.
# </details>

# %% [markdown]
# ---
# ## Puzzle 6: Simple Matmul with Scratch Accumulator
#
# **Goal**: Implement tiled matrix multiplication `C = A @ B` using a scratch
# buffer for accumulation across K tiles.
#
# ### Theory
#
# This is the bread-and-butter of Pallas. Tiled matmul has a **3D grid**:
# `(tiles_m, tiles_n, tiles_k)`. For each `(m, n)` output tile, we iterate
# over K tiles and accumulate `A_tile @ B_tile`.
#
# The key pattern:
# 1. **Zero** scratch accumulator when `k_i == 0`
# 2. **Accumulate**: `acc += A_tile @ B_tile`
# 3. **Store** to output when `k_i == tiles_k - 1` (last K tile)
#
# We use **scratch memory** (`scratch_shapes`) for the accumulator because
# the output BlockSpec maps `(m, n, k) → (m, n)` — multiple K iterations
# write to the same output tile. Without scratch, we'd overwrite previous
# accumulations.
#
# Inside a kernel, use `jax.lax.dot(a, b)` for the matrix multiply. This
# maps to the TPU's MXU (Matrix Multiplier Unit) when running on hardware.
#
# ```
# A: (M, K)          B: (K, N)          C: (M, N)
# ┌──┬──┐            ┌──┬──┐            ┌──┬──┐
# │  │  │  bm6×bk6     │  │  │  bk6×bn6     │  │  │  bm6×bn6
# ├──┼──┤     ×      ├──┼──┤     =      ├──┼──┤
# │  │  │            │  │  │            │  │  │
# └──┴──┘            └──┴──┘            └──┴──┘
#
# For each (m_i, n_i): acc = Σ_k  A[m_i, k] @ B[k, n_i]
# ```

# %%
M6, K6, N6 = 128, 256, 128
bm6, bk6, bn6 = 64, 128, 64
tiles_m = M6 // bm6
tiles_n = N6 // bn6
tiles_k = K6 // bk6

# --- Reference ---
def matmul_spec(a, b):
    """a: (M6, K6), b: (K6, N6) → (M6, N6)"""
    return a @ b

# --- Kernel skeleton ---
def matmul_kernel(a_ref, b_ref, o_ref, acc_ref):
    # a_ref: (bm6, bk6) — tile of A
    # b_ref: (bk6, bn6) — tile of B
    # o_ref: (bm6, bn6) — output tile
    # acc_ref: (bm6, bn6) — scratch accumulator (VMEM on TPU)
    k_i = pl.program_id(2)
    pass  # YOUR CODE HERE
    # 1. Zero acc_ref when k_i == 0
    # 2. Accumulate: acc_ref[...] += a_tile @ b_tile
    # 3. Store acc_ref → o_ref when k_i == tiles_k - 1


# %%
a = jax.random.normal(jax.random.key(7), (M6, K6))
b = jax.random.normal(jax.random.key(8), (K6, N6))

check(matmul_kernel, matmul_spec, (a, b),
      grid=(tiles_m, tiles_n, tiles_k),
      in_specs=[
          pl.BlockSpec((bm6, bk6), lambda m, n, k: (m, k)),
          pl.BlockSpec((bk6, bn6), lambda m, n, k: (k, n)),
      ],
      out_specs=pl.BlockSpec((bm6, bn6), lambda m, n, k: (m, n)),
      out_shape=jax.ShapeDtypeStruct((M6, N6), jnp.float32),
      scratch_shapes=[pltpu.VMEM((bm6, bn6), jnp.float32)])

# %% [markdown]
# <details><summary>💡 Hint</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     acc_ref[...] = jnp.zeros((bm6, bn6), dtype=jnp.float32)
#
# acc_ref[...] += jax.lax.dot(a_ref[...], b_ref[...])
#
# @pl.when(k_i == tiles_k - 1)
# def _store():
#     o_ref[...] = acc_ref[...]
# ```
# </details>

# %% [markdown]
# ---
# ## Summary
#
# You've learned the core Pallas building blocks:
#
# | Concept | Puzzle |
# |---------|--------|
# | `pallas_call`, Refs, `interpret=True` | 1 |
# | `grid`, `BlockSpec`, `program_id` | 2 |
# | 2D grids and tiling | 3 |
# | Broadcasting inside kernels | 4 |
# | Reduction accumulation, `@pl.when` | 5 |
# | Matmul with scratch accumulator | 6 |
#
# **Next**: `10b_pallas_intermediate.py` — scalar prefetch, indirect indexing,
# group metadata, and masked stores.
