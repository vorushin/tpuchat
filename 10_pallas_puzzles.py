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
# <a href="https://colab.research.google.com/github/vorushin/tpuchat/blob/master/10_pallas_puzzles.ipynb?flush_caches=true" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 10 — Pallas Puzzles: From First Kernel to Ragged Dot
#
# **Pallas** is JAX's kernel language for writing custom operations that run on
# TPU (and GPU). Think of it as "NumPy inside a tile" — you write a function
# that operates on small blocks of data, and `pallas_call` maps that function
# across a grid of tiles covering the full arrays.
#
# This notebook contains **20 progressive exercises** (some with sub-parts)
# that build your Pallas intuition from scratch and culminate in a full
# **ragged_dot** kernel for Mixture-of-Experts. Every puzzle runs on **CPU**
# via `interpret=True` — no TPU needed. Fill in the kernel skeletons and
# run the check cells.
#
# **Prerequisites**: solid JAX/NumPy. No prior Pallas required.
#
# **Key Pallas docs**: https://docs.jax.dev/en/latest/pallas/index.html
#
# | Part | Puzzles | Focus |
# |------|---------|-------|
# | I — Foundations | 1–6 | Refs, grids, BlockSpec, `@pl.when` |
# | II — Matmul Patterns | 7–10 | Scratch, accumulation, fusion |
# | III — Scalar Prefetch | 11–15 | Runtime index maps, group metadata, masking |
# | IV — Ragged Dot | 16–20 | Grouped matmul, tgmm, pipelining |

# %% [markdown]
# ## Setup

# %%
# !pip install -q jax jaxtyping

# %%
import functools
import jax
import jax.numpy as jnp
from jax import lax
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
        diff = jnp.abs(actual - expected)
        max_err = float(jnp.max(diff))
        worst_idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)
        print(f"FAILED ✗  max error: {max_err:.6f} at index {tuple(int(i) for i in worst_idx)}")
        n = min(4, expected.shape[0])
        print(f"  Expected (first {n}):\n{expected[:n]}")
        print(f"  Got      (first {n}):\n{actual[:n]}")


# %% [markdown]
# ---
# # Part I: Foundations (Puzzles 1–6)

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
# entire block) and write with `ref[...] = value`. The `[...]` (Ellipsis)
# means "all elements" — it's the standard way to read or write an entire Ref
# in Pallas. You can also use slicing like `ref[0:4]`, but full `ref[...]`
# reads/writes are by far the most common pattern.
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
N1 = 32

# --- Reference (spec) ---
def add10_spec(x):
    """x: (N1,) → x + 10"""
    return x + 10.0

# --- Kernel skeleton ---
def add10_kernel(x_ref, o_ref):
    # x_ref: Ref to input block (shape (N1,))
    # o_ref: Ref to output block (shape (N1,))
    pass  # YOUR CODE HERE


# %%
x1 = jax.random.uniform(jax.random.key(0), (N1,))
check(add10_kernel, add10_spec, (x1,))

# %% [markdown]
# <details><summary>Hint</summary>
#
# Read the entire input with `x_ref[...]`, add 10, write to `o_ref[...] = ...`
# </details>

# %% [markdown]
# ---
# ## Puzzle 2a: Tiled Vector Add
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
# you often don't need `program_id` at all for element-wise ops!
# The kernel body stays identical whether you have 4 blocks or 400.

# %%
N2 = 256   # vector length
bm2 = 64   # tile (block) size — each kernel invocation processes bm2 elements

# --- Reference ---
def vadd_spec(x, y):
    """x, y: (N2,) → x + y"""
    return x + y

# --- Kernel skeleton ---
def vadd_kernel(x_ref, y_ref, o_ref):
    # Each invocation sees a (bm2,) slice thanks to BlockSpec
    pass  # YOUR CODE HERE


# %%
x2 = jax.random.uniform(jax.random.key(1), (N2,))
y2 = jax.random.uniform(jax.random.key(2), (N2,))

check(vadd_kernel, vadd_spec, (x2, y2),
      grid=(N2 // bm2,),              # 256 // 64 = 4 invocations
      in_specs=[
          pl.BlockSpec((bm2,), lambda i: (i,)),  # x: invocation i → block i
          pl.BlockSpec((bm2,), lambda i: (i,)),  # y: invocation i → block i
      ],
      out_specs=pl.BlockSpec((bm2,), lambda i: (i,)))  # out: invocation i → block i

# %% [markdown]
# <details><summary>Hint</summary>
#
# The BlockSpecs handle all the slicing. Your kernel just needs:
# `o_ref[...] = x_ref[...] + y_ref[...]`
# </details>

# %% [markdown]
# ---
# ## Puzzle 2b: Reversed Block Add — Index Map Manipulation
#
# **Goal**: Add `x` to a **block-reversed** version of `y` by changing
# only the index map. The kernel body is identical to Puzzle 2a!
#
# ### Theory
#
# The index map in a `BlockSpec` controls **which block** each grid
# invocation sees. So far every index map was `lambda i: (i,)` — grid
# invocation `i` sees block `i` (sequential order). But the map can
# be any function: `lambda i: (3 - i,)` would read blocks in reverse.
#
# ```
# y = [  y₀  |  y₁  |  y₂  |  y₃  ]      4 blocks, bm=64
#
# Normal index map     λi: (i,)         → y₀  y₁  y₂  y₃
# Reversed index map   λi: (3-i,)       → y₃  y₂  y₁  y₀
#
# x:          [  x₀  ][  x₁  ][  x₂  ][  x₃  ]
# y reversed: [  y₃  ][  y₂  ][  y₁  ][  y₀  ]
# result:     [x₀+y₃ ][x₁+y₂ ][x₂+y₁ ][x₃+y₀]
# ```
#
# This is the key insight behind all advanced Pallas kernels: the index
# map decides what data the kernel sees, while the kernel body stays
# simple and generic.

# %%
N2a = 256   # vector length (same as Puzzle 2a)
bm2a = 64   # tile size
num_blocks_2a = N2a // bm2a   # 4 blocks total

# --- Reference ---
def vadd_rev_spec(x, y):
    """x, y: (N2a,) → x + block_reverse(y)"""
    y_rev = y.reshape(num_blocks_2a, bm2a)[::-1].reshape(N2a)
    return x + y_rev

# Kernel is provided (same body as Puzzle 2a):
def vadd_rev_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


# %%
x2a = jax.random.uniform(jax.random.key(100), (N2a,))
y2a = jax.random.uniform(jax.random.key(101), (N2a,))

# YOUR TASK: Fix the y BlockSpec so it reads blocks in reversed order.
# Only the y index map needs to change — x and out are correct.
check(vadd_rev_kernel, vadd_rev_spec, (x2a, y2a),
      grid=(num_blocks_2a,),
      in_specs=[
          pl.BlockSpec((bm2a,), lambda i: (i,)),              # x: block i (correct)
          pl.BlockSpec((bm2a,), lambda i: (i,)),              # y: block i — FIX THIS
      ],
      out_specs=pl.BlockSpec((bm2a,), lambda i: (i,)))

# %% [markdown]
# <details><summary>Hint</summary>
#
# The y index map should map grid index `i` to the reversed block position.
# With 4 blocks, `i=0 → block 3`, `i=1 → block 2`, etc.:
# ```python
# pl.BlockSpec((bm2a,), lambda i: (num_blocks_2a - 1 - i,))
# ```
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
# **Key insight**: The kernel body is identical to Puzzle 2a — just
# `o_ref[...] = f(x_ref[...])`. The BlockSpec handles all the 2D
# indexing. This is the power of Pallas's tiling abstraction: the
# kernel doesn't care whether the grid is 1D, 2D, or 3D.
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
M3, N3 = 128, 128
bm3, bn3 = 32, 32

# --- Reference ---
def mul2d_spec(x):
    """x: (M3, N3) → x * 2"""
    return x * 2.0

# --- Kernel skeleton ---
def mul2d_kernel(x_ref, o_ref):
    pass  # YOUR CODE HERE


# %%
x3 = jax.random.uniform(jax.random.key(3), (M3, N3))
check(mul2d_kernel, mul2d_spec, (x3,),
      grid=(M3 // bm3, N3 // bn3),
      in_specs=[pl.BlockSpec((bm3, bn3), lambda i, j: (i, j))],
      out_specs=pl.BlockSpec((bm3, bn3), lambda i, j: (i, j)))

# %% [markdown]
# <details><summary>Hint</summary>
#
# Same as Puzzle 2a — `o_ref[...] = x_ref[...] * 2.0`. The 2D BlockSpec
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
    # a_ref: (bm4,) — a slice of vector a
    # b_ref: (bn4,) — a slice of vector b
    # o_ref: (bm4, bn4) — output tile
    pass  # YOUR CODE HERE


# %%
a4 = jax.random.uniform(jax.random.key(4), (M4,))
b4 = jax.random.uniform(jax.random.key(5), (N4,))

check(outer_kernel, outer_spec, (a4, b4),
      grid=(M4 // bm4, N4 // bn4),
      in_specs=[
          pl.BlockSpec((bm4,), lambda i, j: (i,)),
          pl.BlockSpec((bn4,), lambda i, j: (j,)),
      ],
      out_specs=pl.BlockSpec((bm4, bn4), lambda i, j: (i, j)),
      out_shape=jax.ShapeDtypeStruct((M4, N4), jnp.float32))

# %% [markdown]
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# You need to broadcast `a_ref[...]` (shape `(bm4,)`) and `b_ref[...]` (shape `(bn4,)`) to produce shape `(bm4, bn4)`. Use NumPy-style broadcasting: add a new axis with `[:, None]` and `[None, :]`.
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# o_ref[...] = a_ref[...][:, None] * b_ref[...][None, :]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 5: Configure Your Own `pallas_call` — Vector Add
#
# **Goal**: Given a working kernel, fill in the `grid`, `in_specs`, and
# `out_specs` arguments yourself.
#
# ### Theory
#
# So far we've given you the `pallas_call` setup and you only wrote the
# kernel body. Now it's your turn to configure the call. You need:
#
# 1. **`grid`**: a tuple specifying how many tiles in each dimension.
#    For a 1D vector of length `N` with tile size `bm`: `grid = (N // bm,)`.
#
# 2. **`in_specs`**: a list of `BlockSpec`, one per input. Each says what
#    shape the kernel sees and how grid indices map to tile positions.
#
# 3. **`out_specs`**: a single `BlockSpec` for the output.
#
# The kernel below is the solved version from Puzzle 2a. Your task is to
# wire up the tiling so it processes `N5`-element vectors in blocks of
# `bm5`.

# %%
N5 = 256
bm5 = 64

def vadd_spec5(x, y):
    return x + y

# Kernel is provided (solved):
def vadd_kernel_solved(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


# %%
x5 = jax.random.uniform(jax.random.key(10), (N5,))
y5 = jax.random.uniform(jax.random.key(11), (N5,))

# YOUR TASK: Define grid, in_specs, out_specs to tile the computation
# into bm5-sized blocks. The kernel processes one block per invocation.
vadd_grid = ...       # TODO: how many tiles? (should be a tuple)
vadd_in_specs = ...   # TODO: list of BlockSpec, one per input
vadd_out_specs = ...  # TODO: BlockSpec for output

check(vadd_kernel_solved, vadd_spec5, (x5, y5),
      grid=vadd_grid,
      in_specs=vadd_in_specs,
      out_specs=vadd_out_specs)

# %% [markdown]
# <details><summary>Hint 1 of 2 — What to fill in</summary>
#
# ```python
# vadd_grid = (N5 // bm5,)  # 256 // 64 = 4 tiles
# vadd_in_specs = [
#     pl.BlockSpec((bm5,), lambda i: (i,)),  # one per input
#     pl.BlockSpec((bm5,), lambda i: (i,)),
# ]
# vadd_out_specs = pl.BlockSpec((bm5,), lambda i: (i,))
# ```
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# vadd_grid = (N5 // bm5,)
# vadd_in_specs = [
#     pl.BlockSpec((bm5,), lambda i: (i,)),
#     pl.BlockSpec((bm5,), lambda i: (i,)),
# ]
# vadd_out_specs = pl.BlockSpec((bm5,), lambda i: (i,))
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 6: Reduction — Row Sum with `@pl.when`
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
# r=0 │ k=0  │ k=1  │ k=2  │ k=3  │  → sum → out[0:bm]
#     ├──────┼──────┼──────┼──────┤
# r=1 │ k=0  │ k=1  │ k=2  │ k=3  │  → sum → out[bm:2*bm]
#     └──────┴──────┴──────┴──────┘
# ```
#
# **`@pl.when(condition)`** is Pallas's conditional execution primitive.
# It compiles to **predicated execution** on TPU — no branch divergence
# penalty. Use it to guard operations that should only run on certain
# grid iterations:
#
# ```python
# @pl.when(k_i == 0)           # only runs when k_i is 0
# def _():
#     acc[...] = jnp.zeros(...)
# ```
#
# This is the key pattern for all reduction kernels: conditionally zero
# the accumulator on the first tile, accumulate on every tile, and
# (for matmul) conditionally store on the last tile.

# %%
ROWS6, COLS6 = 16, 256
bm6, bk6 = 16, 64
tiles_k6 = COLS6 // bk6

# --- Reference ---
def rowsum_spec(x):
    """x: (ROWS6, COLS6) → (ROWS6,)"""
    return x.sum(axis=1)

# --- Kernel skeleton ---
def rowsum_kernel(x_ref, o_ref):
    # x_ref: (bm6, bk6) — one tile of x
    # o_ref: (bm6,) — accumulator for this row block
    # Grid: (ROWS6 // bm6, COLS6 // bk6) — iterates (row_block, k_block)
    k_i = pl.program_id(1)
    pass  # YOUR CODE HERE
    # 1. On first k tile (k_i == 0), initialize the output
    # 2. Add this tile's contribution to the running sum


# %%
x6 = jax.random.uniform(jax.random.key(6), (ROWS6, COLS6))
check(rowsum_kernel, rowsum_spec, (x6,),
      grid=(ROWS6 // bm6, tiles_k6),
      in_specs=[pl.BlockSpec((bm6, bk6), lambda i, k: (i, k))],
      out_specs=pl.BlockSpec((bm6,), lambda i, k: (i,)),
      out_shape=jax.ShapeDtypeStruct((ROWS6,), jnp.float32))

# %% [markdown]
# <details><summary>Hint 1 of 3 — Approach</summary>
#
# Use `@pl.when(k_i == 0)` to conditionally zero the output on the first K tile. On every tile, accumulate the partial row sum with `o_ref[...] += x_ref[...].sum(axis=1)`.
# </details>
#
# <details><summary>Hint 2 of 3 — Pattern skeleton</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     o_ref[...] = jnp.zeros((bm6,), dtype=jnp.float32)
#
# o_ref[...] += ...  # partial row sum of x_ref
# ```
# </details>
#
# <details><summary>Hint 3 of 3 — Full solution</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     o_ref[...] = jnp.zeros((bm6,), dtype=jnp.float32)
#
# o_ref[...] += x_ref[...].sum(axis=1)
# ```
# </details>

# %% [markdown]
# ---
# # Part II: Matmul Patterns (Puzzles 7–10)

# %% [markdown]
# ---
# ## Puzzle 7: Tiled Matmul with Scratch Accumulator
#
# **Goal**: Implement tiled matrix multiplication `C = A @ B` using a scratch
# buffer for accumulation across K tiles.
#
# ### Theory
#
# This is the bread-and-butter of Pallas. Tiled matmul has a **3D grid**:
# `(tiles_m, tiles_n, tiles_k)`. For each `(m, n)` output tile, we iterate
# over K tiles (K for "Kontracting" dimension) and accumulate
# `A_tile @ B_tile`.
#
# We use **scratch memory** (`scratch_shapes`) for the accumulator.
# Scratch is allocated in **VMEM** — TPU's fast on-chip SRAM (like shared
# memory on GPU). Why not just accumulate directly in `o_ref`? Two reasons:
# 1. **Performance**: `o_ref` points to HBM. Reading and writing it on
#    every K iteration means round-trips to slow off-chip memory.
#    Scratch stays in fast VMEM throughout all K iterations.
# 2. **Correctness**: The output BlockSpec maps `(m, n, k) → (m, n)` —
#    multiple K iterations target the same output tile. Without a local
#    accumulator, K iteration 1 would overwrite K iteration 0's result.
#
# Specify scratch with `pltpu.VMEM(shape, dtype)`.
#
# **`out_shape`** tells `pallas_call` the shape and dtype of the output
# array to allocate. It's a `jax.ShapeDtypeStruct` — just metadata, no
# actual data:
# ```python
# out_shape = jax.ShapeDtypeStruct((M, N), jnp.float32)
# ```
# (In earlier puzzles, the `check` helper inferred this automatically
# from the reference output. From here on, you'll see it explicitly.)
#
# Inside a kernel, use `a @ b` (or equivalently `jax.lax.dot(a, b)`) for
# the matrix multiply. Both map to the TPU's MXU (Matrix Multiplier Unit).
#
# The production-ready pattern uses `@pl.when` guards:
# ```python
# @pl.when(k_i == 0)           # ZERO on first K tile
# def _(): acc[...] = zeros
#
# acc[...] += a @ b             # ACCUMULATE on every tile
#
# @pl.when(k_i == tiles_k - 1) # STORE on last K tile
# def _(): out[...] = acc[...]
# ```
#
# On TPU hardware, `@pl.when` compiles to predicated execution — no branch
# divergence penalty. This zero/accumulate/store pattern is used in every
# production Pallas kernel.
#
# ```
# A: (M, K)          B: (K, N)          C: (M, N)
# ┌──┬──┐            ┌──┬──┐            ┌──┬──┐
# │  │  │  bm×bk     │  │  │  bk×bn     │  │  │  bm×bn
# ├──┼──┤     ×      ├──┼──┤     =      ├──┼──┤
# │  │  │            │  │  │            │  │  │
# └──┴──┘            └──┴──┘            └──┴──┘
#
# For each (m_i, n_i): acc = Σ_k  A[m_i, k] @ B[k, n_i]
# ```

# %%
M7, K7, N7 = 128, 256, 128
bm7, bk7, bn7 = 64, 128, 64
tiles_m7 = M7 // bm7
tiles_n7 = N7 // bn7
tiles_k7 = K7 // bk7

# --- Reference ---
def matmul_spec(a, b):
    """a: (M7, K7), b: (K7, N7) → (M7, N7)"""
    return a @ b

# --- Kernel skeleton ---
def matmul_kernel(a_ref, b_ref, o_ref, acc_ref):
    # a_ref: (bm7, bk7) — tile of A
    # b_ref: (bk7, bn7) — tile of B
    # o_ref: (bm7, bn7) — output tile
    # acc_ref: (bm7, bn7) — scratch accumulator (VMEM on TPU)
    k_i = pl.program_id(2)
    pass  # YOUR CODE HERE
    # 1. Zero acc_ref when k_i == 0
    # 2. Accumulate: acc_ref[...] += a_ref[...] @ b_ref[...]
    # 3. Store acc_ref → o_ref when k_i == tiles_k7 - 1


# %%
a7 = jax.random.normal(jax.random.key(7), (M7, K7))
b7 = jax.random.normal(jax.random.key(8), (K7, N7))

check(matmul_kernel, matmul_spec, (a7, b7),
      grid=(tiles_m7, tiles_n7, tiles_k7),
      in_specs=[
          pl.BlockSpec((bm7, bk7), lambda m, n, k: (m, k)),
          pl.BlockSpec((bk7, bn7), lambda m, n, k: (k, n)),
      ],
      out_specs=pl.BlockSpec((bm7, bn7), lambda m, n, k: (m, n)),
      out_shape=jax.ShapeDtypeStruct((M7, N7), jnp.float32),
      scratch_shapes=[pltpu.VMEM((bm7, bn7), jnp.float32)])

# %% [markdown]
# <details><summary>Hint 1 of 2 — Pattern skeleton</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     acc_ref[...] = jnp.zeros((bm7, bn7), dtype=jnp.float32)
#
# acc_ref[...] += ...  # A_tile @ B_tile
#
# @pl.when(k_i == tiles_k7 - 1)
# def _store():
#     o_ref[...] = acc_ref[...]
# ```
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     acc_ref[...] = jnp.zeros((bm7, bn7), dtype=jnp.float32)
#
# acc_ref[...] += a_ref[...] @ b_ref[...]
#
# @pl.when(k_i == tiles_k7 - 1)
# def _store():
#     o_ref[...] = acc_ref[...]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 8: Configure Your Own Matmul `pallas_call`
#
# **Goal**: Given a working matmul kernel, fill in **all** the `pallas_call`
# arguments: `grid`, `in_specs`, `out_specs`, `out_shape`, and
# `scratch_shapes`.
#
# ### Theory
#
# This is the next step in learning to configure `pallas_call` yourself.
# Unlike Puzzle 5 (1D vector add), matmul has a **3D grid** and requires
# scratch memory. You need to understand how `BlockSpec` index maps route
# tiles in a 3D grid:
#
# - `A` tile `(m, k)` is at `A[m*bm:(m+1)*bm, k*bk:(k+1)*bk]`
#   → index map: `lambda m, n, k: (m, k)`
# - `B` tile `(k, n)` is at `B[k*bk:(k+1)*bk, n*bn:(n+1)*bn]`
#   → index map: `lambda m, n, k: (k, n)`
# - `C` tile `(m, n)` is at `C[m*bm:(m+1)*bm, n*bn:(n+1)*bn]`
#   → index map: `lambda m, n, k: (m, n)` (no K dependency!)
#
# Don't forget `out_shape` (the full output shape, not the tile shape)
# and `scratch_shapes` (the VMEM accumulator from Puzzle 7).

# %%
M8, K8, N8 = 128, 256, 128
bm8, bk8, bn8 = 64, 128, 64
tiles_m8 = M8 // bm8
tiles_n8 = N8 // bn8
tiles_k8 = K8 // bk8

def matmul_spec8(a, b):
    return a @ b

# Kernel is provided (solved — same pattern as Puzzle 7):
def matmul_kernel_solved(a_ref, b_ref, o_ref, acc_ref):
    k_i = pl.program_id(2)
    @pl.when(k_i == 0)
    def _zero():
        acc_ref[...] = jnp.zeros((bm8, bn8), dtype=jnp.float32)
    acc_ref[...] += a_ref[...] @ b_ref[...]
    @pl.when(k_i == tiles_k8 - 1)
    def _store():
        o_ref[...] = acc_ref[...]


# %%
a8 = jax.random.normal(jax.random.key(20), (M8, K8))
b8 = jax.random.normal(jax.random.key(21), (K8, N8))

# YOUR TASK: Replace ALL arguments with correct values.
check(matmul_kernel_solved, matmul_spec8, (a8, b8),
      grid=(),                   # FIX THIS — 3D grid (tiles_m, tiles_n, tiles_k)
      in_specs=None,             # FIX THIS — BlockSpec for A and B
      out_specs=None,            # FIX THIS — BlockSpec for C
      out_shape=None,            # FIX THIS — output ShapeDtypeStruct
      scratch_shapes=())         # FIX THIS — VMEM scratch for accumulator

# %% [markdown]
# <details><summary>Hint 1 of 2 — What to fill in</summary>
#
# You need five things:
# - `grid = (tiles_m8, tiles_n8, tiles_k8)`
# - `in_specs` with two BlockSpecs: A maps `(m,n,k)→(m,k)`, B maps `(m,n,k)→(k,n)`
# - `out_specs` maps `(m,n,k)→(m,n)`
# - `out_shape = jax.ShapeDtypeStruct((M8, N8), jnp.float32)`
# - `scratch_shapes = [pltpu.VMEM((bm8, bn8), jnp.float32)]`
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# check(matmul_kernel_solved, matmul_spec8, (a8, b8),
#       grid=(tiles_m8, tiles_n8, tiles_k8),
#       in_specs=[
#           pl.BlockSpec((bm8, bk8), lambda m, n, k: (m, k)),
#           pl.BlockSpec((bk8, bn8), lambda m, n, k: (k, n)),
#       ],
#       out_specs=pl.BlockSpec((bm8, bn8), lambda m, n, k: (m, n)),
#       out_shape=jax.ShapeDtypeStruct((M8, N8), jnp.float32),
#       scratch_shapes=[pltpu.VMEM((bm8, bn8), jnp.float32)])
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 9: Batched Matmul — Batch Dimension on RHS
#
# **Goal**: Compute `out[g] = lhs[g] @ rhs[g]` for `G` independent batches.
# The RHS has a leading batch dimension.
#
# ### Theory
#
# In batched matmul, the RHS is `(G, K, N)` — a stack of `G` weight
# matrices. Each batch element `g` has its own `(K, N)` matrix.
#
# (We use `G` for the batch size here because in Part IV, this same
# structure will represent **groups** in ragged_dot.)
#
# The grid adds a **batch dimension**: `grid = (G,)`.
# Each iteration's BlockSpec selects one batch element at a time.
#
# **`None` vs integer in block_shape**: Using `None` means "load the entire
# axis and **squeeze** that dimension". The ref will NOT have that dim.
# Using an integer (e.g. `1`) means "load 1 element" — the ref keeps that
# dim with size 1.
#
# For a batch dim, `None` is convenient — the kernel sees simple 2D
# shapes like `(M, K)` instead of `(1, M, K)`:
# ```
# BlockSpec((None, M, K), lambda g: (g, 0, 0))
#            ^^^^
#            squeezed — ref shape is (M, K), not (1, M, K)
# ```
#
# This is the precursor to ragged_dot, where different row-ranges of a
# single LHS matrix are multiplied by different group weight matrices.

# %%
G9, M9, K9, N9 = 4, 64, 128, 64

# --- Reference ---
def batched_matmul_spec(lhs, rhs):
    """lhs: (G, M, K), rhs: (G, K, N) → (G, M, N)"""
    return jnp.einsum('gmk,gkn->gmn', lhs, rhs)

# --- Kernel skeleton ---
def batched_matmul_kernel(lhs_ref, rhs_ref, o_ref):
    # With None in block_shape, the batch dim is squeezed:
    # lhs_ref: (M9, K9) — one group's lhs (batch dim squeezed)
    # rhs_ref: (K9, N9) — one group's rhs (batch dim squeezed)
    # o_ref: (M9, N9) — one group's output (batch dim squeezed)
    pass  # YOUR CODE HERE


# %%
lhs9 = jax.random.normal(jax.random.key(12), (G9, M9, K9))
rhs9 = jax.random.normal(jax.random.key(13), (G9, K9, N9))

check(batched_matmul_kernel, batched_matmul_spec, (lhs9, rhs9),
      grid=(G9,),
      in_specs=[
          pl.BlockSpec((None, M9, K9), lambda g: (g, 0, 0)),
          pl.BlockSpec((None, K9, N9), lambda g: (g, 0, 0)),
      ],
      out_specs=pl.BlockSpec((None, M9, N9), lambda g: (g, 0, 0)),
      out_shape=jax.ShapeDtypeStruct((G9, M9, N9), jnp.float32))

# %% [markdown]
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# With `None` in BlockSpec, the batch dimension is **squeezed** — the refs have shape `(M9, K9)` and `(K9, N9)` directly (no leading dim). So the kernel just needs a single matmul.
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# o_ref[...] = lhs_ref[...] @ rhs_ref[...]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 10: Fused Matmul + ReLU
#
# **Goal**: Compute `ReLU(A @ B)` in a single fused kernel — matmul and
# activation in one pass, no intermediate materialization.
#
# ### Theory
#
# On TPU, fusing operations into the kernel avoids an extra HBM round-trip.
# Without fusion: matmul writes `C` to HBM, then a separate kernel reads
# `C` back and applies ReLU. With fusion: ReLU is applied inside the kernel
# before the final store, saving one full read+write of the output matrix.
#
# The pattern is the same zero/accumulate/store from Puzzle 7, but the
# **store** step applies the activation before writing:
#
# ```python
# @pl.when(k_i == tiles_k - 1)
# def _store():
#     o_ref[...] = jnp.maximum(acc_ref[...], 0)  # fused ReLU!
# ```
#
# This fusion pattern generalizes to any elementwise activation (GELU,
# SiLU, etc.) and is used in production MoE kernels.

# %%
M10, K10, N10 = 128, 256, 128
bm10, bk10, bn10 = 64, 128, 64
tiles_m10 = M10 // bm10
tiles_n10 = N10 // bn10
tiles_k10 = K10 // bk10

# --- Reference ---
def fused_relu_spec(a, b):
    """a: (M10, K10), b: (K10, N10) → ReLU(a @ b)"""
    return jnp.maximum(a @ b, 0)

# --- Kernel skeleton ---
def fused_relu_kernel(a_ref, b_ref, o_ref, acc_ref):
    k_i = pl.program_id(2)
    pass  # YOUR CODE HERE
    # Same zero/accumulate/store as Puzzle 7, but apply ReLU before storing


# %%
a10 = jax.random.normal(jax.random.key(22), (M10, K10))
b10 = jax.random.normal(jax.random.key(23), (K10, N10))

check(fused_relu_kernel, fused_relu_spec, (a10, b10),
      grid=(tiles_m10, tiles_n10, tiles_k10),
      in_specs=[
          pl.BlockSpec((bm10, bk10), lambda m, n, k: (m, k)),
          pl.BlockSpec((bk10, bn10), lambda m, n, k: (k, n)),
      ],
      out_specs=pl.BlockSpec((bm10, bn10), lambda m, n, k: (m, n)),
      out_shape=jax.ShapeDtypeStruct((M10, N10), jnp.float32),
      scratch_shapes=[pltpu.VMEM((bm10, bn10), jnp.float32)])

# %% [markdown]
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# Copy the Puzzle 7 solution, but change the store step to apply `jnp.maximum(..., 0)` before writing to `o_ref`.
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     acc_ref[...] = jnp.zeros((bm10, bn10), dtype=jnp.float32)
#
# acc_ref[...] += a_ref[...] @ b_ref[...]
#
# @pl.when(k_i == tiles_k10 - 1)
# def _store():
#     o_ref[...] = jnp.maximum(acc_ref[...], 0)
# ```
# </details>

# %% [markdown]
# ---
# # Part III: Scalar Prefetch & Group Metadata (Puzzles 11–15)

# %% [markdown]
# ---
# ## Puzzle 11: Scalar Prefetch — Permuted Batched Matmul
#
# **Goal**: Implement a **permuted batched matmul** where the mapping from
# output group → rhs group is determined at runtime by a permutation array.
#
# ### Theory
#
# In ragged_dot, the tile-to-group mapping is computed at runtime (from
# `group_sizes`). Standard `BlockSpec` index maps only see grid indices —
# they can't access runtime arrays.
#
# **Scalar prefetch** solves this. With `PrefetchScalarGridSpec`:
# - Small arrays are loaded into **SMEM** (scalar memory) before the kernel
# - Index maps receive these SMEM refs as extra arguments
# - The kernel also receives them as leading arguments
#
# ```python
# PrefetchScalarGridSpec(
#     num_scalar_prefetch=1,  # first 1 arg is scalar-prefetched
#     in_specs=[...],
#     out_specs=...,
#     grid=(...),
# )
# ```
#
# Index map signature becomes: `lambda grid_idx0, ..., *prefetch_refs: (...)`
#
# The kernel signature becomes: `kernel(prefetch_ref0, ..., in_ref0, ..., out_ref, *scratch)`
#
# We skip teaching plain `GridSpec` as a separate concept — the simpler
# `grid=` kwarg to `pallas_call` (used in Puzzles 1–10) handles the basic
# case. `PrefetchScalarGridSpec` is introduced now because it's what
# production kernels use.
#
# **Note**: From this point on, puzzles use `grid_spec=` instead of
# `grid=`, so the `check` helper from earlier won't work. The checking
# code is inline instead.

# %%
G11 = 4
M11, K11, N11 = 64, 64, 64

# --- Reference ---
def permuted_matmul_spec(lhs, rhs, perm):
    """lhs: (G, M, K), rhs: (G, K, N), perm: (G,) → (G, M, N)
    out[i] = lhs[i] @ rhs[perm[i]]
    """
    return jnp.stack([lhs[i] @ rhs[perm[i]] for i in range(G11)])

# --- Kernel skeleton ---
def permuted_matmul_kernel(perm_ref, lhs_ref, rhs_ref, o_ref):
    # perm_ref: scalar-prefetched permutation array (in SMEM)
    # lhs_ref: (M11, K11) — current group's lhs
    # rhs_ref: (K11, N11) — permuted group's rhs (loaded via index map)
    # o_ref: (M11, N11) — output tile
    pass  # YOUR CODE HERE


# --- Index maps ---
def lhs_index_map11(g, perm_ref):
    return (g, 0, 0)

def rhs_index_map11(g, perm_ref):
    # Use the scalar-prefetched perm to look up which rhs group to load
    return (perm_ref[g], 0, 0)

def out_index_map11(g, perm_ref):
    return (g, 0, 0)

# %%
lhs11 = jax.random.normal(jax.random.key(14), (G11, M11, K11))
rhs11 = jax.random.normal(jax.random.key(15), (G11, K11, N11))
perm11 = jnp.array([2, 0, 3, 1], dtype=jnp.int32)  # permutation

expected11 = permuted_matmul_spec(lhs11, rhs11, perm11)

actual11 = pl.pallas_call(
    permuted_matmul_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=[
            pl.BlockSpec((None, M11, K11), lhs_index_map11),
            pl.BlockSpec((None, K11, N11), rhs_index_map11),
        ],
        out_specs=pl.BlockSpec((None, M11, N11), out_index_map11),
        grid=(G11,),
    ),
    out_shape=jax.ShapeDtypeStruct((G11, M11, N11), jnp.float32),
    interpret=True,
)(perm11, lhs11, rhs11)

if jnp.allclose(actual11, expected11, atol=1e-3):
    print(f"PASSED ✓  (shape={actual11.shape})")
else:
    max_err = float(jnp.max(jnp.abs(actual11 - expected11)))
    print(f"FAILED ✗  max error: {max_err:.6f}")

# %% [markdown]
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# The index maps handle the permutation using `perm_ref[g]`. By the time the kernel runs, `rhs_ref` already points to the correct permuted group. So the kernel body is identical to Puzzle 9 — just `lhs_ref[...] @ rhs_ref[...]`.
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# o_ref[...] = lhs_ref[...] @ rhs_ref[...]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 12: Group Metadata — CSR-style Tile Mapping
#
# **Goal**: Implement the `make_group_metadata` function that computes
# the tile-to-group mapping for ragged_dot. This is **pure JAX** — not a
# kernel puzzle.
#
# ### Theory
#
# In ragged_dot, `lhs` has shape `(M, K)` where rows are divided into `G`
# groups of variable sizes. We need to figure out which **tiles** belong to
# which **groups**.
#
# Given `group_sizes = [300, 212, 512]` with `bm = 128`:
#
# ```
# Row:  0              300    512         1024
#       ├── group 0 ───┤├─ g1 ─┤├── group 2 ──┤
#
# Tiles (bm=128):
#       [  0  ][ 128 ][ 256 ][ 384 ][ 512 ][ 640 ][ 768 ][ 896 ]
#       ├─g0──┤├─g0──┤├g0/g1┤├─g1──┤├─g2──┤├─g2──┤├─g2──┤├─g2──┤
#                      ^ partial tile: visited by BOTH g0 and g1
# ```
#
# Tile at row 256 straddles the group boundary. It gets visited **twice**:
# once for group 0 (rows 256-299 are valid) and once for group 1 (rows
# 300-383 are valid). The kernel uses a **mask** to only store the valid
# rows for each visit.
#
# **Rule of thumb**: `num_tiles = tiles_m + (number of non-aligned group
# boundaries)`. Aligned boundaries don't cause extra visits.
#
# **Output arrays**:
# - `group_offsets`: `[0, 300, 512, 1024]` — cumsum with leading 0
# - `group_ids`: maps each grid index → group id
# - `m_tile_ids`: maps each grid index → which m-tile to process
# - `num_tiles`: total number of grid iterations needed
#
# The arrays can be longer than `num_tiles` (padded with the last group).

# %%
def make_group_metadata_reference(group_sizes, m, bm):
    """Simple reference implementation — O(m) but correct."""
    num_groups = len(group_sizes)
    group_offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(group_sizes)])

    # Assign each row to a group
    row_to_group = jnp.zeros(m, dtype=jnp.int32)
    for g in range(num_groups):
        start = int(group_offsets[g])
        end = int(group_offsets[g + 1])
        row_to_group = row_to_group.at[start:end].set(g)

    # Assign each tile to group(s)
    tiles_m = m // bm
    group_ids_list = []
    m_tile_ids_list = []

    for t in range(tiles_m):
        tile_start = t * bm
        tile_end = (t + 1) * bm
        # Which groups touch this tile?
        groups_in_tile = jnp.unique(row_to_group[tile_start:tile_end])
        for g in groups_in_tile:
            group_ids_list.append(int(g))
            m_tile_ids_list.append(t)

    num_tiles = len(group_ids_list)

    # Pad to max possible length
    max_len = tiles_m + num_groups - 1
    group_ids = jnp.zeros(max_len, dtype=jnp.int32)
    m_tile_ids = jnp.zeros(max_len, dtype=jnp.int32)
    group_ids = group_ids.at[:num_tiles].set(jnp.array(group_ids_list, dtype=jnp.int32))
    m_tile_ids = m_tile_ids.at[:num_tiles].set(jnp.array(m_tile_ids_list, dtype=jnp.int32))
    # Pad remainder with last values
    if num_tiles < max_len:
        group_ids = group_ids.at[num_tiles:].set(group_ids_list[-1])
        m_tile_ids = m_tile_ids.at[num_tiles:].set(m_tile_ids_list[-1])

    return (group_offsets.astype(jnp.int32), group_ids, m_tile_ids), num_tiles


# %% [markdown]
# ### Your implementation — decomposed into 5 testable steps
#
# We break `make_group_metadata` into independent functions,
# each tested before combining them.
#
# ### Step 12a: Group Offsets
#
# **Goal**: Compute CSR-style prefix sum `[0, cumsum(group_sizes)]`.
#
# ```
# group_sizes = [300, 212, 512]
# group_offsets = [0, 300, 512, 1024]
#                  ^    ^    ^     ^
#                  g0   g1   g2   end
# ```

# %%
def compute_group_offsets(group_sizes):
    """[0, cumsum(group_sizes)] — maps group id → start row.

    Args:
        group_sizes: (G,) int32
    Returns:
        (G+1,) int32
    """
    pass  # YOUR CODE HERE


# %%
assert jnp.array_equal(
    compute_group_offsets(jnp.array([256, 256, 256, 256], dtype=jnp.int32)),
    jnp.array([0, 256, 512, 768, 1024], dtype=jnp.int32))
assert jnp.array_equal(
    compute_group_offsets(jnp.array([300, 212, 512], dtype=jnp.int32)),
    jnp.array([0, 300, 512, 1024], dtype=jnp.int32))
assert jnp.array_equal(
    compute_group_offsets(jnp.array([512, 0, 512], dtype=jnp.int32)),
    jnp.array([0, 512, 512, 1024], dtype=jnp.int32))
print("Step 12a — compute_group_offsets: PASSED ✓")

# %% [markdown]
# <details><summary>Hint — Full solution</summary>
#
# ```python
# return jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(group_sizes)])
# ```
# </details>

# %% [markdown]
# ### Step 12b: Tiles per Group

# %%
def compute_group_tiles(group_sizes, group_offsets, bm):
    """Number of tile visits per group (boundary tiles counted by both neighbors).

    Args:
        group_sizes: (G,) int32
        group_offsets: (G+1,) int32 from compute_group_offsets
        bm: tile size
    Returns:
        (G,) int32
    """
    pass  # YOUR CODE HERE
    # 1. Extract group starts and ends from offsets
    # 2. Round starts DOWN and ends UP to tile boundaries
    # 3. Handle zero-size groups
    # 4. Convert rounded range sizes to tile counts


# %%
assert jnp.array_equal(
    compute_group_tiles(jnp.array([256, 256, 256, 256], dtype=jnp.int32),
                        jnp.array([0, 256, 512, 768, 1024], dtype=jnp.int32), 128),
    jnp.array([2, 2, 2, 2]))
assert jnp.array_equal(
    compute_group_tiles(jnp.array([300, 212, 512], dtype=jnp.int32),
                        jnp.array([0, 300, 512, 1024], dtype=jnp.int32), 128),
    jnp.array([3, 2, 4]))
assert jnp.array_equal(
    compute_group_tiles(jnp.array([512, 0, 512], dtype=jnp.int32),
                        jnp.array([0, 512, 512, 1024], dtype=jnp.int32), 128),
    jnp.array([4, 0, 4]))
assert jnp.array_equal(
    compute_group_tiles(jnp.array([300, 0, 724], dtype=jnp.int32),
                        jnp.array([0, 300, 300, 1024], dtype=jnp.int32), 128),
    jnp.array([3, 0, 6]))
print("Step 12b — compute_group_tiles: PASSED ✓")

# %% [markdown]
# <details><summary>Hint — Full solution</summary>
#
# ```python
# group_starts = group_offsets[:-1]
# group_ends = group_offsets[1:]
# rounded_starts = (group_starts // bm * bm).astype(jnp.int32)
# rounded_ends = ((group_ends + bm - 1) // bm * bm).astype(jnp.int32)
# rounded_sizes = jnp.where(group_sizes == 0, 0, rounded_ends - rounded_starts)
# return rounded_sizes // bm
# ```
# </details>

# %% [markdown]
# ### Step 12c: Group IDs
#
# ```
# group_tiles = [3, 2, 4]  →  group_ids = [0,0,0, 1,1, 2,2,2,2]
# ```

# %%
def compute_group_ids(group_tiles, num_groups, max_len):
    """Flat array mapping grid index → group id.

    Args:
        group_tiles: (G,) int32 from compute_group_tiles
        num_groups: G
        max_len: output array length (padded)
    Returns:
        (max_len,) int32
    """
    pass  # YOUR CODE HERE


# %%
assert compute_group_ids(jnp.array([2, 2, 2, 2]), 4, 11)[:8].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
assert compute_group_ids(jnp.array([3, 2, 4]), 3, 10)[:9].tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]
print("Step 12c — compute_group_ids: PASSED ✓")

# %% [markdown]
# <details><summary>Hint — Full solution</summary>
#
# ```python
# return jnp.repeat(
#     jnp.arange(num_groups, dtype=jnp.int32),
#     group_tiles,
#     total_repeat_length=max_len,
# )
# ```
# </details>

# %% [markdown]
# ### Step 12d: Tile Visits
#
# ```
# group_offsets = [0, 300, 512, 1024],  bm = 128
# Group 1 starts at row 300 → inside tile 2 → extra visit
# tile_visits = [1, 1, 2, 1, 1, 1, 1, 1]
# ```

# %%
def compute_tile_visits(group_sizes, group_offsets, tiles_m, bm):
    """Visit count per tile (1 + extra for each mid-tile group boundary).

    Every tile is visited at least once. When a group boundary falls
    in the MIDDLE of a tile (not aligned to bm), that tile gets an
    extra visit. We need to count how many non-aligned boundaries
    land in each tile.

    Args:
        group_sizes: (G,) int32
        group_offsets: (G+1,) int32
        tiles_m: M // bm
        bm: tile size
    Returns:
        (tiles_m,) int32
    """
    pass  # YOUR CODE HERE
    # 1. Find group start positions (from offsets, skip the leading 0)
    # 2. Identify which starts are non-aligned (start % bm != 0)
    #    AND belong to non-empty groups
    # 3. For non-aligned starts, compute which tile they land in (start // bm)
    # 4. Count how many non-aligned boundaries per tile (jnp.histogram)
    # 5. Result = 1 + extra_visits_per_tile


# %%
assert compute_tile_visits(
    jnp.array([256, 256, 256, 256], dtype=jnp.int32),
    jnp.array([0, 256, 512, 768, 1024], dtype=jnp.int32), 8, 128
).tolist() == [1, 1, 1, 1, 1, 1, 1, 1]
assert compute_tile_visits(
    jnp.array([300, 212, 512], dtype=jnp.int32),
    jnp.array([0, 300, 512, 1024], dtype=jnp.int32), 8, 128
).tolist() == [1, 1, 2, 1, 1, 1, 1, 1]
assert compute_tile_visits(
    jnp.array([512, 0, 512], dtype=jnp.int32),
    jnp.array([0, 512, 512, 1024], dtype=jnp.int32), 8, 128
).tolist() == [1, 1, 1, 1, 1, 1, 1, 1]
assert compute_tile_visits(
    jnp.array([300, 0, 724], dtype=jnp.int32),
    jnp.array([0, 300, 300, 1024], dtype=jnp.int32), 8, 128
).tolist() == [1, 1, 2, 1, 1, 1, 1, 1]
assert compute_tile_visits(
    jnp.array([300, 212, 512], dtype=jnp.int32),
    jnp.array([0, 300, 512, 1024], dtype=jnp.int32), 8, 128
).dtype == jnp.int32, "compute_tile_visits must return int32, not float32"
print("Step 12d — compute_tile_visits: PASSED ✓")

# %% [markdown]
# <details><summary>Hint — Full solution</summary>
#
# ```python
# group_starts = group_offsets[:-1]
# aligned_or_empty = ((group_starts % bm) == 0) | (group_sizes == 0)
# partial_tile_ids = jnp.where(aligned_or_empty, tiles_m + 1, group_starts // bm)
# extra_visits = jnp.histogram(
#     partial_tile_ids, bins=tiles_m, range=(0, tiles_m)
# )[0]
# return (extra_visits + 1).astype(jnp.int32)
# ```
# </details>

# %% [markdown]
# ### Step 12e: M-tile IDs
#
# ```
# tile_visits = [1,1,2,1,1,1,1,1]  →  m_tile_ids = [0,1,2,2,3,4,5,6,7]
# ```

# %%
def compute_m_tile_ids(tile_visits, tiles_m, max_len):
    """Flat array mapping grid index → m-tile id.

    Args:
        tile_visits: (tiles_m,) int32 from compute_tile_visits
        tiles_m: M // bm
        max_len: output array length (padded)
    Returns:
        (max_len,) int32
    """
    pass  # YOUR CODE HERE


# %%
assert compute_m_tile_ids(jnp.array([1,1,1,1,1,1,1,1]), 8, 11)[:8].tolist() == [0,1,2,3,4,5,6,7]
assert compute_m_tile_ids(jnp.array([1,1,2,1,1,1,1,1]), 8, 10)[:9].tolist() == [0,1,2,2,3,4,5,6,7]
print("Step 12e — compute_m_tile_ids: PASSED ✓")

# %% [markdown]
# <details><summary>Hint — Full solution</summary>
#
# ```python
# return jnp.repeat(
#     jnp.arange(tiles_m, dtype=jnp.int32),
#     tile_visits,
#     total_repeat_length=max_len,
# )
# ```
# </details>

# %% [markdown]
# ### Step 12f: Combined `make_group_metadata`
#
# **Goal**: Chain the 5 steps above into the complete function.

# %%
def make_group_metadata_yours(group_sizes, m, bm):
    """Vectorized group metadata — chains steps 12a–12e.

    Args:
        group_sizes: jnp.array of shape (num_groups,), dtype int32
        m: total number of rows
        bm: tile size for m dimension

    Returns:
        (group_offsets, group_ids, m_tile_ids), num_tiles
    """
    num_groups = group_sizes.shape[0]
    tiles_m = m // bm
    max_len = tiles_m + num_groups - 1

    # YOUR CODE HERE — chain steps 12a–12e, then compute num_tiles
    # Replace this raise with your implementation:
    raise NotImplementedError("Chain compute_group_offsets → ... → compute_m_tile_ids")

    return (group_offsets, group_ids, m_tile_ids), num_tiles


# %%
# Integration tests — compare against reference
def check_metadata(name, group_sizes, m, bm):
    ref, ref_nt = make_group_metadata_reference(group_sizes, m, bm)
    yours, your_nt = make_group_metadata_yours(group_sizes, m, bm)
    ok = (ref_nt == your_nt
          and bool(jnp.array_equal(ref[0], yours[0]))
          and bool(jnp.array_equal(ref[1][:ref_nt], yours[1][:your_nt]))
          and bool(jnp.array_equal(ref[2][:ref_nt], yours[2][:your_nt])))
    status = "PASSED ✓" if ok else "FAILED ✗"
    print(f"  {name}: {status}  (num_tiles: ref={ref_nt}, yours={your_nt})")
    if not ok:
        print(f"    group_ids ref:   {ref[1][:ref_nt].tolist()}")
        print(f"    group_ids yours: {yours[1][:your_nt].tolist()}")
        print(f"    m_tile_ids ref:   {ref[2][:ref_nt].tolist()}")
        print(f"    m_tile_ids yours: {yours[2][:your_nt].tolist()}")

print("=== Integration tests ===")
check_metadata("Aligned groups",
               jnp.array([256, 256, 256, 256], dtype=jnp.int32), 1024, 128)
check_metadata("Unaligned groups",
               jnp.array([300, 212, 512], dtype=jnp.int32), 1024, 128)
check_metadata("Zero-size group (aligned)",
               jnp.array([512, 0, 512], dtype=jnp.int32), 1024, 128)
check_metadata("Zero-size group (non-aligned)",
               jnp.array([300, 0, 724], dtype=jnp.int32), 1024, 128)

# %% [markdown]
# <details><summary>Hint — Full solution</summary>
#
# ```python
# group_offsets = compute_group_offsets(group_sizes)
# group_tiles = compute_group_tiles(group_sizes, group_offsets, bm)
# group_ids = compute_group_ids(group_tiles, num_groups, max_len)
# tile_visits = compute_tile_visits(group_sizes, group_offsets, tiles_m, bm)
# m_tile_ids = compute_m_tile_ids(tile_visits, tiles_m, max_len)
# num_tiles = int(group_tiles.sum())
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 13: Configure Your Own Scalar-Prefetch `pallas_call`
#
# **Goal**: Given a working kernel, write the **entire** `pl.pallas_call`
# invocation from scratch, including `PrefetchScalarGridSpec` with
# `num_scalar_prefetch=3`, `in_specs` with runtime index maps, `out_specs`,
# and `grid`.
#
# ### Theory
#
# This is the most challenging configuration exercise. You need to
# understand:
# - **Which args are scalar-prefetched**: metadata arrays that index maps
#   need at runtime. They appear as leading args in the call and leading
#   refs in the kernel.
# - **How index maps receive prefetch refs**: each index map gets the grid
#   indices first, then all prefetch refs. E.g.,
#   `lambda i, go, gi, mt: (mt[i], 0)` — `i` is the grid index, `go/gi/mt`
#   are the three scalar-prefetched refs.
# - **How grid size comes from num_tiles**: the grid iterates over
#   `num_tiles` (from `make_group_metadata`), not over `tiles_m`.
# - **Call argument order**: scalar-prefetched args come first, then
#   regular inputs.

# %%
M13c, N13c = 1024, 64
bm13c = 128
G13c = 3

group_sizes_13c = jnp.array([300, 212, 512], dtype=jnp.int32)
(group_offsets_13c, group_ids_13c, m_tile_ids_13c), num_tiles_13c = \
    make_group_metadata_reference(group_sizes_13c, M13c, bm13c)

# The kernel is provided (solved):
def masked_copy_kernel_solved(group_offsets_ref, group_ids_ref, m_tile_ids_ref,
                               x_ref, o_ref):
    """Copy rows from x to output, masked by group boundaries."""
    grid_id = pl.program_id(0)
    group_id = group_ids_ref[grid_id]
    m_tile = m_tile_ids_ref[grid_id]
    group_start = group_offsets_ref[group_id]
    group_end = group_offsets_ref[group_id + 1]
    tile_start = m_tile * bm13c

    row_ids = tile_start + jax.lax.broadcasted_iota(jnp.int32, (bm13c, N13c), 0)
    mask = (row_ids >= group_start) & (row_ids < group_end)
    o_ref[...] = jnp.where(mask, x_ref[...], o_ref[...])

# Reference spec
def masked_copy_spec_13c(x, group_offsets, group_ids, m_tile_ids):
    out = jnp.zeros_like(x)
    for grid_id in range(num_tiles_13c):
        g = int(group_ids[grid_id])
        tile_id = int(m_tile_ids[grid_id])
        g_start = int(group_offsets[g])
        g_end = int(group_offsets[g + 1])
        t_start = tile_id * bm13c
        t_end = t_start + bm13c
        for row in range(t_start, t_end):
            if g_start <= row < g_end:
                out = out.at[row].set(x[row])
    return out


# %%
x13c = jax.random.normal(jax.random.key(26), (M13c, N13c))
expected13c = masked_copy_spec_13c(x13c, group_offsets_13c, group_ids_13c, m_tile_ids_13c)

# YOUR TASK: Write the complete pl.pallas_call invocation.
# Replace `None` with your working code.
#
# You need:
# - PrefetchScalarGridSpec with num_scalar_prefetch=3
# - in_specs: BlockSpec that uses m_tile_ids to route tiles
# - out_specs: BlockSpec that uses m_tile_ids to route tiles
# - grid=(num_tiles_13c,)
# - Call args: (group_offsets, group_ids, m_tile_ids, x) — scalar prefetch first!
actual13c = None  # Replace with pl.pallas_call(...)(...) invocation


# %%
if actual13c is not None and jnp.allclose(actual13c, expected13c, atol=1e-5):
    print(f"PASSED ✓  (shape={actual13c.shape})")
else:
    print("FAILED ✗  (fill in the cell above)")

# %% [markdown]
# <details><summary>Hint 1 of 2 — Structure</summary>
#
# ```python
# actual13c = pl.pallas_call(
#     masked_copy_kernel_solved,
#     grid_spec=pltpu.PrefetchScalarGridSpec(
#         num_scalar_prefetch=3,
#         in_specs=[pl.BlockSpec((bm13c, N13c), lambda i, go, gi, mt: (mt[i], 0))],
#         out_specs=pl.BlockSpec((bm13c, N13c), lambda i, go, gi, mt: (mt[i], 0)),
#         grid=(num_tiles_13c,),
#     ),
#     out_shape=...,
#     interpret=True,
# )(...)  # scalar-prefetched args first, then regular inputs
# ```
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# actual13c = pl.pallas_call(
#     masked_copy_kernel_solved,
#     grid_spec=pltpu.PrefetchScalarGridSpec(
#         num_scalar_prefetch=3,
#         in_specs=[pl.BlockSpec((bm13c, N13c), lambda i, go, gi, mt: (mt[i], 0))],
#         out_specs=pl.BlockSpec((bm13c, N13c), lambda i, go, gi, mt: (mt[i], 0)),
#         grid=(num_tiles_13c,),
#     ),
#     out_shape=jax.ShapeDtypeStruct((M13c, N13c), jnp.float32),
#     interpret=True,
# )(group_offsets_13c, group_ids_13c, m_tile_ids_13c, x13c)
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 14: Masked Store with Group Boundaries
#
# **Goal**: Write a kernel that copies input rows to output, but **masks**
# writes based on group boundaries. Only rows belonging to the current
# group are written; other rows retain their previous value (zero).
#
# ### Theory
#
# When a tile straddles a group boundary, some rows belong to group `g`
# and others to group `g+1`. The kernel must only store the rows that
# belong to the **current group** being processed.
#
# The mask is built from:
# - `group_offsets[group_id]` → start row of current group
# - `group_offsets[group_id + 1]` → end row of current group
# - `m_tile_ids[grid_id] * bm` → first row of current tile
#
# ```python
# row_indices = tile_start + jnp.arange(bm)
# mask = (row_indices >= group_start) & (row_indices < group_end)
# ```
#
# For a 2D mask (bm, N), use `jax.lax.broadcasted_iota(dtype, shape, dim)`
# — it creates an array where values along `dim` are `0, 1, 2, ...` and
# all other dimensions are broadcast. Think of it as a multi-dimensional
# `jnp.arange`:
# ```python
# broadcasted_iota(int32, (4, 3), 0) → [[0,0,0], [1,1,1], [2,2,2], [3,3,3]]
# broadcasted_iota(int32, (4, 3), 1) → [[0,1,2], [0,1,2], [0,1,2], [0,1,2]]
# ```
#
# This is exactly the `get_store_mask` pattern used in ragged_dot.

# %%
M14 = 1024
N14 = 64
bm14 = 128
G14 = 3

group_sizes_14 = jnp.array([300, 212, 512], dtype=jnp.int32)

(group_offsets_14, group_ids_14, m_tile_ids_14), num_tiles_14 = \
    make_group_metadata_reference(group_sizes_14, M14, bm14)

# --- Reference ---
def masked_copy_spec(x, group_offsets, group_ids, m_tile_ids):
    """Copy x to output, but only rows within their assigned group."""
    out = jnp.zeros_like(x)
    for grid_id in range(num_tiles_14):
        g = int(group_ids[grid_id])
        tile_id = int(m_tile_ids[grid_id])
        g_start = int(group_offsets[g])
        g_end = int(group_offsets[g + 1])
        t_start = tile_id * bm14
        t_end = t_start + bm14
        for row in range(t_start, t_end):
            if g_start <= row < g_end:
                out = out.at[row].set(x[row])
    return out

# --- Kernel skeleton ---
def masked_copy_kernel(group_offsets_ref, group_ids_ref, m_tile_ids_ref,
                       x_ref, o_ref):
    # group_offsets_ref, group_ids_ref, m_tile_ids_ref: metadata in SMEM
    # x_ref: (bm14, N14) — tile of input
    # o_ref: (bm14, N14) — tile of output
    grid_id = pl.program_id(0)
    pass  # YOUR CODE HERE
    # 1. Look up which group and tile this grid iteration processes
    # 2. Get the group's row boundaries
    # 3. Build a 2D boolean mask for rows inside this group
    # 4. Masked store: only write rows belonging to this group


# %%
x14 = jax.random.normal(jax.random.key(27), (M14, N14))
expected14 = masked_copy_spec(x14, group_offsets_14, group_ids_14, m_tile_ids_14)

actual14 = pl.pallas_call(
    masked_copy_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        in_specs=[pl.BlockSpec((bm14, N14), lambda i, go, gi, mt: (mt[i], 0))],
        out_specs=pl.BlockSpec((bm14, N14), lambda i, go, gi, mt: (mt[i], 0)),
        grid=(num_tiles_14,),
    ),
    out_shape=jax.ShapeDtypeStruct((M14, N14), jnp.float32),
    interpret=True,
)(group_offsets_14, group_ids_14, m_tile_ids_14, x14)

if jnp.allclose(actual14, expected14, atol=1e-5):
    print(f"PASSED ✓  (shape={actual14.shape})")
else:
    nan_count = int(jnp.isnan(actual14).sum())
    if nan_count > 0:
        nan_rows = jnp.where(jnp.isnan(actual14).any(axis=1))[0]
        print(f"FAILED ✗  {nan_count} NaN values in output (rows: {nan_rows.tolist()[:8]}...)")
        print(f"  Common cause: indexing group_offsets_ref with grid_id instead of group_id")
    else:
        diff = jnp.abs(actual14 - expected14)
        worst_row = int(jnp.argmax(diff.max(axis=1)))
        max_err = float(diff[worst_row].max())
        print(f"FAILED ✗  max error: {max_err:.6f} at row {worst_row}")
        g_boundaries = group_offsets_14.tolist()
        print(f"  Group boundaries at rows: {g_boundaries}")

# %% [markdown]
# <details><summary>Hint 1 of 2 — Key pattern</summary>
#
# ```python
# group_id = group_ids_ref[grid_id]
# m_tile = m_tile_ids_ref[grid_id]
# group_start = group_offsets_ref[group_id]
# group_end = group_offsets_ref[group_id + 1]
# tile_start = m_tile * bm14
#
# # Build a (bm14, N14) mask where row_index in [group_start, group_end)
# # Tip: jax.lax.broadcasted_iota(dtype, shape, dimension)
# ```
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# group_id = group_ids_ref[grid_id]
# m_tile = m_tile_ids_ref[grid_id]
# group_start = group_offsets_ref[group_id]
# group_end = group_offsets_ref[group_id + 1]
# tile_start = m_tile * bm14
#
# row_ids = tile_start + jax.lax.broadcasted_iota(jnp.int32, (bm14, N14), 0)
# mask = (row_ids >= group_start) & (row_ids < group_end)
#
# o_ref[...] = jnp.where(mask, x_ref[...], o_ref[...])
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 15: Softmax Kernel
#
# **Goal**: Write a softmax kernel **and** configure the `pallas_call`
# yourself. Each kernel invocation processes one row-block of the full
# matrix.
#
# ### Theory
#
# Softmax is a non-matmul kernel that combines **reduction** (max, sum)
# with **elementwise** ops (exp, divide). For each row:
#
# ```
# softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
# ```
#
# Since each row fits entirely within one tile (no column-tiling needed),
# the kernel is simpler than matmul — no `@pl.when` guards, no scratch
# accumulator, just straight computation. The grid only tiles along rows.
#
# **This puzzle is also a "configure your own pallas_call" exercise** —
# you need to write both the kernel body AND the `grid`, `in_specs`,
# `out_specs`.
#
# In production (FlashAttention), the column dimension is also tiled using
# an **online softmax** algorithm that maintains running max and sum across
# column tiles. The core pattern of max → subtract → exp → normalize is
# the same.

# %%
ROWS15, COLS15 = 256, 128
bm15 = 64

# --- Reference ---
def softmax_spec(x):
    """x: (ROWS15, COLS15) → row-wise softmax"""
    return jax.nn.softmax(x, axis=1)

# --- Kernel skeleton ---
def softmax_kernel(x_ref, o_ref):
    # x_ref: (bm15, COLS15) — one row block (full width)
    # o_ref: (bm15, COLS15) — output
    pass  # YOUR CODE HERE
    # 1. Compute row max for numerical stability
    # 2. Subtract max, exponentiate
    # 3. Divide by row sum


# %%
x15 = jax.random.normal(jax.random.key(28), (ROWS15, COLS15))

# YOUR TASK: Write the kernel above AND define the config below.
softmax_grid = ...       # TODO: how many row blocks?
softmax_in_specs = ...   # TODO: list with one BlockSpec
softmax_out_specs = ...  # TODO: BlockSpec for output

check(softmax_kernel, softmax_spec, (x15,),
      grid=softmax_grid,
      in_specs=softmax_in_specs,
      out_specs=softmax_out_specs)

# %% [markdown]
# <details><summary>Hint 1 of 3 — Kernel</summary>
#
# ```python
# x = x_ref[...]
# row_max = x.max(axis=1, keepdims=True)
# exp_x = jnp.exp(x - row_max)
# row_sum = exp_x.sum(axis=1, keepdims=True)
# o_ref[...] = exp_x / row_sum
# ```
# </details>
#
# <details><summary>Hint 2 of 3 — Config</summary>
#
# ```python
# softmax_grid = (ROWS15 // bm15,)  # 256 // 64 = 4 row blocks
# softmax_in_specs = [pl.BlockSpec((bm15, COLS15), lambda i: (i, 0))]
# softmax_out_specs = pl.BlockSpec((bm15, COLS15), lambda i: (i, 0))
# ```
# </details>
#
# <details><summary>Hint 3 of 3 — Full solution</summary>
#
# ```python
# def softmax_kernel(x_ref, o_ref):
#     x = x_ref[...]
#     row_max = x.max(axis=1, keepdims=True)
#     exp_x = jnp.exp(x - row_max)
#     row_sum = exp_x.sum(axis=1, keepdims=True)
#     o_ref[...] = exp_x / row_sum
#
# softmax_grid = (ROWS15 // bm15,)
# softmax_in_specs = [pl.BlockSpec((bm15, COLS15), lambda i: (i, 0))]
# softmax_out_specs = pl.BlockSpec((bm15, COLS15), lambda i: (i, 0))
# ```
# </details>

# %% [markdown]
# ---
# # Part IV: Ragged Dot (Puzzles 16–20)

# %% [markdown]
# ### Provided utilities
#
# These are the building blocks from Parts I–III. They're provided
# here so you can focus on the kernel logic.

# %%
def make_group_metadata(group_sizes, m, bm, *, visit_empty_groups=False):
    """Compute tile-to-group mapping for ragged_dot.

    Returns:
        (group_offsets, group_ids, m_tile_ids), num_tiles
    """
    num_groups = group_sizes.shape[0]
    tiles_m = m // bm

    group_ends = jnp.cumsum(group_sizes)
    group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])

    group_starts = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]])
    rounded_ends = ((group_ends + bm - 1) // bm * bm).astype(jnp.int32)
    rounded_starts = (group_starts // bm * bm).astype(jnp.int32)
    rounded_sizes = rounded_ends - rounded_starts
    rounded_sizes = jnp.where(group_sizes == 0, 0, rounded_sizes)
    group_tiles = rounded_sizes // bm

    if visit_empty_groups:
        group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

    group_ids = jnp.repeat(
        jnp.arange(num_groups, dtype=jnp.int32),
        group_tiles,
        total_repeat_length=tiles_m + num_groups - 1,
    )

    partial_mask = ((group_offsets[:-1] % bm) == 0) | (group_sizes == 0)
    if visit_empty_groups:
        partial_mask = jnp.where(group_sizes == 0, 0, partial_mask)
    partial_tile_ids = jnp.where(partial_mask, tiles_m + 1, group_offsets[:-1] // bm)
    tile_visits = (
        jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m))[0] + 1
    )
    m_tile_ids = jnp.repeat(
        jnp.arange(tiles_m, dtype=jnp.int32),
        tile_visits.astype(jnp.int32),
        total_repeat_length=tiles_m + num_groups - 1,
    )

    num_tiles = int(group_tiles.sum())
    return (group_offsets, group_ids, m_tile_ids), num_tiles


def get_store_mask(grid_id, group_offsets, group_ids, m_tile_ids, bm, bn):
    """Build a (bm, bn) boolean mask for rows belonging to the current group."""
    group_id = group_ids[grid_id]
    group_start = group_offsets[group_id]
    group_end = group_offsets[group_id + 1]
    m_id = m_tile_ids[grid_id] * bm
    iota = jax.lax.broadcasted_iota(jnp.int32, (bm, bn), 0) + m_id
    return (iota >= group_start) & (iota < group_end)


# --- Shared index maps for ragged_dot ---
def lhs_imap(n_i, grid_id, k_i, group_meta_ref, group_offset_ref):
    _, _, m_tile_ids = group_meta_ref
    return (m_tile_ids[grid_id], k_i)

def rhs_imap(n_i, grid_id, k_i, group_meta_ref, group_offset_ref):
    _, group_ids, _ = group_meta_ref
    return (group_ids[grid_id], k_i, n_i)

def out_imap(n_i, grid_id, k_i, group_meta_ref, group_offset_ref):
    _, _, m_tile_ids = group_meta_ref
    return (m_tile_ids[grid_id], n_i)


# %% [markdown]
# ---
# ## Puzzle 16: Simple Grouped Matmul — Equal Groups, Tile-Aligned
#
# **Goal**: Implement grouped matmul for the simplest case: all groups have
# equal size and group sizes are divisible by the tile size.
#
# ### Theory
#
# This is the "easy mode" ragged_dot. With equal, tile-aligned groups:
# - No partial tiles (every tile belongs to exactly one group)
# - `group_ids` is a simple repeat: `[0,0,1,1,2,2,3,3]`
# - `m_tile_ids` = `[0,1,2,3,4,5,6,7]` (just sequential)
# - No masking needed on stores
#
# **Grid**: `(tiles_n, num_tiles, tiles_k)`
# - `tiles_n`: N dimension (parallel — independent output columns)
# - `num_tiles`: M tiles across all groups (may revisit same output)
# - `tiles_k`: K reduction dimension (accumulates)
#
# **Kernel structure** (same as tokamax `gmm`):
# 1. Get `grid_id = program_id(1)`, `k_i = program_id(2)`
# 2. Zero accumulator when `k_i == 0`
# 3. Accumulate `lhs_tile @ rhs_tile`
# 4. Store on last K tile
#
# **Index maps** (provided — study them!):
# - `lhs_imap`: `(n_i, grid_id, k_i) → (m_tile_ids[grid_id], k_i)`
# - `rhs_imap`: `(n_i, grid_id, k_i) → (group_ids[grid_id], k_i, n_i)`
# - `out_imap`: `(n_i, grid_id, k_i) → (m_tile_ids[grid_id], n_i)`
#
# The `group_ids` lookup in `rhs_imap` is what routes each tile to
# the correct group's weight matrix!

# %%
G16 = 4
M16, K16, N16 = 512, 256, 128
bm16, bk16, bn16 = 128, 128, 128

group_sizes_16 = jnp.array([M16 // G16] * G16, dtype=jnp.int32)
tiles_k16 = K16 // bk16
tiles_n16 = N16 // bn16

(group_offsets_16, group_ids_16, m_tile_ids_16), num_tiles_16 = \
    make_group_metadata(group_sizes_16, M16, bm16)

# --- Reference ---
def simple_gmm_spec(lhs, rhs, group_sizes):
    """lhs: (M, K), rhs: (G, K, N), group_sizes: (G,) → (M, N)"""
    offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(group_sizes)])
    out = jnp.zeros((lhs.shape[0], rhs.shape[2]), dtype=jnp.float32)
    for g in range(len(group_sizes)):
        s, e = int(offsets[g]), int(offsets[g + 1])
        out = out.at[s:e].set(lhs[s:e] @ rhs[g])
    return out

# --- Kernel skeleton ---
def simple_gmm_kernel(group_metadata_ref, group_offset_ref,
                      lhs_ref, rhs_ref, o_ref, acc_ref):
    # group_metadata_ref: (group_offsets, group_ids, m_tile_ids) in SMEM
    # group_offset_ref: unused here (for sharding)
    # lhs_ref: (bm16, bk16) — tile of lhs
    # rhs_ref: (bk16, bn16) — tile of rhs (group dim squeezed by None)
    # o_ref: (bm16, bn16) — output tile
    # acc_ref: (bm16, bn16) — scratch accumulator
    grid_id = pl.program_id(1)
    k_i = pl.program_id(2)

    pass  # YOUR CODE HERE
    # 1. Zero accumulator on first K tile
    # 2. Accumulate tile matmul
    # 3. Store result on last K tile


# %%
lhs16 = jax.random.normal(jax.random.key(30), (M16, K16))
rhs16 = jax.random.normal(jax.random.key(31), (G16, K16, N16))
expected16 = simple_gmm_spec(lhs16, rhs16, group_sizes_16)

group_metadata_16 = (group_offsets_16, group_ids_16, m_tile_ids_16)
group_offset_16 = jnp.array([0], dtype=jnp.int32)

actual16 = pl.pallas_call(
    simple_gmm_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=[
            pl.BlockSpec((bm16, bk16), lhs_imap),
            pl.BlockSpec((None, bk16, bn16), rhs_imap),
        ],
        out_specs=pl.BlockSpec((bm16, bn16), out_imap),
        grid=(tiles_n16, num_tiles_16, tiles_k16),
        scratch_shapes=[pltpu.VMEM((bm16, bn16), jnp.float32)],
    ),
    out_shape=jax.ShapeDtypeStruct((M16, N16), jnp.float32),
    interpret=True,
)(group_metadata_16, group_offset_16, lhs16, rhs16)

if jnp.allclose(actual16, expected16, atol=1e-2, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual16.shape})")
else:
    max_err = float(jnp.max(jnp.abs(actual16 - expected16)))
    print(f"FAILED ✗  max error: {max_err:.6f}")
    print(f"  Expected[:2,:4]:\n{expected16[:2,:4]}")
    print(f"  Actual[:2,:4]:\n{actual16[:2,:4]}")

# %% [markdown]
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# The kernel body is identical to Puzzle 7: zero / accumulate / store with `@pl.when`. The index maps (already provided) handle all the group-to-tile routing via `group_ids` and `m_tile_ids`. With `None` in the rhs BlockSpec, the group dimension is squeezed — `rhs_ref` is just `(bk, bn)`.
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     acc_ref[...] = jnp.zeros((bm16, bn16), dtype=jnp.float32)
#
# acc_ref[...] += lhs_ref[...] @ rhs_ref[...]
#
# @pl.when(k_i == tiles_k16 - 1)
# def _store():
#     o_ref[...] = acc_ref[...]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 17: Full Ragged Dot — Unequal Groups
#
# **Goal**: Handle **variable group sizes** where tiles can straddle group
# boundaries. This is the real ragged_dot.
#
# ### Theory
#
# The only difference from Puzzle 16: when groups are unequal, a tile may
# be visited **multiple times** (once per group it straddles). On each visit,
# the kernel must **mask** the store so only rows belonging to the current
# group are written.
#
# `make_group_metadata` handles all the complexity — the `group_ids` and
# `m_tile_ids` arrays already encode the repeated visits. The kernel just
# needs to add the mask at store time:
#
# ```python
# mask = get_store_mask(grid_id, group_offsets, group_ids, m_tile_ids, bm, bn)
# o_ref[...] = jnp.where(mask, acc[...], o_ref[...])
# ```
#
# ```
# Tile at row 256, bm=128:
# ┌────────────────────────┐
# │ rows 256-299: group 0  │ ← Visit 1: mask=True for rows 256-299
# │ rows 300-383: group 1  │ ← Visit 2: mask=True for rows 300-383
# └────────────────────────┘
# ```

# %%
G17 = 3
M17, K17, N17 = 1024, 256, 128
bm17, bk17, bn17 = 128, 128, 128

group_sizes_17 = jnp.array([300, 212, 512], dtype=jnp.int32)
tiles_k17 = K17 // bk17
tiles_n17 = N17 // bn17

(group_offsets_17, group_ids_17, m_tile_ids_17), num_tiles_17 = \
    make_group_metadata(group_sizes_17, M17, bm17)

print(f"M={M17}, G={G17}, group_sizes={group_sizes_17.tolist()}")
print(f"num_tiles={num_tiles_17} (vs {M17//bm17} base tiles)")
print(f"group_ids[:num_tiles]={group_ids_17[:num_tiles_17].tolist()}")
print(f"m_tile_ids[:num_tiles]={m_tile_ids_17[:num_tiles_17].tolist()}")

# --- Reference ---
def ragged_dot_spec(lhs, rhs, group_sizes):
    """Same as jax.lax.ragged_dot but explicit for clarity."""
    offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(group_sizes)])
    out = jnp.zeros((lhs.shape[0], rhs.shape[2]), dtype=jnp.float32)
    for g in range(len(group_sizes)):
        s, e = int(offsets[g]), int(offsets[g + 1])
        if s < e:
            out = out.at[s:e].set(lhs[s:e] @ rhs[g])
    return out

# --- Kernel skeleton ---
def ragged_dot_kernel(group_metadata_ref, group_offset_ref,
                      lhs_ref, rhs_ref, o_ref, acc_ref):
    group_offsets, group_ids, m_tile_ids = group_metadata_ref
    grid_id = pl.program_id(1)
    k_i = pl.program_id(2)

    pass  # YOUR CODE HERE
    # Same as Puzzle 16, but on the last K tile, apply a masked store
    # so only rows belonging to the current group are written.


# %%
lhs17 = jax.random.normal(jax.random.key(40), (M17, K17))
rhs17 = jax.random.normal(jax.random.key(41), (G17, K17, N17))
expected17 = ragged_dot_spec(lhs17, rhs17, group_sizes_17)

group_metadata_17 = (group_offsets_17, group_ids_17, m_tile_ids_17)
group_offset_17 = jnp.array([0], dtype=jnp.int32)

actual17 = pl.pallas_call(
    ragged_dot_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=[
            pl.BlockSpec((bm17, bk17), lhs_imap),
            pl.BlockSpec((None, bk17, bn17), rhs_imap),
        ],
        out_specs=pl.BlockSpec((bm17, bn17), out_imap),
        grid=(tiles_n17, num_tiles_17, tiles_k17),
        scratch_shapes=[pltpu.VMEM((bm17, bn17), jnp.float32)],
    ),
    out_shape=jax.ShapeDtypeStruct((M17, N17), jnp.float32),
    interpret=True,
)(group_metadata_17, group_offset_17, lhs17, rhs17)

total_rows17 = int(group_sizes_17.sum())
if jnp.allclose(actual17[:total_rows17], expected17[:total_rows17], atol=1e-2, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual17.shape})")
    print(f"  Verified {total_rows17} active rows")
else:
    max_err = float(jnp.max(jnp.abs(actual17[:total_rows17] - expected17[:total_rows17])))
    print(f"FAILED ✗  max error: {max_err:.6f}")

# %% [markdown]
# <details><summary>Hint 1 of 2 — The masked store block</summary>
#
# ```python
# # Steps 1-2 are the same as Puzzle 16 (zero + accumulate)
#
# @pl.when(k_i == tiles_k17 - 1)
# def _store():
#     mask = get_store_mask(grid_id, group_offsets, group_ids,
#                           m_tile_ids, bm17, bn17)
#     acc = acc_ref[...]
#     o_ref[...] = jnp.where(mask, acc, o_ref[...].astype(acc.dtype))
# ```
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     acc_ref[...] = jnp.zeros((bm17, bn17), dtype=jnp.float32)
#
# acc_ref[...] += lhs_ref[...] @ rhs_ref[...]
#
# @pl.when(k_i == tiles_k17 - 1)
# def _store():
#     mask = get_store_mask(grid_id, group_offsets, group_ids,
#                           m_tile_ids, bm17, bn17)
#     acc = acc_ref[...]
#     o_ref[...] = jnp.where(mask, acc, o_ref[...].astype(acc.dtype))
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 18: Transpose Grouped Matmul (tgmm)
#
# **Goal**: Implement the **backward-pass** kernel: `tgmm` computes the
# gradient w.r.t. the RHS weight matrices.
#
# ### Theory
#
# In the backward pass of ragged_dot, we need:
# - `dlhs = dout @ rhs[g].T` (gradient w.r.t. lhs — another gmm)
# - `drhs[g] = lhs[g_rows].T @ dout[g_rows]` (gradient w.r.t. rhs — this is tgmm)
#
# **tgmm** computes `lhs.T @ rhs` accumulated per group:
# - `lhs`: `(M, K)` (original lhs, or the transposed `(K, M)` passed as `(M, K)`)
# - `rhs`: `(M, N)` (dout)
# - `out`: `(G, K, N)` — one output per group
#
# **Key difference from gmm**: In gmm, multiple K tiles contribute to the
# same output tile (accumulate over K). In tgmm, multiple **M tiles** from
# the same group contribute to the same output tile (accumulate over group
# rows). This requires a different accumulation pattern:
#
# - **Prologue** (entering new group): zero the accumulator
# - **Body**: accumulate `lhs_tile.T @ rhs_tile`, masked by group boundaries
# - **Epilogue** (leaving group): store accumulator to output
#
# Group transitions detected by comparing consecutive group_ids.
#
# ```
# Grid iteration:  0   1   2   3   4   5   6   7   8   9
# group_ids:      [0,  0,  0,  1,  1,  2,  2,  2,  2,  2]
#                  P       E  P    E  P               E
#                  P = prologue (zero), E = epilogue (store)
# ```
#
# **New concepts in this puzzle:**
#
# - **`visit_empty_groups=True`**: If a group has zero rows, we still
#   need one grid iteration for it — so the kernel can zero and store
#   an empty accumulator. Without this, the output for that group
#   would contain garbage.
#
# - **`pl.num_programs(axis)`**: Returns the total number of grid
#   iterations along an axis (like `gridDim` in CUDA). Used here to
#   detect the very last iteration for the final epilogue.
#
# - **Grid axis order is `(tiles_n, tiles_k, num_tiles)`** — note
#   that `num_tiles` is now on axis 2 (not axis 1 like in gmm).
#   This is because the M-tile iteration is the "reduction" axis
#   in tgmm (we accumulate across M tiles), so it must be the
#   innermost `"arbitrary"` dimension for correct pipelining.

# %%
G18 = 3
M18, K18, N18 = 1024, 128, 128
bm18, bk18, bn18 = 128, 128, 128

group_sizes_18 = jnp.array([300, 340, 384], dtype=jnp.int32)
tiles_k18 = K18 // bk18
tiles_n18 = N18 // bn18

(group_offsets_18, group_ids_18, m_tile_ids_18), num_tiles_18 = \
    make_group_metadata(group_sizes_18, M18, bm18, visit_empty_groups=True)

print(f"group_sizes={group_sizes_18.tolist()}, num_tiles={num_tiles_18}")
print(f"group_ids={group_ids_18[:num_tiles_18].tolist()}")

# --- Reference ---
def tgmm_spec(lhs_t, rhs, group_sizes):
    """lhs_t: (K, M), rhs: (M, N) → (G, K, N)
    Computes lhs_t[:, g_start:g_end] @ rhs[g_start:g_end, :] per group.
    """
    offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(group_sizes)])
    G = len(group_sizes)
    K, N = lhs_t.shape[0], rhs.shape[1]
    out = jnp.zeros((G, K, N), dtype=jnp.float32)
    for g in range(G):
        s, e = int(offsets[g]), int(offsets[g + 1])
        if s < e:
            out = out.at[g].set(lhs_t[:, s:e] @ rhs[s:e, :])
    return out

# --- Kernel skeleton ---
def tgmm_kernel(group_metadata_ref, group_offset_ref,
                lhs_ref, rhs_ref, o_ref, acc_ref):
    # lhs_ref: (bm18, bk18) — tile of lhs (M, K)
    # rhs_ref: (bm18, bn18) — tile of rhs
    # o_ref: (bk18, bn18) — output tile for one group (None dim squeezed)
    # acc_ref: (bk18, bn18) — scratch accumulator
    group_offsets, group_ids, m_tile_ids = group_metadata_ref
    grid_id = pl.program_id(2)  # tgmm grid: (tiles_n, tiles_k, num_tiles)

    pass  # YOUR CODE HERE
    # 1. Detect group transitions: when does a new group start? end?
    # 2. Zero accumulator at the start of each group
    # 3. Accumulate masked lhs.T @ rhs
    # 4. Store accumulator at the end of each group


# --- Index maps for tgmm ---
def tgmm_lhs_imap(n_i, k_i, grid_id, group_meta_ref, group_offset_ref):
    _, _, m_tile_ids = group_meta_ref
    return (m_tile_ids[grid_id], k_i)

def tgmm_rhs_imap(n_i, k_i, grid_id, group_meta_ref, group_offset_ref):
    _, _, m_tile_ids = group_meta_ref
    return (m_tile_ids[grid_id], n_i)

def tgmm_out_imap(n_i, k_i, grid_id, group_meta_ref, group_offset_ref):
    _, group_ids, _ = group_meta_ref
    return (group_ids[grid_id], k_i, n_i)


# %%
lhs_t_18 = jax.random.normal(jax.random.key(50), (K18, M18))
rhs18 = jax.random.normal(jax.random.key(51), (M18, N18))
expected18 = tgmm_spec(lhs_t_18, rhs18, group_sizes_18)

# tgmm works on (M, K) internally — transpose lhs
lhs18 = lhs_t_18.T  # (M, K)

group_metadata_18 = (group_offsets_18, group_ids_18, m_tile_ids_18)
group_offset_18 = jnp.array([0], dtype=jnp.int32)

actual18 = pl.pallas_call(
    tgmm_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=[
            pl.BlockSpec((bm18, bk18), tgmm_lhs_imap),
            pl.BlockSpec((bm18, bn18), tgmm_rhs_imap),
        ],
        out_specs=pl.BlockSpec((None, bk18, bn18), tgmm_out_imap),
        grid=(tiles_n18, tiles_k18, num_tiles_18),
        scratch_shapes=[pltpu.VMEM((bk18, bn18), jnp.float32)],
    ),
    out_shape=jax.ShapeDtypeStruct((G18, K18, N18), jnp.float32),
    interpret=True,
)(group_metadata_18, group_offset_18, lhs18, rhs18)

if jnp.allclose(actual18, expected18, atol=1e-1, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual18.shape})")
else:
    max_err = float(jnp.max(jnp.abs(actual18 - expected18)))
    print(f"FAILED ✗  max error: {max_err:.6f}")
    print(f"  Expected[0,:2,:4]:\n{expected18[0,:2,:4]}")
    print(f"  Actual[0,:2,:4]:\n{actual18[0,:2,:4]}")

# %% [markdown]
# <details><summary>Hint 1 of 3 — Prologue/epilogue detection</summary>
#
# ```python
# group = group_ids[grid_id]
# prev_group = group_ids[jnp.where(grid_id > 0, grid_id - 1, 0)]
# is_prologue = (grid_id == 0) | (group != prev_group)
#
# is_end = grid_id == (pl.num_programs(2) - 1)
# next_group = group_ids[jnp.where(is_end, grid_id, grid_id + 1)]
# is_epilogue = is_end | (group != next_group)
# ```
# </details>
#
# <details><summary>Hint 2 of 3 — Masked accumulation</summary>
#
# ```python
# @pl.when(is_prologue)
# def _zero():
#     acc_ref[...] = jnp.zeros((bk18, bn18), dtype=jnp.float32)
#
# mask_lhs = get_store_mask(grid_id, group_offsets, group_ids,
#                            m_tile_ids, bm18, bk18)
# mask_rhs = get_store_mask(grid_id, group_offsets, group_ids,
#                            m_tile_ids, bm18, bn18)
# lhs_masked = jnp.where(mask_lhs, lhs_ref[...], 0)
# rhs_masked = jnp.where(mask_rhs, rhs_ref[...], 0)
# acc_ref[...] += lhs_masked.T @ rhs_masked
# ```
# </details>
#
# <details><summary>Hint 3 of 3 — Full solution</summary>
#
# ```python
# group = group_ids[grid_id]
# prev_group = group_ids[jnp.where(grid_id > 0, grid_id - 1, 0)]
# is_prologue = (grid_id == 0) | (group != prev_group)
#
# is_end = grid_id == (pl.num_programs(2) - 1)
# next_group = group_ids[jnp.where(is_end, grid_id, grid_id + 1)]
# is_epilogue = is_end | (group != next_group)
#
# group_size = group_offsets[group + 1] - group_offsets[group]
# nonzero_gs = group_size > 0
#
# @pl.when(is_prologue)
# def _zero():
#     acc_ref[...] = jnp.zeros((bk18, bn18), dtype=jnp.float32)
#
# @pl.when(nonzero_gs)
# def _compute():
#     mask_lhs = get_store_mask(grid_id, group_offsets, group_ids,
#                                m_tile_ids, bm18, bk18)
#     mask_rhs = get_store_mask(grid_id, group_offsets, group_ids,
#                                m_tile_ids, bm18, bn18)
#     lhs_masked = jnp.where(mask_lhs, lhs_ref[...], 0)
#     rhs_masked = jnp.where(mask_rhs, rhs_ref[...], 0)
#     acc_ref[...] += lhs_masked.T @ rhs_masked
#
# @pl.when(is_epilogue)
# def _store():
#     o_ref[...] = acc_ref[...]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 19: Understanding `emit_pipeline` — Annotated Walkthrough
#
# This is a **reading exercise**, not a coding puzzle. We walk through the
# tokamax `custom_buffered_pallas_call` to understand how production kernels
# use software pipelining for async DMA on TPU.
#
# ### Why pipelining?
#
# On TPU, data lives in **HBM** (32 GB, high bandwidth but high latency).
# Computation happens in **VMEM** (small, fast SRAM). Without pipelining:
#
# ```
# Time:  [DMA load] [compute] [DMA load] [compute] ...
#        ^^^idle^^^            ^^^idle^^^
# ```
#
# With double-buffered pipelining:
#
# ```
# DMA:      [load 0] [load 1] [load 2] [load 3] ...
# Compute:           [comp 0] [comp 1] [comp 2] ...
# ```
#
# The DMA engine and compute engine run in parallel, hiding memory latency.
#
# ### The `emit_pipeline` wrapper
#
# `pltpu.emit_pipeline` transforms a simple kernel into a pipelined one:
#
# ```python
# pltpu.emit_pipeline(
#     kernel_fn,          # Your original kernel
#     grid=grid,          # Iteration space
#     in_specs=in_specs,  # How to tile inputs
#     out_specs=out_specs, # How to tile outputs
#     dimension_semantics=("parallel", "arbitrary", "arbitrary"),
# )
# ```
#
# **`dimension_semantics`** tells the compiler about loop dependencies:
# - `"parallel"`: iterations are independent → can be reordered freely
# - `"arbitrary"`: iterations may have dependencies → must execute in order
#
# For ragged_dot: `(tiles_n, num_tiles, tiles_k)`:
# - `tiles_n` is `"parallel"` — different output columns are independent
# - `num_tiles` is `"arbitrary"` — tiles may share output locations
# - `tiles_k` is `"arbitrary"` — accumulation across K must be ordered

# %%
# Annotated version of the tokamax custom_buffered_pallas_call
# (Read and understand — no code to write)

import dataclasses

def annotated_custom_buffered_pallas_call(kernel, out_shape, grid_spec,
                                          compiler_params,
                                          input_buffer_count=None, **kw):
    """Wraps a kernel with emit_pipeline for async DMA pipelining.

    The outer pallas_call sees all data in HBM. Inside, emit_pipeline
    creates a software-pipelined loop that overlaps DMA with compute.
    """
    num_scalar_prefetch = grid_spec.num_scalar_prefetch

    def pipeline(*args_refs):
        # === Phase 1: Unpack grid and SMEM refs ===
        smem_refs = args_refs[1 : num_scalar_prefetch + 1]

        # === Phase 2: Bind SMEM refs to index maps ===
        def _augment_blockspec(bs):
            index_map_ = lambda *idxs: bs.index_map(*idxs, *smem_refs)
            return pl.BlockSpec(bs.block_shape, index_map_)

        in_specs = jax.tree.map(_augment_blockspec, grid_spec.in_specs)
        out_specs = jax.tree.map(_augment_blockspec, grid_spec.out_specs)

        # === Phase 3: Separate input/output/scratch refs ===
        input_output_refs = args_refs[num_scalar_prefetch + 1:]

        # === Phase 4: Emit the pipeline! ===
        pltpu.emit_pipeline(
            lambda *args: kernel(*smem_refs, *args),
            grid=grid_spec.grid,
            in_specs=in_specs,
            out_specs=out_specs,
            dimension_semantics=compiler_params.dimension_semantics,
        )(*input_output_refs)

    # The OUTER pallas_call has NO grid — single invocation.
    return pl.pallas_call(
        pipeline,
        out_shape,
        compiler_params=dataclasses.replace(compiler_params, dimension_semantics=()),
        in_specs=(
            jax.tree.map(lambda _: pl.BlockSpec(memory_space=pltpu.SMEM),
                        tuple(range(num_scalar_prefetch + 1))),
            jax.tree.map(lambda _: pl.BlockSpec(memory_space=pl.ANY),
                        tuple(grid_spec.in_specs)),
        ),
        out_specs=jax.tree.map(lambda _: pl.BlockSpec(memory_space=pl.ANY),
                               grid_spec.out_specs),
        **kw,
    )

print("emit_pipeline annotated walkthrough loaded.")
print("Study the code above — on real TPU, this is what makes the kernel fast!")

# %% [markdown]
# ### Comprehension questions
#
# Answer these by reading the annotated code above:
#
# 1. **Why does the outer `pallas_call` have no grid?** What happens
#    inside `emit_pipeline` that replaces the grid?
#
# 2. **What does `_augment_blockspec` do?** Why can't we pass the
#    original `grid_spec.in_specs` directly to `emit_pipeline`?
#
# 3. **Why is `tiles_n` labeled `"parallel"` but `num_tiles` and
#    `tiles_k` are `"arbitrary"`?** What would go wrong if we marked
#    `tiles_k` as `"parallel"`?
#
# 4. **Double buffering requires 2× the VMEM.** Why is this trade-off
#    worth it on TPU?
#
# <details><summary>Answers</summary>
#
# 1. The outer `pallas_call` runs once. Inside, `emit_pipeline` creates
#    its own software-pipelined loop over the grid, overlapping DMA with
#    compute. The grid iteration is "inlined" into the pipeline.
#
# 2. `_augment_blockspec` rebinds the index maps to include the SMEM
#    refs. The original index maps expect `(grid_idx, *smem_refs)`,
#    but `emit_pipeline` only passes grid indices. The wrapper curries
#    in the SMEM refs.
#
# 3. `"parallel"` means iterations are independent and can be reordered
#    or executed concurrently. `tiles_k` has accumulation dependencies —
#    K tile 2 adds to the same accumulator as K tile 1. Marking it
#    `"parallel"` would allow reordering, producing wrong results.
#    `num_tiles` has masked-store dependencies (boundary tiles are
#    visited by multiple groups sequentially).
#
# 4. TPU HBM latency is high (~100s of cycles). Without pipelining,
#    the MXU sits idle during every DMA load. With double buffering,
#    the MXU computes on buffer A while DMA fills buffer B. The 2×
#    VMEM cost is small compared to the throughput gain (often 2-3×
#    higher MFU).
# </details>

# %% [markdown]
# ---
# ## Puzzle 20: Challenge — Fused Ragged Dot + ReLU
#
# **Goal**: Combine Puzzle 17's ragged_dot with Puzzle 10's activation
# fusion: apply ReLU at the masked store step.
#
# ### Theory
#
# This ties together all the concepts from the notebook:
# - Zero/accumulate/store pattern (Puzzle 7)
# - Activation fusion (Puzzle 10)
# - Group metadata and masked stores (Puzzles 12, 14)
# - Ragged dot (Puzzle 17)
#
# The change from Puzzle 17 is minimal: apply `jnp.maximum(..., 0)` to
# the accumulated result **before** the masked store. This avoids a
# separate kernel pass for the activation.
#
# **Minimal scaffolding** — modify your Puzzle 17 solution.

# %%
G20 = 3
M20, K20, N20 = 1024, 256, 128
bm20, bk20, bn20 = 128, 128, 128

group_sizes_20 = jnp.array([300, 212, 512], dtype=jnp.int32)
tiles_k20 = K20 // bk20
tiles_n20 = N20 // bn20

(group_offsets_20, group_ids_20, m_tile_ids_20), num_tiles_20 = \
    make_group_metadata(group_sizes_20, M20, bm20)

# --- Reference ---
def fused_ragged_relu_spec(lhs, rhs, group_sizes):
    """ragged_dot + ReLU"""
    offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(group_sizes)])
    out = jnp.zeros((lhs.shape[0], rhs.shape[2]), dtype=jnp.float32)
    for g in range(len(group_sizes)):
        s, e = int(offsets[g]), int(offsets[g + 1])
        if s < e:
            out = out.at[s:e].set(jnp.maximum(lhs[s:e] @ rhs[g], 0))
    return out

# --- Kernel skeleton ---
def fused_ragged_relu_kernel(group_metadata_ref, group_offset_ref,
                              lhs_ref, rhs_ref, o_ref, acc_ref):
    group_offsets, group_ids, m_tile_ids = group_metadata_ref
    grid_id = pl.program_id(1)
    k_i = pl.program_id(2)

    pass  # YOUR CODE HERE
    # Modify Puzzle 17's solution: apply jnp.maximum(acc, 0) at the
    # masked store step.


# %%
lhs20 = jax.random.normal(jax.random.key(60), (M20, K20))
rhs20 = jax.random.normal(jax.random.key(61), (G20, K20, N20))
expected20 = fused_ragged_relu_spec(lhs20, rhs20, group_sizes_20)

group_metadata_20 = (group_offsets_20, group_ids_20, m_tile_ids_20)
group_offset_20 = jnp.array([0], dtype=jnp.int32)

actual20 = pl.pallas_call(
    fused_ragged_relu_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=[
            pl.BlockSpec((bm20, bk20), lhs_imap),
            pl.BlockSpec((None, bk20, bn20), rhs_imap),
        ],
        out_specs=pl.BlockSpec((bm20, bn20), out_imap),
        grid=(tiles_n20, num_tiles_20, tiles_k20),
        scratch_shapes=[pltpu.VMEM((bm20, bn20), jnp.float32)],
    ),
    out_shape=jax.ShapeDtypeStruct((M20, N20), jnp.float32),
    interpret=True,
)(group_metadata_20, group_offset_20, lhs20, rhs20)

total_rows20 = int(group_sizes_20.sum())
if jnp.allclose(actual20[:total_rows20], expected20[:total_rows20], atol=1e-2, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual20.shape})")
    print(f"  Verified {total_rows20} active rows")
else:
    max_err = float(jnp.max(jnp.abs(actual20[:total_rows20] - expected20[:total_rows20])))
    print(f"FAILED ✗  max error: {max_err:.6f}")

# %% [markdown]
# <details><summary>Hint — Full solution</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     acc_ref[...] = jnp.zeros((bm20, bn20), dtype=jnp.float32)
#
# acc_ref[...] += lhs_ref[...] @ rhs_ref[...]
#
# @pl.when(k_i == tiles_k20 - 1)
# def _store():
#     mask = get_store_mask(grid_id, group_offsets, group_ids,
#                           m_tile_ids, bm20, bn20)
#     acc = jnp.maximum(acc_ref[...], 0)  # fused ReLU!
#     o_ref[...] = jnp.where(mask, acc, o_ref[...].astype(acc.dtype))
# ```
# </details>

# %% [markdown]
# ---
# ## Summary
#
# You've built every component of a production ragged_dot kernel:
#
# | Concept | Puzzle |
# |---------|--------|
# | `pallas_call`, Refs, `ref[...]` syntax | 1 |
# | `grid`, `BlockSpec`, `program_id` | 2a, 3 |
# | Index map manipulation | 2b |
# | Broadcasting inside kernels | 4 |
# | Configure your own `pallas_call` | 5, 8, 13, 15 |
# | `@pl.when` conditional execution | 6 |
# | Matmul with scratch accumulator (VMEM) | 7 |
# | Batched matmul, `None` dim squeeze | 9 |
# | Activation fusion | 10, 20 |
# | `PrefetchScalarGridSpec` + SMEM | 11 |
# | `make_group_metadata` (CSR mapping) | 12 |
# | `get_store_mask`, `broadcasted_iota` | 14 |
# | Softmax kernel | 15 |
# | Simple grouped matmul (gmm) | 16 |
# | Full ragged_dot with masking | 17 |
# | tgmm, `pl.num_programs`, prologue/epilogue | 18 |
# | `emit_pipeline` for pipelining | 19 |
# | Fused ragged_dot + ReLU | 20 |
#
# ### What's left for production?
#
# The tokamax kernel adds several features beyond what we built:
# - **Quantization**: int8/int4 inputs with scale factors
# - **Transpose RHS**: `rhs` shape `(G, N, K)` for the backward pass
# - **Dynamic in-kernel quantization**: quantize lhs/rhs on the fly
# - **Activation fusion**: apply ReLU/tanh after the dot (Puzzle 10, 20)
# - **Sharding**: `group_offset` for processing a subset of groups
# - **Autotuning**: lookup tables for optimal tile sizes per problem shape
# - **Cost estimation**: FLOPs and bytes-accessed hints for the compiler
#
# But the **core kernel logic** is exactly what you implemented in
# Puzzles 16–18.
