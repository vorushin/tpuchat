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
# <a href="https://colab.research.google.com/github/vorushin/tpuchat/blob/master/10b_pallas_intermediate.ipynb?flush_caches=true" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 10b — Pallas Intermediate: Toward Grouped Operations
#
# This notebook builds on the Pallas foundations from `10a` and introduces the
# patterns needed for **grouped matrix multiplication** (ragged dot):
#
# - Explicit scratch memory and the zero/accumulate/store pattern
# - Batched matmul with a group dimension
# - **Scalar prefetch** for runtime-dependent index maps
# - **Group metadata**: CSR-style tile-to-group mapping
# - **Masked stores** with group boundaries
#
# All puzzles run on **CPU** via `interpret=True`.
#
# **Key reference**: tokamax `pallas_mosaic_tpu_kernel.py` — the production
# ragged_dot kernel we're building toward.

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
    """Run a Pallas kernel in interpret mode and compare against a reference spec."""
    expected = spec_fn(*inputs)
    if out_shape is None:
        out_shape = jax.ShapeDtypeStruct(expected.shape, expected.dtype)
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
# ## Puzzle 7: Matmul with `@pl.when` — Zero / Accumulate / Store
#
# **Goal**: Implement tiled matmul using the explicit **zero → accumulate →
# store** pattern with `@pl.when` guards.
#
# ### Theory
#
# In Puzzle 6 you saw scratch accumulators. Now let's make the pattern
# production-ready using `@pl.when(condition)` — Pallas's conditional
# execution primitive.
#
# The pattern (used in every tokamax kernel):
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
# On TPU hardware, this pattern maps well because `@pl.when` compiles to
# predicated execution — no branch divergence penalty.
#
# **New**: This time the accumulator is passed as a **scratch shape**
# parameter to `pallas_call`, not as an extra BlockSpec input.

# %%
M7, K7, N7 = 256, 512, 256
bm7, bk7, bn7 = 128, 256, 128
tiles_m7 = M7 // bm7
tiles_n7 = N7 // bn7
tiles_k7 = K7 // bk7

# --- Reference ---
def matmul_acc_spec(a, b):
    """a: (M7, K7), b: (K7, N7) → (M7, N7)"""
    return a @ b

# --- Kernel skeleton ---
def matmul_acc_kernel(a_ref, b_ref, o_ref, acc_ref):
    # a_ref: (bm7, bk7), b_ref: (bk7, bn7)
    # o_ref: (bm7, bn7) — output tile
    # acc_ref: (bm7, bn7) — scratch VMEM accumulator
    k_i = pl.program_id(2)
    pass  # YOUR CODE HERE
    # 1. @pl.when(k_i == 0) → zero acc_ref
    # 2. acc_ref[...] += a_ref[...] @ b_ref[...]  (use jax.lax.dot)
    # 3. @pl.when(k_i == tiles_k7 - 1) → copy acc_ref to o_ref


# %%
a7 = jax.random.normal(jax.random.key(10), (M7, K7))
b7 = jax.random.normal(jax.random.key(11), (K7, N7))

# Note: scratch_shapes provides the accumulator — no BlockSpec needed for it
check(matmul_acc_kernel, matmul_acc_spec, (a7, b7),
      grid=(tiles_m7, tiles_n7, tiles_k7),
      in_specs=[
          pl.BlockSpec((bm7, bk7), lambda m, n, k: (m, k)),
          pl.BlockSpec((bk7, bn7), lambda m, n, k: (k, n)),
      ],
      out_specs=pl.BlockSpec((bm7, bn7), lambda m, n, k: (m, n)),
      out_shape=jax.ShapeDtypeStruct((M7, N7), jnp.float32),
      scratch_shapes=[pltpu.VMEM((bm7, bn7), jnp.float32)])

# %% [markdown]
# <details><summary>💡 Hint</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     acc_ref[...] = jnp.zeros((bm7, bn7), dtype=jnp.float32)
#
# acc_ref[...] += jax.lax.dot(a_ref[...], b_ref[...])
#
# @pl.when(k_i == tiles_k7 - 1)
# def _store():
#     o_ref[...] = acc_ref[...]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 8: Batched Matmul — Group Dimension on RHS
#
# **Goal**: Compute `out[g] = lhs[g] @ rhs[g]` for `G` independent groups.
# The RHS has a leading group dimension.
#
# ### Theory
#
# In grouped/batched matmul, the RHS is `(G, K, N)` — a stack of `G`
# weight matrices. Each group `g` has its own `(K, N)` matrix.
#
# The grid adds a **group dimension**: `grid = (G,)`.
# Each group's BlockSpec selects one batch element at a time.
#
# **`None` vs integer in block_shape**: Using `None` means "load the entire
# axis and **squeeze** that dimension". The ref will NOT have that dim.
# Using an integer (e.g. `1`) means "load 1 element" — the ref keeps that
# dim with size 1. For a batch dim, `None` is convenient — the kernel
# sees simple 2D shapes like `(M, K)` instead of `(1, M, K)`.
#
# This is the precursor to ragged_dot, where different row-ranges of a
# single LHS matrix are multiplied by different group weight matrices.

# %%
G8, M8, K8, N8 = 4, 64, 128, 64

# --- Reference ---
def batched_matmul_spec(lhs, rhs):
    """lhs: (G, M8, K8), rhs: (G, K8, N8) → (G, M8, N8)"""
    return jnp.einsum('gmk,gkn->gmn', lhs, rhs)

# --- Kernel skeleton ---
def batched_matmul_kernel(lhs_ref, rhs_ref, o_ref):
    # With None in block_shape, the batch dim is squeezed:
    # lhs_ref: (M8, K8) — one group's lhs (batch dim squeezed)
    # rhs_ref: (K8, N8) — one group's rhs (batch dim squeezed)
    # o_ref: (M8, N8) — one group's output (batch dim squeezed)
    pass  # YOUR CODE HERE
    # Just compute lhs @ rhs and store


# %%
lhs8 = jax.random.normal(jax.random.key(12), (G8, M8, K8))
rhs8 = jax.random.normal(jax.random.key(13), (G8, K8, N8))

check(batched_matmul_kernel, batched_matmul_spec, (lhs8, rhs8),
      grid=(G8,),
      in_specs=[
          pl.BlockSpec((None, M8, K8), lambda g: (g, 0, 0)),
          pl.BlockSpec((None, K8, N8), lambda g: (g, 0, 0)),
      ],
      out_specs=pl.BlockSpec((None, M8, N8), lambda g: (g, 0, 0)),
      out_shape=jax.ShapeDtypeStruct((G8, M8, N8), jnp.float32))

# %% [markdown]
# <details><summary>💡 Hint</summary>
#
# ```python
# o_ref[...] = jax.lax.dot(lhs_ref[...], rhs_ref[...])
# ```
# With `None` in BlockSpec, the batch dimension is **squeezed** — the refs
# have shape `(M8, K8)` and `(K8, N8)` directly (no leading dim).
# </details>

# %% [markdown]
# ---
# ## Puzzle 9: Scalar Prefetch — Runtime Index Maps
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
# This is how tokamax passes `group_metadata` and `group_offset` to the
# ragged_dot kernel — they are scalar-prefetched.

# %%
G9 = 4
M9, K9, N9 = 64, 64, 64

# --- Reference ---
def permuted_matmul_spec(lhs, rhs, perm):
    """lhs: (G, M, K), rhs: (G, K, N), perm: (G,) → (G, M, N)
    out[i] = lhs[i] @ rhs[perm[i]]
    """
    return jnp.stack([lhs[i] @ rhs[perm[i]] for i in range(G9)])

# --- Kernel skeleton ---
def permuted_matmul_kernel(perm_ref, lhs_ref, rhs_ref, o_ref):
    # perm_ref: scalar-prefetched permutation array (in SMEM)
    # lhs_ref: (M9, K9) — current group's lhs
    # rhs_ref: (K9, N9) — permuted group's rhs (loaded via index map)
    # o_ref: (M9, N9) — output tile
    pass  # YOUR CODE HERE
    # Just compute the matmul — the index map already selected the right rhs!


# --- Index maps ---
def lhs_index_map(g, perm_ref):
    return (g, 0, 0)

def rhs_index_map(g, perm_ref):
    # Use the scalar-prefetched perm to look up which rhs group to load
    return (perm_ref[g], 0, 0)

def out_index_map(g, perm_ref):
    return (g, 0, 0)

# %%
lhs9 = jax.random.normal(jax.random.key(14), (G9, M9, K9))
rhs9 = jax.random.normal(jax.random.key(15), (G9, K9, N9))
perm9 = jnp.array([2, 0, 3, 1], dtype=jnp.int32)  # permutation

expected9 = permuted_matmul_spec(lhs9, rhs9, perm9)

actual9 = pl.pallas_call(
    permuted_matmul_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=[
            pl.BlockSpec((None, M9, K9), lhs_index_map),
            pl.BlockSpec((None, K9, N9), rhs_index_map),
        ],
        out_specs=pl.BlockSpec((None, M9, N9), out_index_map),
        grid=(G9,),
    ),
    out_shape=jax.ShapeDtypeStruct((G9, M9, N9), jnp.float32),
    interpret=True,
)(perm9, lhs9, rhs9)

if jnp.allclose(actual9, expected9, atol=1e-3):
    print(f"PASSED ✓  (shape={actual9.shape})")
else:
    max_err = float(jnp.max(jnp.abs(actual9 - expected9)))
    print(f"FAILED ✗  max error: {max_err:.6f}")

# %% [markdown]
# <details><summary>💡 Hint</summary>
#
# The index maps handle the permutation. The kernel is just:
# ```python
# o_ref[...] = jax.lax.dot(lhs_ref[...], rhs_ref[...])
# ```
# With `None` in BlockSpec, the batch dim is squeezed.
# </details>

# %% [markdown]
# ---
# ## Puzzle 10: Group Metadata — CSR-style Tile Mapping
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
# Given `group_sizes = [300, 212, 512]` with `tm = 128`:
#
# ```
# Row:  0              300    512         1024
#       ├── group 0 ───┤├─ g1 ─┤├── group 2 ──┤
#
# Tiles (tm=128):
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
# **Output arrays**:
# - `group_offsets`: `[0, 300, 512, 1024]` — cumsum with leading 0
# - `group_ids`: maps each grid index → group id
# - `m_tile_ids`: maps each grid index → which m-tile to process
# - `num_tiles`: total number of grid iterations needed
#
# The arrays can be longer than `num_tiles` (padded with the last group).

# %%
def make_group_metadata_reference(group_sizes, m, tm):
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
    tiles_m = m // tm
    group_ids_list = []
    m_tile_ids_list = []

    for t in range(tiles_m):
        tile_start = t * tm
        tile_end = (t + 1) * tm
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
# ### Step 10a: Group Offsets
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
print("Step 10a — compute_group_offsets: PASSED ✓")

# %% [markdown]
# <details><summary>💡 Hint</summary>
#
# ```python
# return jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(group_sizes)])
# ```
# </details>

# %% [markdown]
# ### Step 10b: Tiles per Group
#
# **Goal**: Compute how many tile visits each group requires, accounting
# for boundary tiles shared between neighbors.
#
# Round group starts DOWN and ends UP to tile boundaries:
# ```
# group 0: rows 0–299   → rounded to tiles [0, 384)   → 3 tiles
# group 1: rows 300–511  → rounded to tiles [256, 512)  → 2 tiles
# group 2: rows 512–1023 → rounded to tiles [512, 1024) → 4 tiles
#                                              total = 9 (> 8 physical tiles)
# ```
# Zero-size groups get zero tiles.

# %%
def compute_group_tiles(group_sizes, group_offsets, tm):
    """Number of tile visits per group (boundary tiles counted by both neighbors).

    Args:
        group_sizes: (G,) int32
        group_offsets: (G+1,) int32 from compute_group_offsets
        tm: tile size
    Returns:
        (G,) int32
    """
    pass  # YOUR CODE HERE
    # 1. Extract group_starts = group_offsets[:-1], group_ends = group_offsets[1:]
    # 2. Round starts DOWN to tile boundary: start // tm * tm
    # 3. Round ends UP to tile boundary: (end + tm - 1) // tm * tm
    # 4. Handle zero-size groups with jnp.where
    # 5. Return rounded_sizes // tm


# %%
# Aligned groups — no sharing: 256/128 = 2 tiles each
assert jnp.array_equal(
    compute_group_tiles(jnp.array([256, 256, 256, 256], dtype=jnp.int32),
                        jnp.array([0, 256, 512, 768, 1024], dtype=jnp.int32), 128),
    jnp.array([2, 2, 2, 2]))
# Unaligned — g0 gets 3, g1 gets 2, g2 gets 4
assert jnp.array_equal(
    compute_group_tiles(jnp.array([300, 212, 512], dtype=jnp.int32),
                        jnp.array([0, 300, 512, 1024], dtype=jnp.int32), 128),
    jnp.array([3, 2, 4]))
# Zero-size group gets 0 tiles
assert jnp.array_equal(
    compute_group_tiles(jnp.array([512, 0, 512], dtype=jnp.int32),
                        jnp.array([0, 512, 512, 1024], dtype=jnp.int32), 128),
    jnp.array([4, 0, 4]))
print("Step 10b — compute_group_tiles: PASSED ✓")

# %% [markdown]
# <details><summary>💡 Hint</summary>
#
# ```python
# group_starts = group_offsets[:-1]
# group_ends = group_offsets[1:]
# rounded_starts = (group_starts // tm * tm).astype(jnp.int32)
# rounded_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)
# rounded_sizes = jnp.where(group_sizes == 0, 0, rounded_ends - rounded_starts)
# return rounded_sizes // tm
# ```
# </details>

# %% [markdown]
# ### Step 10c: Group IDs
#
# **Goal**: Expand `group_tiles` into a flat mapping from grid index to
# group id.
#
# ```
# group_tiles = [3, 2, 4]  →  group_ids = [0,0,0, 1,1, 2,2,2,2]
# ```
#
# Use `jnp.repeat` with `total_repeat_length` for fixed output size.

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
print("Step 10c — compute_group_ids: PASSED ✓")

# %% [markdown]
# <details><summary>💡 Hint</summary>
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
# ### Step 10d: Tile Visits
#
# **Goal**: Compute how many times each physical m-tile is visited.
# Each tile is visited once by default. A group boundary falling *inside*
# a tile adds an extra visit.
#
# ```
# group_offsets = [0, 300, 512, 1024],  tm = 128
#
# Group 1 starts at row 300 → inside tile 2 (rows 256–383) → extra visit
# Group 2 starts at row 512 → tile-aligned → no extra visit
#
# tile_visits = [1, 1, 2, 1, 1, 1, 1, 1]
#                         ^ tile 2 visited by both g0 and g1
# ```
#
# Strategy: find non-aligned group starts, use `jnp.histogram` to count
# extra visits per tile, add 1 for the base visit.

# %%
def compute_tile_visits(group_sizes, group_offsets, tiles_m, tm):
    """Visit count per tile (1 + extra for each mid-tile group boundary).

    Args:
        group_sizes: (G,) int32
        group_offsets: (G+1,) int32
        tiles_m: M // tm
        tm: tile size
    Returns:
        (tiles_m,) int32
    """
    pass  # YOUR CODE HERE
    # 1. group_starts = group_offsets[:-1]
    # 2. Build mask: aligned (start % tm == 0) or empty group → no extra visit
    # 3. For non-aligned starts: partial_tile_id = start // tm
    #    For aligned/empty: use tiles_m as sentinel (out of histogram range)
    # 4. jnp.histogram(..., bins=tiles_m, range=(0, tiles_m - 1)) counts extras
    # 5. Return extra_visits + 1


# %%
# Aligned — all tiles visited once
assert compute_tile_visits(
    jnp.array([256, 256, 256, 256], dtype=jnp.int32),
    jnp.array([0, 256, 512, 768, 1024], dtype=jnp.int32), 8, 128
).tolist() == [1, 1, 1, 1, 1, 1, 1, 1]
# Unaligned — tile 2 (rows 256–383) visited twice
assert compute_tile_visits(
    jnp.array([300, 212, 512], dtype=jnp.int32),
    jnp.array([0, 300, 512, 1024], dtype=jnp.int32), 8, 128
).tolist() == [1, 1, 2, 1, 1, 1, 1, 1]
# Zero-size group — no extra visits
assert compute_tile_visits(
    jnp.array([512, 0, 512], dtype=jnp.int32),
    jnp.array([0, 512, 512, 1024], dtype=jnp.int32), 8, 128
).tolist() == [1, 1, 1, 1, 1, 1, 1, 1]
print("Step 10d — compute_tile_visits: PASSED ✓")

# %% [markdown]
# <details><summary>💡 Hint</summary>
#
# ```python
# group_starts = group_offsets[:-1]
# aligned_or_empty = ((group_starts % tm) == 0) | (group_sizes == 0)
# partial_tile_ids = jnp.where(aligned_or_empty, tiles_m, group_starts // tm)
# extra_visits = jnp.histogram(
#     partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1)
# )[0]
# return (extra_visits + 1).astype(jnp.int32)
# ```
# </details>

# %% [markdown]
# ### Step 10e: M-tile IDs
#
# **Goal**: Expand `tile_visits` into a flat mapping from grid index to
# m-tile id. Same idea as Step 10c but for tiles.
#
# ```
# tile_visits = [1,1,2,1,1,1,1,1]  →  m_tile_ids = [0,1,2,2,3,4,5,6,7]
#                    ^ tile 2 visited twice              ^^^ appears twice
# ```

# %%
def compute_m_tile_ids(tile_visits, tiles_m, max_len):
    """Flat array mapping grid index → m-tile id.

    Args:
        tile_visits: (tiles_m,) int32 from compute_tile_visits
        tiles_m: M // tm
        max_len: output array length (padded)
    Returns:
        (max_len,) int32
    """
    pass  # YOUR CODE HERE


# %%
assert compute_m_tile_ids(jnp.array([1,1,1,1,1,1,1,1]), 8, 11)[:8].tolist() == [0,1,2,3,4,5,6,7]
assert compute_m_tile_ids(jnp.array([1,1,2,1,1,1,1,1]), 8, 10)[:9].tolist() == [0,1,2,2,3,4,5,6,7]
print("Step 10e — compute_m_tile_ids: PASSED ✓")

# %% [markdown]
# <details><summary>💡 Hint</summary>
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
# ### Step 10f: Combined `make_group_metadata`
#
# **Goal**: Chain the 5 steps above into the complete function.

# %%
def make_group_metadata(group_sizes, m, tm):
    """Vectorized group metadata — chains steps 10a–10e.

    Args:
        group_sizes: jnp.array of shape (num_groups,), dtype int32
        m: total number of rows
        tm: tile size for m dimension

    Returns:
        (group_offsets, group_ids, m_tile_ids), num_tiles
    """
    num_groups = group_sizes.shape[0]
    tiles_m = m // tm
    max_len = tiles_m + num_groups - 1

    pass  # YOUR CODE HERE
    # Call compute_group_offsets, compute_group_tiles, compute_group_ids,
    # compute_tile_visits, compute_m_tile_ids in sequence.
    # num_tiles = int(group_tiles.sum())

    return (group_offsets, group_ids, m_tile_ids), num_tiles


# %%
# Integration tests — compare against reference
def check_metadata(name, group_sizes, m, tm):
    ref, ref_nt = make_group_metadata_reference(group_sizes, m, tm)
    yours, your_nt = make_group_metadata(group_sizes, m, tm)
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
check_metadata("Zero-size group",
               jnp.array([512, 0, 512], dtype=jnp.int32), 1024, 128)

# %% [markdown]
# <details><summary>💡 Hint</summary>
#
# ```python
# group_offsets = compute_group_offsets(group_sizes)
# group_tiles = compute_group_tiles(group_sizes, group_offsets, tm)
# group_ids = compute_group_ids(group_tiles, num_groups, max_len)
# tile_visits = compute_tile_visits(group_sizes, group_offsets, tiles_m, tm)
# m_tile_ids = compute_m_tile_ids(tile_visits, tiles_m, max_len)
# num_tiles = int(group_tiles.sum())
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 11: Masked Store with Group Boundaries
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
# - `m_tile_ids[grid_id] * tm` → first row of current tile
#
# ```python
# row_indices = tile_start + jnp.arange(tm)
# mask = (row_indices >= group_start) & (row_indices < group_end)
# ```
#
# This is exactly the `_get_store_mask` pattern from tokamax.

# %%
M11 = 1024
N11 = 64
bm11 = 128
G11 = 3

# Group sizes that create partial tiles
group_sizes_11 = jnp.array([300, 212, 512], dtype=jnp.int32)

# Pre-compute metadata using the reference function
(group_offsets_11, group_ids_11, m_tile_ids_11), num_tiles_11 = \
    make_group_metadata_reference(group_sizes_11, M11, bm11)

# --- Reference ---
def masked_copy_spec(x, group_offsets, group_ids, m_tile_ids):
    """Copy x to output, but only rows within their assigned group.

    For each grid iteration, we look up which group and tile we're processing.
    Only rows belonging to that group get copied; others stay zero.
    """
    out = jnp.zeros_like(x)
    for grid_id in range(num_tiles_11):
        g = int(group_ids[grid_id])
        tile_id = int(m_tile_ids[grid_id])
        g_start = int(group_offsets[g])
        g_end = int(group_offsets[g + 1])
        t_start = tile_id * bm11
        t_end = t_start + bm11
        for row in range(t_start, t_end):
            if g_start <= row < g_end:
                out = out.at[row].set(x[row])
    return out

# --- Kernel skeleton ---
def masked_copy_kernel(group_offsets_ref, group_ids_ref, m_tile_ids_ref,
                       x_ref, o_ref):
    # group_offsets_ref, group_ids_ref, m_tile_ids_ref: metadata in SMEM
    # x_ref: (bm11, N11) — tile of input
    # o_ref: (bm11, N11) — tile of output
    grid_id = pl.program_id(0)
    pass  # YOUR CODE HERE
    # 1. Look up group_id = group_ids_ref[grid_id]
    # 2. Look up m_tile = m_tile_ids_ref[grid_id]
    # 3. Get group_start and group_end from group_offsets_ref
    # 4. Build row mask: tile_start + arange(tm) in [group_start, group_end)
    # 5. Store: o_ref[...] = jnp.where(mask[:, None], x_ref[...], o_ref[...])


# %%
x11 = jax.random.normal(jax.random.key(20), (M11, N11))
expected11 = masked_copy_spec(x11, group_offsets_11, group_ids_11, m_tile_ids_11)

# Pack metadata into the scalar prefetch args
actual11 = pl.pallas_call(
    masked_copy_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        in_specs=[pl.BlockSpec((bm11, N11), lambda i, go, gi, mt: (mt[i], 0))],
        out_specs=pl.BlockSpec((bm11, N11), lambda i, go, gi, mt: (mt[i], 0)),
        grid=(num_tiles_11,),
    ),
    out_shape=jax.ShapeDtypeStruct((M11, N11), jnp.float32),
    interpret=True,
)(group_offsets_11, group_ids_11, m_tile_ids_11, x11)

if jnp.allclose(actual11, expected11, atol=1e-5):
    print(f"PASSED ✓  (shape={actual11.shape})")
else:
    max_err = float(jnp.max(jnp.abs(actual11 - expected11)))
    print(f"FAILED ✗  max error: {max_err:.6f}")
    # Show a few rows around the first group boundary
    print(f"  Around row 300 (group boundary):")
    print(f"    expected[296:304] nonzero cols: {[int(expected11[r].any()) for r in range(296, 304)]}")
    print(f"    actual[296:304] nonzero cols:   {[int(actual11[r].any()) for r in range(296, 304)]}")

# %% [markdown]
# <details><summary>💡 Hint</summary>
#
# ```python
# group_id = group_ids_ref[grid_id]
# m_tile = m_tile_ids_ref[grid_id]
# group_start = group_offsets_ref[group_id]
# group_end = group_offsets_ref[group_id + 1]
# tile_start = m_tile * bm11
#
# # Build 2D mask: (bm11, N11)
# row_ids = tile_start + jax.lax.broadcasted_iota(jnp.int32, (bm11, N11), 0)
# mask = (row_ids >= group_start) & (row_ids < group_end)
#
# o_ref[...] = jnp.where(mask, x_ref[...], o_ref[...])
# ```
# </details>

# %% [markdown]
# ---
# ## Summary
#
# You've learned the intermediate Pallas patterns:
#
# | Concept | Puzzle |
# |---------|--------|
# | `@pl.when` zero/accumulate/store | 7 |
# | Batched matmul with group dim on RHS | 8 |
# | `PrefetchScalarGridSpec` + SMEM | 9 |
# | Group metadata (CSR-style tile mapping) | 10 |
# | Masked stores with group boundaries | 11 |
#
# **Next**: `10c_pallas_ragged_dot.py` — combining everything into the full
# ragged_dot (grouped matmul) kernel.
