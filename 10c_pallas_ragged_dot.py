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
# # 10c — Pallas Ragged Dot: Building Grouped Matmul
#
# This is the culminating notebook. We combine everything from `10a` (Pallas
# foundations) and `10b` (intermediate patterns) to build the **ragged_dot**
# (grouped matrix multiplication) kernel — the core operation behind
# dropless MoE (MegaBlocks) on TPU.
#
# **What is ragged_dot?**
#
# Given:
# - `lhs`: `(M, K)` — rows from multiple groups concatenated together
# - `rhs`: `(G, K, N)` — one weight matrix per group
# - `group_sizes`: `(G,)` — how many rows each group has in lhs
#
# Compute: `out[group_start:group_end] = lhs[group_start:group_end] @ rhs[g]`
# for each group `g`, producing `out: (M, N)`.
#
# **Puzzles**:
# - **12**: Simple GMM — equal groups, tile-aligned
# - **13**: Full ragged_dot — unequal groups with partial tiles
# - **14**: Transpose GMM (tgmm) — the backward pass
# - **15**: Understanding `emit_pipeline` — annotated walkthrough
#
# **Key reference**: tokamax `pallas_mosaic_tpu_kernel.py`
#
# All puzzles run on **CPU** via `interpret=True`.

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


# %% [markdown]
# ### Provided utilities
#
# These are the building blocks from notebooks 10a and 10b. They're provided
# here so you can focus on the kernel logic.

# %%
def make_group_metadata(group_sizes, m, tm, *, visit_empty_groups=False):
    """Compute tile-to-group mapping for ragged_dot.

    Returns:
        (group_offsets, group_ids, m_tile_ids), num_tiles

    This is the same algorithm as tokamax's make_group_metadata.
    """
    num_groups = group_sizes.shape[0]
    tiles_m = m // tm

    # CSR-style offsets
    group_ends = jnp.cumsum(group_sizes)
    group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])

    # Round boundaries to tile boundaries
    group_starts = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]])
    rounded_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)
    rounded_starts = (group_starts // tm * tm).astype(jnp.int32)
    rounded_sizes = rounded_ends - rounded_starts
    rounded_sizes = jnp.where(group_sizes == 0, 0, rounded_sizes)
    group_tiles = rounded_sizes // tm

    if visit_empty_groups:
        group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

    # Build group_ids
    group_ids = jnp.repeat(
        jnp.arange(num_groups, dtype=jnp.int32),
        group_tiles,
        total_repeat_length=tiles_m + num_groups - 1,
    )

    # Build m_tile_ids
    partial_mask = ((group_offsets[:-1] % tm) == 0) | (group_sizes == 0)
    if visit_empty_groups:
        partial_mask = jnp.where(group_sizes == 0, 0, partial_mask)
    partial_tile_ids = jnp.where(partial_mask, tiles_m, group_offsets[:-1] // tm)
    tile_visits = (
        jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0] + 1
    )
    m_tile_ids = jnp.repeat(
        jnp.arange(tiles_m, dtype=jnp.int32),
        tile_visits.astype(jnp.int32),
        total_repeat_length=tiles_m + num_groups - 1,
    )

    num_tiles = int(group_tiles.sum())
    return (group_offsets, group_ids, m_tile_ids), num_tiles


def get_store_mask(grid_id, group_offsets, group_ids, m_tile_ids, tm, tn):
    """Build a (tm, tn) boolean mask for rows belonging to the current group."""
    group_id = group_ids[grid_id]
    group_start = group_offsets[group_id]
    group_end = group_offsets[group_id + 1]
    m_id = m_tile_ids[grid_id] * tm
    iota = jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + m_id
    return (iota >= group_start) & (iota < group_end)


# %% [markdown]
# ---
# ## Puzzle 12: Simple Grouped Matmul — Equal Groups, Tile-Aligned
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
# - `num_tiles`: M tiles across all groups (arbitrary — may revisit same output)
# - `tiles_k`: K reduction dimension (arbitrary — accumulates)
#
# **Kernel structure** (same as tokamax `gmm`):
# 1. Get `grid_id = program_id(1)`, `k_i = program_id(2)`
# 2. Look up `group_id` and `m_tile_id` from prefetched metadata
# 3. Zero accumulator when `k_i == 0`
# 4. Accumulate `lhs_tile @ rhs_tile`
# 5. Store on last K tile
#
# **Index maps** (the key to understanding ragged_dot):
# - `lhs_index_map`: `(n_i, grid_id, k_i) → (m_tile_ids[grid_id], k_i)`
# - `rhs_index_map`: `(n_i, grid_id, k_i) → (group_ids[grid_id], k_i, n_i)`
# - `out_index_map`: `(n_i, grid_id, k_i) → (m_tile_ids[grid_id], n_i)`
#
# The `group_ids` lookup in `rhs_index_map` is what routes each tile to
# the correct group's weight matrix!

# %%
G12 = 4
M12, K12, N12 = 512, 256, 128
tm12, tk12, tn12 = 128, 128, 128

# Equal groups
group_sizes_12 = jnp.array([M12 // G12] * G12, dtype=jnp.int32)
tiles_k12 = K12 // tk12
tiles_n12 = N12 // tn12

# Pre-compute metadata
(group_offsets_12, group_ids_12, m_tile_ids_12), num_tiles_12 = \
    make_group_metadata(group_sizes_12, M12, tm12)

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
    # group_metadata_ref: tuple of (group_offsets, group_ids, m_tile_ids) in SMEM
    # group_offset_ref: unused here (for sharding)
    # lhs_ref: (tm12, tk12) — tile of lhs
    # rhs_ref: (tk12, tn12) — tile of rhs (group dim squeezed by None in BlockSpec)
    # o_ref: (tm12, tn12) — output tile
    # acc_ref: (tm12, tn12) — scratch accumulator
    grid_id = pl.program_id(1)
    k_i = pl.program_id(2)

    pass  # YOUR CODE HERE
    # 1. @pl.when(k_i == 0): zero acc_ref
    # 2. acc_ref[...] += jax.lax.dot(lhs_ref[...], rhs_ref[...])
    # 3. @pl.when(k_i == tiles_k12 - 1): store acc_ref → o_ref


# --- Index maps (these are provided — study them!) ---
def lhs_imap(n_i, grid_id, k_i, group_meta_ref, group_offset_ref):
    _, _, m_tile_ids = group_meta_ref
    return (m_tile_ids[grid_id], k_i)

def rhs_imap(n_i, grid_id, k_i, group_meta_ref, group_offset_ref):
    _, group_ids, _ = group_meta_ref
    return (group_ids[grid_id], k_i, n_i)

def out_imap(n_i, grid_id, k_i, group_meta_ref, group_offset_ref):
    _, _, m_tile_ids = group_meta_ref
    return (m_tile_ids[grid_id], n_i)


# %%
lhs12 = jax.random.normal(jax.random.key(30), (M12, K12))
rhs12 = jax.random.normal(jax.random.key(31), (G12, K12, N12))
expected12 = simple_gmm_spec(lhs12, rhs12, group_sizes_12)

group_metadata_12 = (group_offsets_12, group_ids_12, m_tile_ids_12)
group_offset_12 = jnp.array([0], dtype=jnp.int32)

actual12 = pl.pallas_call(
    simple_gmm_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=[
            pl.BlockSpec((tm12, tk12), lhs_imap),
            pl.BlockSpec((None, tk12, tn12), rhs_imap),
        ],
        out_specs=pl.BlockSpec((tm12, tn12), out_imap),
        grid=(tiles_n12, num_tiles_12, tiles_k12),
        scratch_shapes=[pltpu.VMEM((tm12, tn12), jnp.float32)],
    ),
    out_shape=jax.ShapeDtypeStruct((M12, N12), jnp.float32),
    interpret=True,
)(group_metadata_12, group_offset_12, lhs12, rhs12)

if jnp.allclose(actual12, expected12, atol=1e-2, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual12.shape})")
else:
    max_err = float(jnp.max(jnp.abs(actual12 - expected12)))
    print(f"FAILED ✗  max error: {max_err:.6f}")
    print(f"  Expected[:2,:4]:\n{expected12[:2,:4]}")
    print(f"  Actual[:2,:4]:\n{actual12[:2,:4]}")

# %% [markdown]
# <details><summary>💡 Hint</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     acc_ref[...] = jnp.zeros((tm12, tn12), dtype=jnp.float32)
#
# # With None in BlockSpec, the group dim is squeezed — rhs_ref is (tk, tn)
# acc_ref[...] += jax.lax.dot(lhs_ref[...], rhs_ref[...])
#
# @pl.when(k_i == tiles_k12 - 1)
# def _store():
#     o_ref[...] = acc_ref[...]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 13: Full Ragged Dot — Unequal Groups
#
# **Goal**: Handle **variable group sizes** where tiles can straddle group
# boundaries. This is the real ragged_dot.
#
# ### Theory
#
# The only difference from Puzzle 12: when groups are unequal, a tile may
# be visited **multiple times** (once per group it straddles). On each visit,
# the kernel must **mask** the store so only rows belonging to the current
# group are written.
#
# `make_group_metadata` handles all the complexity — the `group_ids` and
# `m_tile_ids` arrays already encode the repeated visits. The kernel just
# needs to add the mask at store time:
#
# ```python
# mask = get_store_mask(grid_id, group_offsets, group_ids, m_tile_ids, tm, tn)
# o_ref[...] = jnp.where(mask, acc[...], o_ref[...])
# ```
#
# This preserves previously-written values from other groups in the same tile.
#
# ```
# Tile at row 256, tm=128:
# ┌────────────────────────┐
# │ rows 256-299: group 0  │ ← Visit 1: mask=True for rows 256-299
# │ rows 300-383: group 1  │ ← Visit 2: mask=True for rows 300-383
# └────────────────────────┘
# ```

# %%
G13 = 3
M13, K13, N13 = 1024, 256, 128
tm13, tk13, tn13 = 128, 128, 128

# Unequal groups!
group_sizes_13 = jnp.array([300, 212, 512], dtype=jnp.int32)
tiles_k13 = K13 // tk13
tiles_n13 = N13 // tn13

(group_offsets_13, group_ids_13, m_tile_ids_13), num_tiles_13 = \
    make_group_metadata(group_sizes_13, M13, tm13)

print(f"M={M13}, G={G13}, group_sizes={group_sizes_13.tolist()}")
print(f"num_tiles={num_tiles_13} (vs {M13//tm13} base tiles)")
print(f"group_ids[:num_tiles]={group_ids_13[:num_tiles_13].tolist()}")
print(f"m_tile_ids[:num_tiles]={m_tile_ids_13[:num_tiles_13].tolist()}")

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
    # Same as Puzzle 12, but on the last k_i, apply mask:
    # 1. @pl.when(k_i == 0): zero acc
    # 2. accumulate
    # 3. @pl.when(k_i == tiles_k13 - 1):
    #    mask = get_store_mask(grid_id, group_offsets, group_ids,
    #                          m_tile_ids, tm13, tn13)
    #    o_ref[...] = jnp.where(mask, acc_ref[...], o_ref[...])


# %%
lhs13 = jax.random.normal(jax.random.key(40), (M13, K13))
rhs13 = jax.random.normal(jax.random.key(41), (G13, K13, N13))
expected13 = ragged_dot_spec(lhs13, rhs13, group_sizes_13)

group_metadata_13 = (group_offsets_13, group_ids_13, m_tile_ids_13)
group_offset_13 = jnp.array([0], dtype=jnp.int32)

actual13 = pl.pallas_call(
    ragged_dot_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=[
            pl.BlockSpec((tm13, tk13), lhs_imap),
            pl.BlockSpec((None, tk13, tn13), rhs_imap),
        ],
        out_specs=pl.BlockSpec((tm13, tn13), out_imap),
        grid=(tiles_n13, num_tiles_13, tiles_k13),
        scratch_shapes=[pltpu.VMEM((tm13, tn13), jnp.float32)],
    ),
    out_shape=jax.ShapeDtypeStruct((M13, N13), jnp.float32),
    interpret=True,
)(group_metadata_13, group_offset_13, lhs13, rhs13)

# Only check rows within the sum of group_sizes
total_rows = int(group_sizes_13.sum())
if jnp.allclose(actual13[:total_rows], expected13[:total_rows], atol=1e-2, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual13.shape})")
    print(f"  Verified {total_rows} active rows")
else:
    max_err = float(jnp.max(jnp.abs(actual13[:total_rows] - expected13[:total_rows])))
    print(f"FAILED ✗  max error: {max_err:.6f}")

# %% [markdown]
# <details><summary>💡 Hint</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     acc_ref[...] = jnp.zeros((tm13, tn13), dtype=jnp.float32)
#
# # None in rhs BlockSpec squeezes the group dim — rhs_ref is (tk, tn)
# acc_ref[...] += jax.lax.dot(lhs_ref[...], rhs_ref[...])
#
# @pl.when(k_i == tiles_k13 - 1)
# def _store():
#     mask = get_store_mask(grid_id, group_offsets, group_ids,
#                           m_tile_ids, tm13, tn13)
#     acc = acc_ref[...]
#     o_ref[...] = jnp.where(mask, acc, o_ref[...].astype(acc.dtype))
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 14: Transpose Grouped Matmul (tgmm)
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
# - `lhs`: `(K, M)` (transposed) → tiles `(tm, tk)` after transposing to `(M, K)`
# - `rhs`: `(M, N)` → tiles `(tm, tn)`
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

# %%
G14 = 3
M14, K14, N14 = 1024, 128, 128
tm14, tk14, tn14 = 128, 128, 128

group_sizes_14 = jnp.array([384, 256, 384], dtype=jnp.int32)
tiles_k14 = K14 // tk14
tiles_n14 = N14 // tn14

(group_offsets_14, group_ids_14, m_tile_ids_14), num_tiles_14 = \
    make_group_metadata(group_sizes_14, M14, tm14, visit_empty_groups=True)

print(f"group_sizes={group_sizes_14.tolist()}, num_tiles={num_tiles_14}")
print(f"group_ids={group_ids_14[:num_tiles_14].tolist()}")

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
    # lhs_ref: (tm14, tk14) — tile of lhs (M, K) — note: we transpose lhs_t to (M, K)
    # rhs_ref: (tm14, tn14) — tile of rhs
    # o_ref: (tk14, tn14) — output tile for one group (None dim squeezed)
    # acc_ref: (tk14, tn14) — scratch accumulator
    group_offsets, group_ids, m_tile_ids = group_metadata_ref
    grid_id = pl.program_id(2)  # tgmm grid: (tiles_n, tiles_k, num_tiles)

    pass  # YOUR CODE HERE
    # 1. Detect prologue: grid_id == 0 or group changed
    #    group = group_ids[grid_id]
    #    prev_group = group_ids[jnp.where(grid_id > 0, grid_id - 1, 0)]
    #    is_prologue = (grid_id == 0) | (group != prev_group)
    #
    # 2. Detect epilogue: last grid point or group about to change
    #    is_end = grid_id == (pl.num_programs(2) - 1)
    #    next_group = group_ids[jnp.where(is_end, grid_id, grid_id + 1)]
    #    is_epilogue = is_end | (group != next_group)
    #
    # 3. @pl.when(is_prologue): zero acc
    # 4. Mask lhs and rhs to group boundaries, then acc += lhs.T @ rhs
    # 5. @pl.when(is_epilogue): store acc to output


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
lhs_t_14 = jax.random.normal(jax.random.key(50), (K14, M14))
rhs14 = jax.random.normal(jax.random.key(51), (M14, N14))
expected14 = tgmm_spec(lhs_t_14, rhs14, group_sizes_14)

# tgmm works on (M, K) internally — transpose lhs
lhs14 = lhs_t_14.T  # (M, K)

group_metadata_14 = (group_offsets_14, group_ids_14, m_tile_ids_14)
group_offset_14 = jnp.array([0], dtype=jnp.int32)

actual14 = pl.pallas_call(
    tgmm_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=[
            pl.BlockSpec((tm14, tk14), tgmm_lhs_imap),
            pl.BlockSpec((tm14, tn14), tgmm_rhs_imap),
        ],
        out_specs=pl.BlockSpec((None, tk14, tn14), tgmm_out_imap),
        grid=(tiles_n14, tiles_k14, num_tiles_14),
        scratch_shapes=[pltpu.VMEM((tk14, tn14), jnp.float32)],
    ),
    out_shape=jax.ShapeDtypeStruct((G14, K14, N14), jnp.float32),
    interpret=True,
)(group_metadata_14, group_offset_14, lhs14, rhs14)

if jnp.allclose(actual14, expected14, atol=1e-1, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual14.shape})")
else:
    max_err = float(jnp.max(jnp.abs(actual14 - expected14)))
    print(f"FAILED ✗  max error: {max_err:.6f}")
    print(f"  Expected[0,:2,:4]:\n{expected14[0,:2,:4]}")
    print(f"  Actual[0,:2,:4]:\n{actual14[0,:2,:4]}")

# %% [markdown]
# <details><summary>💡 Hint</summary>
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
#     acc_ref[...] = jnp.zeros((tk14, tn14), dtype=jnp.float32)
#
# @pl.when(nonzero_gs)
# def _compute():
#     # Mask lhs and rhs to group boundaries
#     mask_lhs = get_store_mask(grid_id, group_offsets, group_ids,
#                                m_tile_ids, tm14, tk14)
#     mask_rhs = get_store_mask(grid_id, group_offsets, group_ids,
#                                m_tile_ids, tm14, tn14)
#     lhs_masked = jnp.where(mask_lhs, lhs_ref[...], 0)
#     rhs_masked = jnp.where(mask_rhs, rhs_ref[...], 0)
#     acc_ref[...] += jax.lax.dot(lhs_masked.T, rhs_masked)
#
# @pl.when(is_epilogue)
# def _store():
#     o_ref[...] = acc_ref[...]  # None in BlockSpec squeezes group dim
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 15: Understanding `emit_pipeline` — Annotated Walkthrough
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
# Time:  [DMA load 0] [DMA load 1 | compute 0] [DMA load 2 | compute 1] ...
#                      ^^^^^^^^^^overlap^^^^^^^^^
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
# - `num_tiles` is `"arbitrary"` — tiles may share output locations (partial tiles)
# - `tiles_k` is `"arbitrary"` — accumulation across K must be ordered
#
# ### The `custom_buffered_pallas_call` pattern
#
# tokamax wraps `emit_pipeline` in an outer `pallas_call` that handles
# the scalar prefetch data:
#
# ```python
# def custom_buffered_pallas_call(kernel, out_shape, grid_spec, compiler_params, ...):
#     def pipeline(*args_refs):
#         # 1. Unpack dynamic grid elements from SMEM
#         # 2. Unpack scalar prefetch refs, bind to index maps
#         # 3. Separate input/output/scratch refs
#         # 4. Call emit_pipeline with bound kernel
#         pltpu.emit_pipeline(
#             lambda *args: kernel(*smem_refs, *args, *scratch_refs),
#             grid=grid,
#             in_specs=...,
#             out_specs=...,
#             dimension_semantics=compiler_params.dimension_semantics,
#         )(*input_output_refs)
#
#     # Outer pallas_call with no grid (single invocation)
#     # All real data is in HBM (BlockSpec(memory_space=pl.ANY))
#     # Scalar prefetch data is in SMEM
#     return pl.pallas_call(pipeline, ...)
# ```
#
# ### `input_buffer_count` and lookahead
#
# For deeper pipelining, you can use more than 2 buffers:
# ```python
# pl.Buffered(buffer_count=3, use_lookahead=True)
# ```
# This prefetches tiles further ahead, hiding more latency at the cost of
# more VMEM usage.
#
# ### Challenge (optional)
#
# Try modifying the simple GMM kernel from Puzzle 12 to use `emit_pipeline`.
# You'll need to:
# 1. Create an outer `pallas_call` with no grid
# 2. Pass all data as HBM BlockSpecs
# 3. Inside, call `emit_pipeline` with your kernel
#
# This won't make a difference in `interpret=True` mode, but on real TPU
# hardware it can significantly improve MXU utilization.

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
        # args_refs[0] contains dynamic grid dimensions (packed in SMEM)
        # args_refs[1:num_scalar_prefetch+1] are scalar-prefetched arrays

        smem_refs = args_refs[1 : num_scalar_prefetch + 1]

        # === Phase 2: Bind SMEM refs to index maps ===
        # The original index maps expect (grid_idx, ..., *smem_refs)
        # We bind the smem_refs to create standard index maps
        def _augment_blockspec(bs):
            index_map_ = lambda *idxs: bs.index_map(*idxs, *smem_refs)
            return pl.BlockSpec(bs.block_shape, index_map_)

        in_specs = jax.tree.map(_augment_blockspec, grid_spec.in_specs)
        out_specs = jax.tree.map(_augment_blockspec, grid_spec.out_specs)

        # === Phase 3: Separate input/output/scratch refs ===
        input_output_refs = args_refs[num_scalar_prefetch + 1:]
        # (scratch_refs would come after if present)

        # === Phase 4: Emit the pipeline! ===
        # This creates a software-pipelined loop with:
        # - Double buffering (or more with input_buffer_count)
        # - Async DMA overlapped with compute
        # - Automatic barrier insertion
        pltpu.emit_pipeline(
            lambda *args: kernel(*smem_refs, *args),
            grid=grid_spec.grid,
            in_specs=in_specs,
            out_specs=out_specs,
            dimension_semantics=compiler_params.dimension_semantics,
        )(*input_output_refs)

    # The OUTER pallas_call has NO grid — single invocation.
    # All actual data sits in HBM (BlockSpec(memory_space=pl.ANY)).
    # SMEM data is prefetched before the kernel starts.
    return pl.pallas_call(
        pipeline,
        out_shape,
        compiler_params=dataclasses.replace(compiler_params, dimension_semantics=()),
        in_specs=(
            # SMEM specs for scalar prefetch args:
            jax.tree.map(lambda _: pl.BlockSpec(memory_space=pltpu.SMEM),
                        tuple(range(num_scalar_prefetch + 1))),
            # HBM specs for input/output data:
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
# ---
# ## Summary: The Full Ragged Dot Architecture
#
# You've now built every component of the tokamax ragged_dot kernel:
#
# | Component | Where |
# |-----------|-------|
# | `pallas_call`, grid, BlockSpec | 10a: Puzzles 1-6 |
# | `@pl.when`, scratch accumulator | 10b: Puzzle 7 |
# | Group dimension on RHS | 10b: Puzzle 8 |
# | `PrefetchScalarGridSpec` + SMEM | 10b: Puzzle 9 |
# | `make_group_metadata` (CSR mapping) | 10b: Puzzle 10 |
# | `get_store_mask` (group boundaries) | 10b: Puzzle 11 |
# | Simple grouped matmul (gmm) | 10c: Puzzle 12 |
# | Full ragged_dot with masking | 10c: Puzzle 13 |
# | Transpose grouped matmul (tgmm) | 10c: Puzzle 14 |
# | `emit_pipeline` for pipelining | 10c: Puzzle 15 |
#
# ### What's left for production?
#
# The tokamax kernel adds several features beyond what we built:
# - **Quantization**: int8/int4 inputs with scale factors
# - **Transpose RHS**: `rhs` shape `(G, N, K)` for the backward pass
# - **Dynamic in-kernel quantization**: quantize lhs/rhs on the fly
# - **Activation fusion**: apply ReLU/tanh after the dot
# - **Sharding**: `group_offset` for processing a subset of groups
# - **Autotuning**: lookup tables for optimal tile sizes per problem shape
# - **Cost estimation**: FLOPs and bytes-accessed hints for the compiler
#
# But the **core kernel logic** is exactly what you implemented in Puzzles 12-14.
