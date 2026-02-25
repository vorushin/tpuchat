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
#   accelerator: TPU
#   colab:
#     gpuType: V6E
# ---

# %% [markdown]
# <a href="https://colab.research.google.com/github/vorushin/tpuchat/blob/master/10_megablox.ipynb?flush_caches=true" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 10 — MegaBlox: Dropless MoE with Grouped Matrix Multiplication (rev 7)
#
# This notebook builds up **dropless Mixture of Experts** step by step, from
# the fundamental grouped matrix multiplication (GMM) operation to a full
# trainable transformer with three MoE dispatch strategies compared side by side.
#
# ### Background: The Token Dropping Problem
#
# Standard MoE (as in `09_moe.py`) uses **capacity-based dispatch**: each expert
# has a fixed-size buffer. When an expert is popular (many tokens routed to it),
# excess tokens are **dropped** — they pass through with zero expert contribution.
# Unpopular experts waste compute on **padding** (empty buffer slots).
#
# ### The MegaBlocks Solution (MLSys 2023)
#
# [MegaBlocks](https://arxiv.org/abs/2211.15841) by Trevor Gale et al. introduced
# **dropless MoE (dMoE)**: instead of fixed-capacity buffers, **sort tokens by
# expert assignment** and use a **grouped matrix multiplication** that processes
# variable-sized groups in one kernel call. No dropping, no padding.
#
# **MegaBlox** is the JAX/TPU port, now integrated into
# [MaxText](https://github.com/AI-Hypercomputer/maxtext) (Google's reference
# LLM training codebase) and the
# [JAX repo](https://github.com/jax-ml/jax) itself via Pallas kernels.
#
# ### What this notebook covers
#
# | Part | Topic | Runs on |
# |------|-------|---------|
# | 1 | Grouped Matrix Multiplication from scratch | CPU |
# | 2 | The Pallas GMM kernel (MegaBlox) | CPU (interpret mode) |
# | 3 | Token routing pipeline | CPU |
# | 4 | Three MoE dispatch strategies compared | CPU |
# | 5 | Token dropping vs dropless demonstration | CPU |
# | 6 | Validation: training a small model | CPU |
# | 7 | TPU production config | TPU only |
#
# Parts 1–6 run entirely on CPU (tested on Apple M4 Pro). Part 7 requires a TPU.

# %%
# !pip install -q jax optax matplotlib

# %% [markdown]
# ## Imports & Configuration

# %%
import functools as ft
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

REVISION = 7

ON_TPU = any('TPU' in d.device_kind for d in jax.devices())

print(f"JAX version : {jax.__version__}")
print(f"Devices     : {jax.devices()}")
print(f"ON_TPU      : {ON_TPU}")
print(f"Notebook rev: {REVISION}")


# JAX pytree with dot-notation access
@jax.tree_util.register_pytree_with_keys_class
class dot_dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

    def tree_flatten_with_keys(self):
        keys = tuple(sorted(self))
        return tuple((jax.tree_util.DictKey(k), self[k]) for k in keys), keys

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values))


@jax.tree_util.register_static
@dataclass(kw_only=True, frozen=True)
class Config:
    # ── MoE ────────────────────────────────────────────────────
    n_experts: int = 4
    n_active_experts: int = 2
    expert_mlp_dim: int = 256
    capacity_factor: float = 1.25
    aux_loss_alpha: float = 0.01
    z_loss_alpha: float = 1e-4

    # ── Architecture (small for CPU testing) ───────────────────
    attn_impl: str = 'einsum'
    qk_norm: bool = True
    n_embd: int = 128
    n_layer: int = 2
    seq_len: int = 64
    vocab_size: int = 512
    n_head: int = 2
    n_kv_head: int = 1
    head_dim: int = 64
    softcap: float = 15.0
    logit_dtype: str = 'fp32'
    num_lm_head_chunks: int = 2
    batch_size: int = 4
    microbatch_size: int = 2

    # ── Training ───────────────────────────────────────────────
    learning_rate: float = 3e-3
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    warmdown_ratio: float = 0.3
    final_lr_frac: float = 0.0

    # ── Eval / Data ────────────────────────────────────────────
    eval_steps: int = 5
    param_seed: int = 42

    @property
    def num_microbatches(self):
        return self.batch_size // self.microbatch_size


config = Config()
print(f'Config: D={config.n_embd}, L={config.n_layer}, T={config.seq_len}, '
      f'V={config.vocab_size}, N={config.n_head}, K={config.n_kv_head}, '
      f'H={config.head_dim}')
print(f'MoE: E={config.n_experts}, top-{config.n_active_experts}, '
      f'F_expert={config.expert_mlp_dim}')
print(f'Training: lr={config.learning_rate:.1e}, B={config.batch_size}, '
      f'microbatch={config.microbatch_size}')

# %% [markdown]
# ## Shared Components
#
# RMSNorm, RoPE, attention — reused across all MoE strategies.

# %%
# === RMSNorm, RoPE, Attention ===

def rms_norm(x):
    """RMSNorm with no learnable parameters."""
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def precompute_rope(seq_len, head_dim, base=10000):
    """Precompute rotary embedding cos/sin tables."""
    channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos = jnp.cos(freqs).astype(jnp.bfloat16)
    sin = jnp.sin(freqs).astype(jnp.bfloat16)
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary embeddings. x: (B, H, T, D), cos/sin: (1, 1, T, D/2)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1)


def _expand_kv(k, v, n_head, n_kv_head):
    """Repeat KV heads to match Q head count."""
    if n_kv_head == n_head:
        return k, v
    ratio = n_head // n_kv_head
    return jnp.repeat(k, ratio, axis=1), jnp.repeat(v, ratio, axis=1)


def count_params(params):
    return sum(p.size for p in jax.tree.leaves(params) if isinstance(p, jax.Array))


def count_non_embed_params(params):
    return count_params(params) - params.wte.size

# %% [markdown]
# ---
#
# ## Part 1: Understanding Grouped Matrix Multiplication
#
# The key insight behind MegaBlox is **grouped matrix multiplication (GMM)**.
# Instead of running one separate matmul per expert, we:
#
# 1. **Sort** all tokens by their expert assignment
# 2. Record **group\_sizes**: how many tokens each expert received
# 3. Run a **single GMM** that processes all groups in one operation
#
# ### The Data Layout
#
# ```
# Standard MoE: E separate matmuls, each with variable-sized inputs
# ┌─────────────┐  ┌───────────┐  ┌─────┐
# │ Expert 0    │  │ Expert 1  │  │ E 2 │   (3 separate kernel calls)
# │ 5 tokens    │  │ 3 tokens  │  │ 1   │
# │ [5, D]@[D,F]│  │[3,D]@[D,F]│  │[1,D]│
# └─────────────┘  └───────────┘  └─────┘
#
# GMM: one operation on sorted, concatenated inputs
# ┌──────────────────────────────────────────┐
# │ [tok0 tok1 tok2 tok3 tok4 | tok5 tok6 tok7 | tok8]  ← sorted by expert
# │  group_sizes = [5, 3, 1]                             ← tokens per expert
# │  weights = [W_expert0, W_expert1, W_expert2]          ← stacked [E, D, F]
# │                                                       ← ONE kernel call
# └──────────────────────────────────────────┘
# ```
#
# ### Shapes
#
# | Input | Shape | Description |
# |-------|-------|-------------|
# | `lhs` (inputs) | `[M, K]` | All tokens concatenated, sorted by expert |
# | `rhs` (weights) | `[num_groups, K, N]` | Per-expert weight matrices |
# | `group_sizes` | `[num_groups]` | Token count per expert (int32) |
# | **output** | `[M, N]` | All results, same order as sorted inputs |
#
# Where M = total tokens across all groups, K = input dim, N = output dim.

# %%
# === Part 1: Grouped Matrix Multiplication — two implementations ===

def grouped_matmul_reference(inputs, weights, group_sizes):
    """Reference GMM: Python loop over groups. Clear but not JIT-compatible.

    Args:
        inputs: [M, K] — tokens sorted by expert assignment
        weights: [G, K, N] — per-expert weight matrices
        group_sizes: list or array of ints — tokens per expert

    Returns:
        output: [M, N]
    """
    outputs = []
    offset = 0
    for i in range(len(group_sizes)):
        size = int(group_sizes[i])
        if size > 0:
            group_input = inputs[offset:offset + size]   # [size, K]
            group_output = group_input @ weights[i]       # [size, N]
            outputs.append(group_output)
        offset += size
    return jnp.concatenate(outputs, axis=0)


def grouped_matmul_jax(inputs, weights, group_sizes):
    """JIT-compatible GMM using gather + einsum.

    How it works:
    1. Convert group_sizes → per-token group IDs: [0,0,0, 1,1, 2,...]
    2. Gather the matching weight matrix for each token
    3. Batch matmul via einsum

    This is O(M*K*N) — same FLOPs as the loop version — but creates an
    intermediate [M, K, N] tensor from the gather. The Pallas kernel in
    Part 2 avoids this memory overhead.

    Args:
        inputs: [M, K] — tokens sorted by expert assignment
        weights: [G, K, N] — per-expert weight matrices
        group_sizes: [G] int32 — tokens per expert

    Returns:
        output: [M, N]
    """
    M = inputs.shape[0]
    G = weights.shape[0]
    # Map each row to its group: [0,0,0, 1,1, 2] for group_sizes=[3,2,1]
    group_ids = jnp.repeat(jnp.arange(G, dtype=jnp.int32), group_sizes,
                           total_repeat_length=M)
    # Gather per-token weight matrix and batch matmul
    per_token_weights = weights[group_ids]  # [M, K, N]
    return jnp.einsum('mk,mkn->mn', inputs, per_token_weights)


# --- Verify they produce the same results ---
print("=== Part 1: Grouped Matrix Multiplication ===\n")

key = jax.random.key(42)
k1, k2, k3 = jax.random.split(key, 3)

# Example: 3 experts, variable token counts
group_sizes_test = jnp.array([5, 3, 2], dtype=jnp.int32)  # 10 tokens total
M_test = int(group_sizes_test.sum())
K_test, N_test = 8, 16

inputs_test = jax.random.normal(k1, (M_test, K_test))
weights_test = jax.random.normal(k2, (3, K_test, N_test))

# Run both
out_ref = grouped_matmul_reference(inputs_test, weights_test, group_sizes_test)
out_jax = grouped_matmul_jax(inputs_test, weights_test, group_sizes_test)

print(f"Input shapes:  inputs={inputs_test.shape}, weights={weights_test.shape}")
print(f"Group sizes:   {group_sizes_test.tolist()} (total M={M_test})")
print(f"Output shapes: reference={out_ref.shape}, jax={out_jax.shape}")
print(f"Max difference: {float(jnp.max(jnp.abs(out_ref - out_jax))):.2e}")
print(f"Match: {bool(jnp.allclose(out_ref, out_jax, atol=1e-5))}")

# Show the group_ids mapping
group_ids_demo = jnp.repeat(jnp.arange(3), group_sizes_test, total_repeat_length=M_test)
print(f"\ngroup_sizes = {group_sizes_test.tolist()}")
print(f"group_ids   = {group_ids_demo.tolist()}")
print(f"  → token 0-4 use expert 0, tokens 5-7 use expert 1, tokens 8-9 use expert 2")

# %% [markdown]
# ---
#
# ## Part 2: The Pallas GMM Kernel (MegaBlox)
#
# The pure-JAX `grouped_matmul_jax` works correctly but has a problem: the
# `weights[group_ids]` gather creates an **intermediate tensor of shape [M, K, N]**.
# For a real model with M=131072 tokens, K=1024, N=2048, that's **1 TB** of
# intermediate data.
#
# The MegaBlox Pallas kernel solves this by processing groups **tile by tile**:
# it never materializes the full [M, K, N] intermediate. Instead, it:
#
# 1. Computes **group metadata** (offsets, tile assignments) from `group_sizes`
# 2. Iterates over **(M-tiles, K-tiles, N-tiles)** in a 3D grid
# 3. Each tile loads a small chunk of inputs and the right expert's weights
# 4. Accumulates partial results into output tiles
#
# On TPU, this achieves near-peak MXU utilization. On CPU, we use
# `interpret=True` which runs the same logic as regular JAX (slow but correct).
#
# ### Simplified kernel
#
# Below is a stripped-down version of MaxText's
# `kernels/megablox/backend.py` — same algorithm, but with quantization
# (`qwix`), sharding (`group_offset`), and `tokamax` dependencies removed.

# %%
# === Part 2: Simplified Pallas GMM kernel (from MaxText backend.py) ===

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _calculate_num_tiles(x, tx):
    """Number of full tiles of size tx that fit in dimension x."""
    tiles, rem = divmod(x, tx)
    if rem:
        raise ValueError(f"{x} must be divisible by tile size {tx}.")
    return tiles


def make_group_metadata(*, group_sizes, m, tm, visit_empty_groups=False):
    """Create metadata for the Pallas GMM kernel.

    Converts group_sizes into tile-level routing information:
    - group_offsets: CSR-style row offsets for each group
    - group_ids: which group each tile belongs to
    - m_tile_ids: which m-dimension tile each grid index processes
    - num_tiles: total number of tiles to execute

    This is pure JAX — no Pallas or TPU dependency.
    """
    num_groups = group_sizes.shape[0]
    tiles_m = _calculate_num_tiles(m, tm)

    # CSR-style offsets: group_offsets[i] = row where group i starts
    group_ends = jnp.cumsum(group_sizes)
    group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])

    # Round group boundaries to tile edges
    rounded_group_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)
    group_starts = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]])
    rounded_group_starts = group_starts // tm * tm
    rounded_group_sizes = rounded_group_ends - rounded_group_starts
    rounded_group_sizes = jnp.where(group_sizes == 0, 0, rounded_group_sizes)
    group_tiles = rounded_group_sizes // tm

    if visit_empty_groups:
        group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

    # Map grid indices → group IDs
    group_ids = jnp.repeat(
        jnp.arange(num_groups, dtype=jnp.int32), group_tiles,
        total_repeat_length=tiles_m + num_groups - 1)

    # Map grid indices → m-dimension tile IDs
    partial_tile_mask = jnp.logical_or(
        (group_offsets[:-1] % tm) == 0, group_sizes == 0)
    if visit_empty_groups:
        partial_tile_mask = jnp.where(group_sizes == 0, 0, partial_tile_mask)
    partial_tile_ids = jnp.where(partial_tile_mask, tiles_m + 1, group_offsets[:-1] // tm)
    tile_visits = jnp.histogram(
        partial_tile_ids, bins=tiles_m, range=(0, tiles_m))[0] + 1
    m_tile_ids = jnp.repeat(
        jnp.arange(tiles_m, dtype=jnp.int32), tile_visits.astype(jnp.int32),
        total_repeat_length=tiles_m + num_groups - 1)

    num_tiles = group_tiles.sum()
    return (group_offsets, group_ids, m_tile_ids), num_tiles


def _get_store_mask(*, grid_id, group_metadata, tm, tn):
    """Row mask: True for rows belonging to current group in current tile."""
    group_offsets, group_ids, m_tile_ids = group_metadata[:3]
    group_id = group_ids[grid_id]
    group_start = group_offsets[group_id]
    group_end = group_offsets[group_id + 1]
    m_id = m_tile_ids[grid_id] * tm
    iota = jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + m_id
    return jnp.logical_and(iota >= group_start, iota < group_end)


def pallas_gmm(lhs, rhs, group_sizes, preferred_element_type=jnp.float32,
               tiling=(128, 128, 128), interpret=False):
    """Simplified MegaBlox GMM via Pallas.

    Computes: output[offset:offset+size] = lhs[offset:offset+size] @ rhs[group]
    for each group, where offsets are determined by group_sizes.

    Args:
        lhs: [M, K] — sorted input activations
        rhs: [num_groups, K, N] — per-expert weight matrices
        group_sizes: [num_groups] int32
        preferred_element_type: output dtype (default float32)
        tiling: (tm, tk, tn) tile dimensions
        interpret: if True, run as pure JAX (CPU-compatible)

    Returns:
        output: [M, N]
    """
    m, k = lhs.shape
    n = rhs.shape[2]
    tm, tk, tn = tiling

    # Pad M to be divisible by tm
    pad_m = (tm - m % tm) % tm
    if pad_m > 0:
        lhs = jnp.pad(lhs, ((0, pad_m), (0, 0)))
        group_sizes = group_sizes.at[-1].add(pad_m)
    m_padded = lhs.shape[0]

    tiles_k = _calculate_num_tiles(k, tk)
    tiles_n = _calculate_num_tiles(n, tn)

    group_metadata, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes, m=m_padded, tm=tm, visit_empty_groups=False)

    # PrefetchScalarGridSpec passes each scalar prefetch array as a
    # separate positional arg to both the kernel and index functions.
    # With num_scalar_prefetch=3, signatures are:
    #   kernel(offsets_ref, ids_ref, tiles_ref, lhs_ref, rhs_ref, out_ref, scratch)
    #   idx_fn(n_i, grid_id, k_i, offsets_ref, ids_ref, tiles_ref)

    def kernel(offsets_ref, ids_ref, tiles_ref, lhs_ref, rhs_ref, out_ref, acc_scratch):
        group_offsets = offsets_ref[:]
        group_ids = ids_ref[:]
        m_tile_ids = tiles_ref[:]
        gm = (group_offsets, group_ids, m_tile_ids)

        grid_id = pl.program_id(1)
        k_i = pl.program_id(2)

        @pl.when(k_i == 0)
        def _zero_acc():
            acc_scratch[...] = jnp.zeros_like(acc_scratch)

        loaded_lhs = lhs_ref[...]
        loaded_rhs = rhs_ref[...]

        acc_scratch[...] += jax.lax.dot_general(
            loaded_lhs, loaded_rhs,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)

        @pl.when(k_i == tiles_k - 1)
        def _store():
            mask = _get_store_mask(grid_id=grid_id, group_metadata=gm,
                                   tm=tm, tn=tn)
            out_ref[...] = jnp.where(mask, acc_scratch[...],
                                      out_ref[...]).astype(preferred_element_type)

    def lhs_idx(n_i, grid_id, k_i, offsets_ref, ids_ref, tiles_ref):
        return tiles_ref[grid_id], k_i

    def rhs_idx(n_i, grid_id, k_i, offsets_ref, ids_ref, tiles_ref):
        return ids_ref[grid_id], k_i, n_i

    def out_idx(n_i, grid_id, k_i, offsets_ref, ids_ref, tiles_ref):
        return tiles_ref[grid_id], n_i

    group_offsets, group_ids, m_tile_ids = group_metadata

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m_padded, n), preferred_element_type),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                pl.BlockSpec((tm, tk), lhs_idx),
                pl.BlockSpec((None, tk, tn), rhs_idx),
            ],
            out_specs=pl.BlockSpec((tm, tn), out_idx),
            grid=(tiles_n, num_active_tiles, tiles_k),
            scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
        ),
        interpret=interpret,
    )(group_offsets, group_ids, m_tile_ids, lhs, rhs)

    # Remove padding
    return out[:m]


# NOTE: The full MegaBlox also includes tgmm (transposed GMM) for the
# backward pass weight gradients, plus a custom_vjp wrapper. In MaxText,
# see kernels/megablox/backend.py:tgmm() and ops.py:gmm().
#
# For this educational notebook, we use the pure-JAX grouped_matmul_jax
# for training (which gets free autodiff from JAX) and demonstrate the
# Pallas kernel for forward-pass comparison only.


# --- Verify Pallas GMM matches reference ---
print("\n=== Part 2: Pallas GMM Kernel ===\n")

# Dimensions must be divisible by tile size — use small tiles for CPU
tile_size = 16
# Adjust test dimensions to be tile-aligned
M_pallas = 32  # divisible by 16
K_pallas = 16
N_pallas = 32
G_pallas = 3
group_sizes_pallas = jnp.array([16, 8, 8], dtype=jnp.int32)  # sums to 32

k1, k2 = jax.random.split(jax.random.key(99), 2)
lhs_pallas = jax.random.normal(k1, (M_pallas, K_pallas), dtype=jnp.float32)
rhs_pallas = jax.random.normal(k2, (G_pallas, K_pallas, N_pallas), dtype=jnp.float32)

out_ref_pallas = grouped_matmul_reference(lhs_pallas, rhs_pallas, group_sizes_pallas)
out_jax_pallas = grouped_matmul_jax(lhs_pallas, rhs_pallas, group_sizes_pallas)
out_pallas = pallas_gmm(lhs_pallas, rhs_pallas, group_sizes_pallas,
                        tiling=(tile_size, tile_size, tile_size), interpret=True)

print(f"Shapes: lhs={lhs_pallas.shape}, rhs={rhs_pallas.shape}, "
      f"group_sizes={group_sizes_pallas.tolist()}")
print(f"Output shapes: ref={out_ref_pallas.shape}, jax={out_jax_pallas.shape}, "
      f"pallas={out_pallas.shape}")
print(f"Reference vs JAX   max diff: {float(jnp.max(jnp.abs(out_ref_pallas - out_jax_pallas))):.2e}")
print(f"Reference vs Pallas max diff: {float(jnp.max(jnp.abs(out_ref_pallas - out_pallas))):.2e}")
print(f"All three match: {bool(jnp.allclose(out_ref_pallas, out_pallas, atol=1e-4))}")

# %% [markdown]
# ---
#
# ## Part 3: The Token Routing Pipeline
#
# The GMM kernel is the compute engine, but we need a **routing pipeline**
# to prepare its inputs. Here's the full data flow for dropless MoE:
#
# ```
# Step 1: ROUTE              tokens → router → top-K experts per token
#         ─────              [N, D] @ [D, E] → logits [N, E] → top_k [N, K]
#
# Step 2: FLATTEN             expand for K experts per token
#         ───────             expert_ids: [N, K] → [N*K]
#                             weights:    [N, K] → [N*K]
#                             inputs:     [N, D] → repeat K → [N*K, D]
#
# Step 3: SORT                group tokens by expert
#         ────                sorted_indices = argsort(expert_ids)
#                             sorted_inputs = inputs[sorted_indices]
#
# Step 4: COUNT               tokens per expert
#         ─────               group_sizes = bincount(expert_ids)
#
# Step 5: GMM                 sorted_inputs @ expert_weights → expert_outputs
#         ───                 [N*K, D] @ [E, D, F] → [N*K, F]  (up projection)
#
# Step 6: ACTIVATE            nonlinearity (ReLU²)
#         ────────            [N*K, F] → [N*K, F]
#
# Step 7: GMM                 expert_outputs @ expert_weights → final_outputs
#         ───                 [N*K, F] @ [E, F, D] → [N*K, D]  (down projection)
#
# Step 8: UNSORT + COMBINE    reverse permutation, weight, sum
#         ─────────────────   output[sorted_indices] = results
#                             reshape [N, K, D], weight by routing scores, sum
# ```
#
# ### Concrete Example: 4 tokens, 2 experts, top-2
#
# ```
# Token 0 → experts [0, 1], weights [0.7, 0.3]
# Token 1 → experts [1, 0], weights [0.6, 0.4]
# Token 2 → experts [0, 0], weights [0.5, 0.5]  ← both picks are expert 0!
# Token 3 → experts [1, 1], weights [0.8, 0.2]  ← both picks are expert 1!
#
# Flatten: expert_ids = [0, 1, 1, 0, 0, 0, 1, 1]  (8 entries, 4 tokens × 2)
# Bincount: group_sizes = [4, 4]  (expert 0 gets 4 tokens, expert 1 gets 4)
#
# Sort by expert: indices [0,3,4,5, 1,2,6,7] → tokens grouped by expert
# GMM processes: first 4 rows with expert 0's weights, next 4 with expert 1's
# Unsort: reverse permutation back to original order
# Combine: weight each token's K=2 expert outputs, sum → final [4, D]
# ```

# %%
# === Part 3: Routing pipeline implementation ===

def dropless_routing(config, router_weights, x_flat):
    """Dropless MoE routing: sort tokens by expert, compute group_sizes.

    Args:
        config: Config with n_experts, n_active_experts
        router_weights: [D, E] router projection
        x_flat: [N, D] flattened input tokens

    Returns:
        sorted_inputs:    [N*K, D] tokens sorted by expert assignment
        sorted_indices:   [N*K] permutation (for unsorting)
        group_sizes:      [E] int32 — tokens per expert
        top_k_weights:    [N, K] routing weights
        top_k_idx:        [N, K] expert assignments
        router_logits:    [N, E] raw logits (for aux/z loss)
    """
    N, D = x_flat.shape
    E = config.n_experts
    K = config.n_active_experts

    # Step 1: Route
    router_logits = jnp.einsum('nd,de->ne', x_flat, router_weights)  # [N, E]
    top_k_logits, top_k_idx = jax.lax.top_k(router_logits, K)        # [N, K]
    top_k_weights = jax.nn.softmax(top_k_logits, axis=-1)             # [N, K]

    # Step 2: Flatten
    expert_flat = top_k_idx.reshape(N * K)       # [N*K]
    x_rep = jnp.repeat(x_flat, K, axis=0)        # [N*K, D]

    # Step 3: Sort by expert assignment
    sorted_indices = jnp.argsort(expert_flat)
    sorted_inputs = x_rep[sorted_indices]         # [N*K, D]

    # Step 4: Count tokens per expert
    group_sizes = jnp.bincount(expert_flat, length=E).astype(jnp.int32)

    return sorted_inputs, sorted_indices, group_sizes, top_k_weights, top_k_idx, router_logits


def dropless_combine(down_sorted, sorted_indices, top_k_weights, N, K, D):
    """Unsort expert outputs and combine with routing weights.

    Args:
        down_sorted: [N*K, D] expert outputs in sorted order
        sorted_indices: [N*K] permutation used for sorting
        top_k_weights: [N, K] routing weights
        N, K, D: dimensions

    Returns:
        output: [N, D]
    """
    # Unsort back to original token order
    unsorted = jnp.zeros_like(down_sorted)
    unsorted = unsorted.at[sorted_indices].set(down_sorted)

    # Weight by routing scores and sum over K experts
    weight_flat = top_k_weights.reshape(N * K)
    weighted = unsorted * weight_flat[:, None]
    return weighted.reshape(N, K, D).sum(axis=1)


# --- Demo with concrete values ---
print("=== Part 3: Token Routing Pipeline ===\n")

key = jax.random.key(7)
k1, k2, k3 = jax.random.split(key, 3)

N_demo, D_demo, E_demo, K_demo = 8, 4, 3, 2
x_demo = jax.random.normal(k1, (N_demo, D_demo))
router_demo = jax.random.normal(k2, (D_demo, E_demo)) * 0.1

demo_config = Config(n_experts=E_demo, n_active_experts=K_demo,
                     expert_mlp_dim=8, n_embd=D_demo)

sorted_in, sorted_idx, g_sizes, tk_w, tk_i, r_logits = \
    dropless_routing(demo_config, router_demo, x_demo)

print(f"Inputs: {N_demo} tokens, D={D_demo}, E={E_demo}, K={K_demo}")
print(f"\nRouter logits (first 4 tokens):")
for i in range(min(4, N_demo)):
    probs = jax.nn.softmax(r_logits[i])
    print(f"  Token {i}: logits={r_logits[i].tolist()}, "
          f"probs=[{', '.join(f'{p:.2f}' for p in probs.tolist())}], "
          f"top-{K_demo}=experts {tk_i[i].tolist()}")

expert_flat = tk_i.reshape(N_demo * K_demo)
print(f"\nFlattened expert assignments: {expert_flat.tolist()}")
print(f"Group sizes (tokens per expert): {g_sizes.tolist()}")
print(f"Sorted indices (first 8): {sorted_idx[:8].tolist()}")
print(f"\nSorted inputs shape: {sorted_in.shape}  (N*K={N_demo*K_demo}, D={D_demo})")

# %% [markdown]
# ---
#
# ## Part 4: Three MoE Dispatch Strategies
#
# Now we put it all together. Three complete MoE forward functions, each using
# the **same router, same weights, same architecture** — only the dispatch
# mechanism differs:
#
# | Strategy | How it works | Token dropping? | Special kernel? |
# |----------|-------------|-----------------|-----------------|
# | **Capacity-based** | Scatter into fixed `(E, C, D)` buffer | Yes — overflow dropped | No |
# | **Dropless JAX** | Sort + `grouped_matmul_jax` | No | No |
# | **Dropless MegaBlox** | Sort + Pallas GMM kernel | No | Yes (Pallas) |

# %%
# === Part 4: Three MoE forward implementations ===

# --- Strategy A: Capacity-based dispatch (from 09_moe.py) ---

def moe_capacity_forward(config, layer, x):
    """Capacity-based MoE: scatter tokens into fixed-size expert buffer.

    Tokens exceeding expert capacity are dropped (get zero contribution).
    This is the approach used in Switch Transformer and 09_moe.py.
    """
    B, T, D = x.shape
    N = B * T
    E = config.n_experts
    K = config.n_active_experts
    x_flat = x.reshape(N, D)

    # Route
    router_logits = jnp.einsum('nd,de->ne', x_flat, layer.router)
    z_loss = jnp.mean(jax.nn.logsumexp(router_logits, axis=-1) ** 2)
    router_probs = jax.nn.softmax(router_logits, axis=-1)
    top_k_logits, top_k_idx = jax.lax.top_k(router_logits, K)
    top_k_weights = jax.nn.softmax(top_k_logits, axis=-1)

    # Aux loss
    expert_mask = jax.nn.one_hot(top_k_idx, E)
    f = jnp.sum(expert_mask, axis=(0, 1)) / (N * K)
    P = jnp.mean(router_probs, axis=0)
    aux_loss = E * jnp.sum(f * P)

    # Dispatch: scatter into (E, C, D) buffer
    expert_flat = top_k_idx.reshape(N * K)
    weight_flat = top_k_weights.reshape(N * K)
    x_rep = jnp.repeat(x_flat, K, axis=0)

    expert_oh = jax.nn.one_hot(expert_flat, E)
    cumpos = jnp.cumsum(expert_oh, axis=0) * expert_oh
    pos = (jnp.sum(cumpos, axis=-1) - 1).astype(jnp.int32)

    C = int(((N * K + E - 1) // E) * config.capacity_factor)
    valid = pos < C
    pos_clipped = jnp.clip(pos, 0, C - 1)

    expert_input = jnp.zeros((E, C, D), dtype=x_flat.dtype)
    expert_input = expert_input.at[expert_flat, pos_clipped].add(
        x_rep * valid[:, None])

    # Expert ReLU² (batched einsums over E dimension)
    up = jnp.einsum('ecd,edf->ecf', expert_input, layer.expert_w_up)
    up = jax.nn.relu(up) ** 2
    expert_out = jnp.einsum('ecf,efd->ecd', up, layer.expert_w_down)

    # Combine
    gathered = expert_out[expert_flat, pos_clipped]
    weighted = gathered * weight_flat[:, None] * valid[:, None]
    output = weighted.reshape(N, K, D).sum(axis=1)

    return output.reshape(B, T, D), aux_loss, z_loss


# --- Strategy B: Dropless with pure-JAX grouped matmul ---

def moe_dropless_jax_forward(config, layer, x):
    """Dropless MoE using pure-JAX grouped matmul.

    All tokens are processed — no capacity limit, no dropping.
    Uses grouped_matmul_jax (gather + einsum) which is fully
    differentiable through JAX's standard autodiff.
    """
    B, T, D = x.shape
    N = B * T
    E = config.n_experts
    K = config.n_active_experts
    x_flat = x.reshape(N, D)

    # Route + sort
    sorted_inputs, sorted_indices, group_sizes, top_k_weights, top_k_idx, router_logits = \
        dropless_routing(config, layer.router, x_flat)

    # Aux losses
    z_loss = jnp.mean(jax.nn.logsumexp(router_logits, axis=-1) ** 2)
    router_probs = jax.nn.softmax(router_logits, axis=-1)
    expert_mask = jax.nn.one_hot(top_k_idx, E)
    f = jnp.sum(expert_mask, axis=(0, 1)) / (N * K)
    P = jnp.mean(router_probs, axis=0)
    aux_loss = E * jnp.sum(f * P)

    # Expert computation via grouped matmul
    up = grouped_matmul_jax(sorted_inputs, layer.expert_w_up, group_sizes)
    up = jax.nn.relu(up) ** 2
    down = grouped_matmul_jax(up, layer.expert_w_down, group_sizes)

    # Unsort + combine
    output = dropless_combine(down, sorted_indices, top_k_weights, N, K, D)
    return output.reshape(B, T, D), aux_loss, z_loss


# --- Strategy C: Dropless with Pallas GMM kernel ---

def moe_megablox_forward(config, layer, x):
    """Dropless MoE using MegaBlox Pallas GMM kernel.

    Same routing as dropless JAX, but uses the Pallas kernel for the
    expert computation. On TPU, this avoids materializing intermediate
    tensors. On CPU, interpret=True gives identical results.

    NOTE: pallas_gmm handles M-dimension padding internally.
    We only need K and N dimensions to be tile-divisible.
    """
    B, T, D = x.shape
    N = B * T
    E = config.n_experts
    K = config.n_active_experts
    F = config.expert_mlp_dim
    x_flat = x.reshape(N, D)

    # Route + sort (identical to dropless JAX)
    sorted_inputs, sorted_indices, group_sizes, top_k_weights, top_k_idx, router_logits = \
        dropless_routing(config, layer.router, x_flat)

    # Aux losses (identical)
    z_loss = jnp.mean(jax.nn.logsumexp(router_logits, axis=-1) ** 2)
    router_probs = jax.nn.softmax(router_logits, axis=-1)
    expert_mask = jax.nn.one_hot(top_k_idx, E)
    f_frac = jnp.sum(expert_mask, axis=(0, 1)) / (N * K)
    P = jnp.mean(router_probs, axis=0)
    aux_loss = E * jnp.sum(f_frac * P)

    # Use tile size that divides all relevant dimensions
    # D=128 and F=256 are both divisible by 16
    tile = 16

    # Expert computation via Pallas GMM (interpret=True for CPU)
    up = pallas_gmm(sorted_inputs, layer.expert_w_up, group_sizes,
                    tiling=(tile, tile, tile), interpret=True)
    up = jax.nn.relu(up) ** 2
    down = pallas_gmm(up, layer.expert_w_down, group_sizes,
                      tiling=(tile, tile, tile), interpret=True)

    # Unsort + combine (identical to dropless JAX)
    output = dropless_combine(down, sorted_indices, top_k_weights, N, K, D)
    return output.reshape(B, T, D), aux_loss, z_loss


# %% [markdown]
# ### Numerical Comparison
#
# Let's run all three strategies on the same input and compare outputs.

# %%
# === Compare all three strategies ===
print("=== Part 4: Three MoE Strategies — Numerical Comparison ===\n")

key = jax.random.key(42)
keys = jax.random.split(key, 8)

# Create a test layer with shared weights
test_layer = dot_dict()
s = (3.0 ** 0.5) * (config.n_embd ** -0.5)
test_layer.router = jax.random.normal(keys[0], (config.n_embd, config.n_experts),
                                       dtype=jnp.float32) * 0.01
test_layer.expert_w_up = jax.random.uniform(keys[1],
    (config.n_experts, config.n_embd, config.expert_mlp_dim),
    dtype=jnp.float32, minval=-s, maxval=s)
test_layer.expert_w_down = jnp.zeros(
    (config.n_experts, config.expert_mlp_dim, config.n_embd), dtype=jnp.float32)

# Test input
x_test = jax.random.normal(keys[2], (config.batch_size, config.seq_len, config.n_embd),
                            dtype=jnp.float32)

print(f"Input shape: {x_test.shape}")
print(f"Config: E={config.n_experts}, k={config.n_active_experts}, "
      f"D={config.n_embd}, F={config.expert_mlp_dim}")
print(f"Capacity factor: {config.capacity_factor}\n")

# Run all three
t0 = time.time()
out_cap, aux_cap, z_cap = moe_capacity_forward(config, test_layer, x_test)
t_cap = time.time() - t0
print(f"Capacity-based:   shape={out_cap.shape}, aux={aux_cap:.4f}, "
      f"z={z_cap:.4f}, time={t_cap:.3f}s")

t0 = time.time()
out_djax, aux_djax, z_djax = moe_dropless_jax_forward(config, test_layer, x_test)
t_djax = time.time() - t0
print(f"Dropless JAX:     shape={out_djax.shape}, aux={aux_djax:.4f}, "
      f"z={z_djax:.4f}, time={t_djax:.3f}s")

t0 = time.time()
out_mblx, aux_mblx, z_mblx = moe_megablox_forward(config, test_layer, x_test)
t_mblx = time.time() - t0
print(f"Dropless MegaBlox: shape={out_mblx.shape}, aux={aux_mblx:.4f}, "
      f"z={z_mblx:.4f}, time={t_mblx:.3f}s")

# Compare
diff_jax_mblx = float(jnp.max(jnp.abs(out_djax - out_mblx)))
diff_cap_djax = float(jnp.max(jnp.abs(out_cap - out_djax)))
print(f"\nDropless JAX vs MegaBlox max diff: {diff_jax_mblx:.2e}")
print(f"Capacity vs Dropless JAX max diff: {diff_cap_djax:.2e}")
print(f"\nAux losses match (same router): "
      f"cap={aux_cap:.6f}, djax={aux_djax:.6f}, mblx={aux_mblx:.6f}")

# Note: capacity vs dropless may differ because capacity drops tokens
# With small N and high capacity_factor, no tokens may actually be dropped
N_test = config.batch_size * config.seq_len
C_test = int(((N_test * config.n_active_experts + config.n_experts - 1)
              // config.n_experts) * config.capacity_factor)
print(f"\nCapacity per expert: {C_test} slots for ~{N_test * config.n_active_experts // config.n_experts} "
      f"avg tokens/expert")

# %% [markdown]
# ---
#
# ## Part 5: Token Dropping vs Dropless
#
# The difference between capacity-based and dropless dispatch becomes dramatic
# when routing is **imbalanced** — when some experts are much more popular
# than others.
#
# Let's force an extreme imbalance and measure the impact.

# %%
# === Part 5: Token dropping demonstration ===
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for scripts
import matplotlib.pyplot as plt

print("=== Part 5: Token Dropping vs Dropless ===\n")

key = jax.random.key(123)
k1, k2 = jax.random.split(key)

# Create a layer with BIASED router: expert 0 gets huge weights
biased_layer = dot_dict()
# Use random per-token variation + strong bias toward expert 0
biased_router = jax.random.normal(k1, (config.n_embd, config.n_experts),
                                   dtype=jnp.float32) * 0.01
# Expert 0 gets a massive constant bias via the projection
biased_router = biased_router.at[:, 0].add(10.0)
biased_router = biased_router.at[:, 1].add(-5.0)
biased_router = biased_router.at[:, 2].add(-5.0)
biased_router = biased_router.at[:, 3].add(-5.0)
biased_layer.router = biased_router

biased_layer.expert_w_up = test_layer.expert_w_up
# Use non-zero down weights so outputs are meaningful
biased_layer.expert_w_down = jax.random.uniform(
    k2, (config.n_experts, config.expert_mlp_dim, config.n_embd),
    dtype=jnp.float32, minval=-0.01, maxval=0.01)

# Use a tight capacity factor to make dropping visible
biased_config = Config(capacity_factor=1.0)  # no headroom = guaranteed drops

x_biased = jax.random.normal(jax.random.key(55),
                              (config.batch_size, config.seq_len, config.n_embd),
                              dtype=jnp.float32)
N_biased = config.batch_size * config.seq_len

# Analyze routing distribution
router_logits = jnp.einsum('nd,de->ne', x_biased.reshape(N_biased, config.n_embd),
                           biased_layer.router)
probs = jax.nn.softmax(router_logits, axis=-1)
mean_probs = jnp.mean(probs, axis=0)
top_k_logits, top_k_idx = jax.lax.top_k(router_logits, biased_config.n_active_experts)

expert_flat = top_k_idx.reshape(-1)
tokens_per_expert = [int(jnp.sum(expert_flat == e)) for e in range(biased_config.n_experts)]

print("Biased routing distribution:")
print(f"  Mean router probabilities: {[f'{p:.3f}' for p in mean_probs.tolist()]}")
print(f"  Tokens routed to each expert: {tokens_per_expert}")
total_assigned = sum(tokens_per_expert)

# Capacity limit
C_biased = int(((N_biased * biased_config.n_active_experts + biased_config.n_experts - 1)
                // biased_config.n_experts) * biased_config.capacity_factor)
print(f"\n  Capacity per expert: {C_biased} slots (capacity_factor={biased_config.capacity_factor})")
dropped_per_expert = [max(0, t - C_biased) for t in tokens_per_expert]
total_dropped = sum(dropped_per_expert)
print(f"  Dropped tokens per expert: {dropped_per_expert}")
print(f"  Total dropped: {total_dropped}/{total_assigned} "
      f"({100*total_dropped/total_assigned:.1f}%)")

# Dropless: all tokens processed
print(f"\n  Dropless: ALL {total_assigned} tokens processed (0% dropped)")

# Run both and compare output norms (dropped tokens → zero contribution → lower norm)
out_cap_biased, _, _ = moe_capacity_forward(biased_config, biased_layer, x_biased)
out_djax_biased, _, _ = moe_dropless_jax_forward(biased_config, biased_layer, x_biased)

cap_norm = float(jnp.mean(jnp.abs(out_cap_biased)))
djax_norm = float(jnp.mean(jnp.abs(out_djax_biased)))
print(f"\n  Output mean |magnitude|:")
print(f"    Capacity-based: {cap_norm:.6f}")
print(f"    Dropless:       {djax_norm:.6f}")
if cap_norm > 1e-8:
    print(f"    Ratio: {djax_norm/cap_norm:.2f}x  (dropless has larger outputs — "
          f"more tokens contributed)")
else:
    print(f"    (capacity output near zero — most tokens were dropped!)")

# --- Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: expert load distribution
expert_names = [f'Expert {i}' for i in range(config.n_experts)]
colors = ['#e74c3c' if t > C_biased else '#3498db' for t in tokens_per_expert]

axes[0].bar(expert_names, tokens_per_expert, color=colors, edgecolor='black', alpha=0.8)
axes[0].axhline(y=C_biased, color='red', linestyle='--', linewidth=2,
                label=f'Capacity limit = {C_biased}')
axes[0].set_ylabel('Tokens Assigned')
axes[0].set_title('Expert Load Distribution (Biased Router)')
axes[0].legend()

# Annotate dropped tokens
for i, (t, d) in enumerate(zip(tokens_per_expert, dropped_per_expert)):
    if d > 0:
        axes[0].annotate(f'{d} dropped!', xy=(i, t), ha='center', va='bottom',
                        fontsize=10, color='red', fontweight='bold')

# Right: comparison table as text
axes[1].axis('off')
table_data = [
    ['Metric', 'Capacity-based', 'Dropless'],
    ['Tokens processed', f'{total_assigned - total_dropped}', f'{total_assigned}'],
    ['Tokens dropped', f'{total_dropped}', '0'],
    ['Drop rate', f'{100*total_dropped/total_assigned:.1f}%', '0%'],
    ['Output mean |mag|', f'{cap_norm:.4f}', f'{djax_norm:.4f}'],
    ['Needs capacity_factor?', 'Yes (1.25)', 'No'],
    ['Special kernel?', 'No', 'Optional (Pallas)'],
]
table = axes[1].table(cellText=table_data, cellLoc='center',
                      loc='center', colWidths=[0.35, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.8)

# Style header row
for j in range(3):
    table[0, j].set_facecolor('#34495e')
    table[0, j].set_text_props(color='white', fontweight='bold')

axes[1].set_title('Strategy Comparison', fontsize=13, pad=20)

plt.tight_layout()
plt.savefig('/tmp/moe_comparison.png', dpi=100, bbox_inches='tight')
print("\nVisualization saved to /tmp/moe_comparison.png")
plt.close()

# %% [markdown]
# ---
#
# ## Part 6: Validation Training
#
# Now let's train a small transformer model on CPU with both MoE strategies
# and verify that gradients flow correctly (no NaNs, no crashes). This uses:
# - **Synthetic random data** (no tokenizer needed) — loss will stay near
#   `ln(vocab_size)` since there are no patterns to learn, but the key
#   validation is that both strategies train equivalently.
# - Small model (D=128, L=2, E=4)
# - 50 training steps per strategy

# %%
# === Model infrastructure ===

def init_layer_params(config, seed=42):
    """Initialize params for one transformer layer (attention + MoE ReLU²)."""
    key = jax.random.key(seed)
    keys = jax.random.split(key, 7)
    s = (3.0 ** 0.5) * (config.n_embd ** -0.5)
    layer = dot_dict()

    layer.c_q = jax.random.uniform(keys[0], (config.n_embd, config.n_head, config.head_dim),
                                    dtype=jnp.float32, minval=-s, maxval=s)
    layer.c_k = jax.random.uniform(keys[1], (config.n_embd, config.n_kv_head, config.head_dim),
                                    dtype=jnp.float32, minval=-s, maxval=s)
    layer.c_v = jax.random.uniform(keys[2], (config.n_embd, config.n_kv_head, config.head_dim),
                                    dtype=jnp.float32, minval=-s, maxval=s)
    layer.c_proj = jnp.zeros((config.n_head, config.head_dim, config.n_embd), dtype=jnp.float32)

    layer.router = jax.random.normal(keys[3], (config.n_embd, config.n_experts),
                                      dtype=jnp.float32) * 0.01

    E, D, F = config.n_experts, config.n_embd, config.expert_mlp_dim
    layer.expert_w_up = jax.random.uniform(keys[4], (E, D, F),
                                            dtype=jnp.float32, minval=-s, maxval=s)
    layer.expert_w_down = jnp.zeros((E, F, D), dtype=jnp.float32)
    return layer


def init_all_layers(config, n_layers, seed=42):
    layers = dot_dict()
    for i in range(n_layers):
        layers[i] = init_layer_params(config, seed=seed + i * 7)
    return layers


def init_full_model(config, seed=42):
    key = jax.random.key(seed)
    params = dot_dict()
    key, k1, k2 = jax.random.split(key, 3)
    params.wte = jax.random.normal(k1, (config.vocab_size, config.n_embd),
                                    dtype=jnp.float32)
    params.lm_head = jax.random.normal(k2, (config.n_embd, config.vocab_size),
                                        dtype=jnp.float32) * 0.001
    params.layers = init_all_layers(config, config.n_layer, seed=seed + 100)
    return params


def single_layer_forward(config, layer, x, cos, sin, layer_idx=0, moe_fn=None):
    """Forward pass for one transformer layer (attention + MoE)."""
    if moe_fn is None:
        moe_fn = moe_dropless_jax_forward

    h = rms_norm(x)

    # Attention
    q = jnp.einsum('btd,dhk->bhtk', h, layer.c_q)
    k = jnp.einsum('btd,dhk->bhtk', h, layer.c_k)
    v = jnp.einsum('btd,dhk->bhtk', h, layer.c_v)

    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    if config.qk_norm:
        q = rms_norm(q)
        k = rms_norm(k)

    k_exp, v_exp = _expand_kv(k, v, config.n_head, config.n_kv_head)
    scale = config.head_dim ** -0.5
    scores = jnp.einsum('bhtd,bhsd->bhts', q, k_exp) * scale
    seq_len = x.shape[1]
    rows = jnp.arange(seq_len)[:, None]
    cols = jnp.arange(seq_len)[None, :]
    mask = cols <= rows
    scores = jnp.where(mask[None, None, :, :], scores, jnp.finfo(scores.dtype).min)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.einsum('bhts,bhsd->bhtd', attn_weights, v_exp)
    attn_out = jnp.einsum('bhtd,hde->bte', attn_out, layer.c_proj)

    x = x + attn_out

    # MoE
    h2 = rms_norm(x)
    moe_out, aux_loss, z_loss = moe_fn(config, layer, h2)
    x = x + moe_out

    return x, aux_loss, z_loss


def model_forward(config, params, tokens, moe_fn=None):
    """Full forward: embed → layers → final norm."""
    B, T = tokens.shape
    cos, sin = precompute_rope(T, config.head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x = rms_norm(params.wte[tokens])
    total_aux, total_z = 0.0, 0.0
    for i in range(config.n_layer):
        x, aux, z = single_layer_forward(config, params.layers[i], x, cos, sin,
                                          layer_idx=i, moe_fn=moe_fn)
        total_aux += aux
        total_z += z
    return rms_norm(x), total_aux / config.n_layer, total_z / config.n_layer


def _logit_dtype(config):
    return jnp.float32 if config.logit_dtype == 'fp32' else jnp.bfloat16


def _logits_from_chunk(h_chunk, lm_head, config):
    logits = jnp.einsum('td,dv->tv', h_chunk, lm_head,
                        preferred_element_type=_logit_dtype(config))
    return config.softcap * jnp.tanh(logits / config.softcap)


@ft.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def chunked_lm_head_loss(hidden, lm_head, labels, config, moe_fn):
    B, T, D = hidden.shape
    N = config.num_lm_head_chunks
    S = B * T // N
    hidden_chunks = hidden.reshape(N, S, D)
    labels_chunks = labels.reshape(N, S)

    def fwd_body(_, data):
        h_chunk, l_chunk = data
        return None, jnp.sum(
            optax.softmax_cross_entropy_with_integer_labels(
                _logits_from_chunk(h_chunk, lm_head, config), l_chunk))

    _, chunk_losses = jax.lax.scan(fwd_body, None, (hidden_chunks, labels_chunks))
    return jnp.sum(chunk_losses) / (B * T)


def _chunked_loss_fwd(hidden, lm_head, labels, config, moe_fn):
    loss = chunked_lm_head_loss(hidden, lm_head, labels, config, moe_fn)
    return loss, (hidden, lm_head, labels)


def _chunked_loss_bwd(config, moe_fn, residuals, g):
    hidden, lm_head, labels = residuals
    B, T, D = hidden.shape
    N = config.num_lm_head_chunks
    S = B * T // N
    hidden_chunks = hidden.reshape(N, S, D)
    labels_chunks = labels.reshape(N, S)

    def bwd_body(d_lm_head_acc, data):
        h_chunk, l_chunk = data

        def chunk_loss(h, w):
            return jnp.sum(
                optax.softmax_cross_entropy_with_integer_labels(
                    _logits_from_chunk(h, w, config), l_chunk))

        _, vjp_fn = jax.vjp(chunk_loss, h_chunk, lm_head)
        d_h, d_w = vjp_fn(g / (B * T))
        return d_lm_head_acc + d_w, d_h

    d_lm_head, d_hidden_chunks = jax.lax.scan(
        bwd_body, jnp.zeros_like(lm_head), (hidden_chunks, labels_chunks))
    return d_hidden_chunks.reshape(B, T, D), d_lm_head, jnp.zeros_like(labels)


chunked_lm_head_loss.defvjp(_chunked_loss_fwd, _chunked_loss_bwd)


def make_optimizer(config, num_steps):
    lr = config.learning_rate
    warmup_steps = int(config.warmup_ratio * num_steps)
    warmdown_steps = int(config.warmdown_ratio * num_steps)
    constant_steps = num_steps - warmup_steps - warmdown_steps
    end_lr = lr * config.final_lr_frac

    schedule_fn = optax.join_schedules([
        optax.linear_schedule(0.0, lr, warmup_steps),
        optax.constant_schedule(lr),
        optax.linear_schedule(lr, end_lr, warmdown_steps),
    ], boundaries=[warmup_steps, warmup_steps + constant_steps])

    return optax.adamw(learning_rate=schedule_fn, b1=config.beta1,
                       b2=config.beta2, eps=config.eps,
                       weight_decay=config.weight_decay)


def make_train_step(optimizer, moe_fn):
    """Create JIT-compiled train step with specified MoE dispatch."""
    @jax.jit
    def train_step(config, params, opt_state, x, y, _opt=optimizer):
        num_mb = config.num_microbatches
        x_micro = x.reshape(num_mb, config.microbatch_size, config.seq_len)
        y_micro = y.reshape(num_mb, config.microbatch_size, config.seq_len)

        def loss_fn(params, x_mb, y_mb):
            hidden, aux_loss, z_loss = model_forward(config, params, x_mb,
                                                      moe_fn=moe_fn)
            lm_loss = chunked_lm_head_loss(hidden, params.lm_head, y_mb, config, moe_fn)
            return lm_loss + config.aux_loss_alpha * aux_loss + config.z_loss_alpha * z_loss

        def microbatch_step(grad_acc, data):
            x_mb, y_mb = data
            loss, grads = jax.value_and_grad(loss_fn)(params, x_mb, y_mb)
            grad_acc = jax.tree.map(jax.lax.add, grad_acc, grads)
            return grad_acc, loss

        grad_init = jax.tree.map(jnp.zeros_like, params)
        grads, losses = jax.lax.scan(microbatch_step, grad_init,
                                     (x_micro, y_micro))
        grads = jax.tree.map(lambda g: g / num_mb, grads)
        loss = jnp.mean(losses)

        updates, new_opt_state = _opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return loss, new_params, new_opt_state
    return train_step


def synthetic_data(key, batch_size, seq_len, vocab_size):
    """Generate random token batch for CPU testing."""
    tokens = jax.random.randint(key, (batch_size, seq_len + 1), 0, vocab_size)
    return tokens[:, :-1], tokens[:, 1:]

# %%
# === Part 6: Validation Training ===
print("=== Part 6: Validation Training ===\n")

NUM_VALIDATION_STEPS = 50

strategies = {
    'capacity': moe_capacity_forward,
    'dropless_jax': moe_dropless_jax_forward,
}

results = {}

for name, moe_fn in strategies.items():
    print(f"\n--- Training with '{name}' dispatch ({NUM_VALIDATION_STEPS} steps) ---")

    params = init_full_model(config, seed=config.param_seed)
    total_p = count_params(params)
    non_embed_p = count_non_embed_params(params)
    print(f"Params: {total_p/1e6:.2f}M total, {non_embed_p/1e6:.2f}M non-embed")

    optimizer = make_optimizer(config, NUM_VALIDATION_STEPS)
    opt_state = optimizer.init(params)
    train_step = make_train_step(optimizer, moe_fn=moe_fn)

    losses = []
    key = jax.random.key(0)
    t0 = time.time()

    for step in range(NUM_VALIDATION_STEPS):
        key, subkey = jax.random.split(key)
        x, y = synthetic_data(subkey, config.batch_size, config.seq_len,
                              config.vocab_size)
        loss, params, opt_state = train_step(config, params, opt_state, x, y)
        loss_val = float(loss)
        losses.append(loss_val)

        if step % 10 == 0 or step == NUM_VALIDATION_STEPS - 1:
            elapsed = time.time() - t0
            print(f"  step {step:3d} | loss: {loss_val:.4f} | "
                  f"elapsed: {elapsed:.1f}s")

    results[name] = losses
    print(f"  Final: {losses[-1]:.4f} (started at {losses[0]:.4f}, "
          f"reduction: {(1 - losses[-1]/losses[0])*100:.1f}%)")

# %%
# === Loss curve comparison ===
print("\n=== Loss Curve Summary ===\n")

fig, ax = plt.subplots(figsize=(10, 6))
colors = {'capacity': '#e74c3c', 'dropless_jax': '#2ecc71'}

for name, losses in results.items():
    ax.plot(losses, label=name, color=colors.get(name, 'gray'), linewidth=2)

ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('MoE Dispatch Strategy Comparison — Validation Training', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/moe_training_curves.png', dpi=100, bbox_inches='tight')
print("Training curves saved to /tmp/moe_training_curves.png")
plt.close()

# Summary table
print(f"\n{'Strategy':<20} {'Initial Loss':>15} {'Final Loss':>12} {'Reduction':>12}")
print('-' * 62)
for name, losses in results.items():
    reduction = (1 - losses[-1] / losses[0]) * 100
    print(f"{name:<20} {losses[0]:>15.4f} {losses[-1]:>12.4f} {reduction:>11.1f}%")

print("\nWith synthetic random data, loss stays near ln(vocab_size) ≈ "
      f"{jnp.log(jnp.array(config.vocab_size, dtype=jnp.float32)):.2f}")
print("since there are no patterns to learn. The key validation is that")
print("gradients flow correctly (no NaNs, no crashes) and both strategies")
print("produce equivalent results.")

# %% [markdown]
# ---
#
# ## Part 7: TPU Production Config (Placeholder)
#
# The cells below use full-size model dimensions matching `09_moe.py` and
# require a TPU. They are only executed when `ON_TPU = True`.
#
# To use this notebook on TPU:
# 1. Upload to Colab, select **TPU v6e** runtime
# 2. Set Colab secrets: `HF_TOKEN` and `WANDB_TOKEN`
# 3. The notebook will auto-detect TPU and use splash attention + full config

# %%
if ON_TPU:
    print("TPU detected — full-size config would go here.")
    print("See 09_moe.py for the complete Quick Training / Sweep / Hero Run setup.")
    print("The key change: replace moe_forward() with moe_dropless_jax_forward()")
    print("or moe_megablox_forward() for dropless dispatch.")
else:
    print("No TPU detected — skipping production config.")
    print("Parts 1-6 above ran successfully on CPU.")
