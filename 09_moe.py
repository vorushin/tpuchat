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
# # 09 — MoE Training Lab (rev 15)
#
# Mixture of Experts variant of the
# [TPU Ablation Lab](https://github.com/vorushin/tpuchat/blob/master/08_tpu_ablations.ipynb).
# Replaces the dense MLP with a routed MoE layer (8 experts, top-2,
# capacity-based dispatch, ReLU² activation). Based on Karpathy's
# [nanochat](https://github.com/karpathy/nanochat) — ported to JAX for a
# single TPU v6e on Google Colab Pro+.
#
# **Runtime type:** In Colab, go to *Runtime → Change runtime type* and select
# **TPU v6e** (or v5e).
#
# **Three modes:**
# 1. **Quick Training** (~300 steps) — XProf capture, MFU measurement
# 2. **Sweep** (wandb) — Bayesian LR search
# 3. **Hero Run** (20 tok/param) — Full training with eval + HuggingFace upload
#
# ### MoE Architecture: E=8, K=2, F_expert=512, D=1024, N=4, K_head=1, H=256, L=8, B=64, T=2048
# | Metric | Value |
# |--------|-------|
# | Total params | ~189M |
# | Non-embed params | ~156M |
# | Active params/token | ~88M (attention + 2 experts) |
# | Experts | 8 × ReLU² (dim=512), top-2 routed |
# | Load balancing | aux_loss (0.01) + z-loss (1e-4) |
#
# ### Setup: Colab Secrets
#
# | Secret | Where to get it | Used by |
# |--------|----------------|---------|
# | `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Downloads tokenizer from `vorushin/tpuchat` |
# | `WANDB_TOKEN` | [wandb.ai/authorize](https://wandb.ai/authorize) | Sweep and Hero Run |

# %%
# !pip install -q "jax[tpu]" optax huggingface_hub tiktoken pyarrow requests wandb tensorboard tensorboard-plugin-profile plotly

# %% [markdown]
# ## Prerequisites
#
# Loads the data and the tokenizer (trained in
# [01_tokenizer.ipynb](https://github.com/vorushin/tpuchat/blob/master/01_tokenizer.ipynb)).

# %%
# === Imports, TPU constants, dot_dict ===
import functools as ft
import time
import os
import pickle
import queue
import threading
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

# TPU v6e-1 constants
PEAK_TFLOPS = 918          # bf16 peak compute per chip

REVISION = 15

print(f"JAX version : {jax.__version__}")
print(f"Devices     : {jax.devices()}")
print(f"Peak TFLOPS : {PEAK_TFLOPS} (bf16, from v6e docs)")
print(f"Notebook rev: {REVISION}")


# JAX pytree with dot-notation access
# (from https://docs.jax.dev/en/latest/the-training-cookbook.html)
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

# %%
# === RMSNorm, RoPE ===

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

# %%
# === Attention backends (einsum + splash) ===

def _expand_kv(k, v, n_head, n_kv_head):
    """Repeat KV heads to match Q head count for non-splash backends."""
    if n_kv_head == n_head:
        return k, v
    ratio = n_head // n_kv_head
    return jnp.repeat(k, ratio, axis=1), jnp.repeat(v, ratio, axis=1)

# %%
# === FLOP counting helpers ===
# Dimension notation: B=batch, T=seq_len, D=d_model, N=n_heads,
#   K_head=n_kv_heads, H=head_dim, E=n_experts, K=n_active_experts,
#   F=expert_mlp_dim, L=n_layers, V=vocab_size

def matmul_flops(M, N, K, batch=1):
    """FLOPs for [M,K] @ [K,N].  2*M*N*K per batch element."""
    return 2 * batch * M * N * K


def attention_flops(B, N, T, H):
    """FLOPs for QK^T + AV (full T×T, not causal-halved)."""
    return 2 * (2 * B * N * T * T * H)


def moe_layer_flops(B, T, D, N, K_head, H, E, K, F_expert):
    """MXU-relevant FLOPs for one MoE transformer layer.

    Counts matmul FLOPs: attention projections + attention core + MoE MLP.
    MoE MLP counts only K active experts (not all E).
    """
    tok = B * T
    q    = 2 * tok * D * N * H          # Q projection
    k    = 2 * tok * D * K_head * H     # K projection
    v    = 2 * tok * D * K_head * H     # V projection
    att  = attention_flops(B, N, T, H)  # core attention
    proj = 2 * tok * N * H * D          # output projection
    # MoE: K active experts × ReLU² (up + down = 2 matmuls)
    moe  = K * 2 * (2 * tok * D * F_expert)
    router = 2 * tok * D * E
    return q + k + v + att + proj + moe + router

# %%
# === Data: HF login, tokenizer, data download, tokenize_shards, PrefetchDataLoader ===
import requests
from multiprocessing import Pool
import pyarrow.parquet as pq
from huggingface_hub import login, hf_hub_download

HF_REPO_ID = 'vorushin/tpuchat'
DATA_DIR = '/content/base_data'
TOKENIZER_DIR = '/content/tokenizer'
MAX_CHARS_PER_DOC = 10_000
NUM_TRAIN_SHARDS = 50
NUM_VAL_SHARDS = 2

# --- HF login + tokenizer ---
from google.colab import userdata
login(token=userdata.get("HF_TOKEN"))

os.makedirs(TOKENIZER_DIR, exist_ok=True)
hf_hub_download(repo_id=HF_REPO_ID, filename='tokenizer/tokenizer.pkl',
                local_dir=TOKENIZER_DIR)
print(f'Downloaded tokenizer to {TOKENIZER_DIR}')

with open(os.path.join(TOKENIZER_DIR, 'tokenizer', 'tokenizer.pkl'), 'rb') as f:
    enc = pickle.load(f)
print(f'Loaded tokenizer: vocab_size={enc.n_vocab}')

# --- Download data shards ---
BASE_URL = 'https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main'
os.makedirs(DATA_DIR, exist_ok=True)

def download_shard(index):
    filename = f'shard_{index:05d}.parquet'
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True
    url = f'{BASE_URL}/{filename}'
    print(f'Downloading {filename}...')
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            tmp = filepath + '.tmp'
            with open(tmp, 'wb') as f:
                for chunk in resp.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(tmp, filepath)
            return True
        except Exception as e:
            print(f'Attempt {attempt}/3 failed for {filename}: {e}')
            for p in [filepath + '.tmp', filepath]:
                if os.path.exists(p):
                    os.remove(p)
            if attempt < 3:
                time.sleep(2 ** attempt)
    return False

total_shards = NUM_TRAIN_SHARDS + NUM_VAL_SHARDS
t0 = time.time()
with Pool(8) as pool:
    results = pool.map(download_shard, range(total_shards))
print(f'\nDownloaded {sum(results)}/{total_shards} shards in {time.time()-t0:.1f}s')


# --- Tokenize shards ---
def tokenize_shards(shard_indices, batch_size, seq_len):
    """Yield (x, y) batches by tokenizing parquet shards on the fly."""
    bos_id = enc.encode_single_token('<|bos|>')
    buf = []

    while True:  # loop over epochs
        for shard_idx in shard_indices:
            filepath = os.path.join(DATA_DIR, f'shard_{shard_idx:05d}.parquet')
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                texts = rg.column('text').to_pylist()
                for doc in texts:
                    if len(doc) > MAX_CHARS_PER_DOC:
                        doc = doc[:MAX_CHARS_PER_DOC]
                    tokens = [bos_id] + enc.encode_ordinary(doc)
                    buf.extend(tokens)

                    tokens_per_batch = batch_size * (seq_len + 1)
                    while len(buf) >= tokens_per_batch:
                        batch_tokens = np.array(buf[:tokens_per_batch], dtype=np.int32)
                        batch_tokens = batch_tokens.reshape(batch_size, seq_len + 1)
                        x = batch_tokens[:, :-1]
                        y = batch_tokens[:, 1:]
                        buf = buf[tokens_per_batch:]
                        yield x, y


train_shard_indices = list(range(NUM_TRAIN_SHARDS))
val_shard_indices = list(range(NUM_TRAIN_SHARDS, NUM_TRAIN_SHARDS + NUM_VAL_SHARDS))
print(f'Train shards: {len(train_shard_indices)}, Val shards: {len(val_shard_indices)}')


# --- PrefetchDataLoader ---
@dataclass
class PrefetchDataLoader:
    """Wraps an iterator and prefetches items in a background thread."""
    iterator: any
    capacity: int = 2

    def __post_init__(self):
        self.queue = queue.Queue(maxsize=self.capacity)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        try:
            for item in self.iterator:
                if self.stop_event.is_set():
                    break
                x, y = item
                item = (jax.device_put(jnp.array(x)), jax.device_put(jnp.array(y)))
                self.queue.put(item)
        except Exception as e:
            print(f"Prefetch worker error: {e}")
            self.stop_event.set()
        finally:
            self.stop_event.set()

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop_event.is_set() and self.queue.empty():
            raise StopIteration
        return self.queue.get()

    def stop(self):
        self.stop_event.set()

# %%
# === Optimizer: AdamW with warmup + linear warmdown ===

def make_optimizer(config, num_steps):
    """Create optax AdamW with linear warmup + constant + linear warmdown."""
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

# %%
# === count_params, count_non_embed_params ===

def count_params(params):
    """Count total parameters."""
    return sum(p.size for p in jax.tree.leaves(params) if isinstance(p, jax.Array))


def count_non_embed_params(params):
    """Non-embedding params (unembed + layers). Excludes wte (lookup table)."""
    return count_params(params) - params.wte.size

# %% [markdown]
# ## Model
#
# Transformer with Mixture of Experts MLP. Architecture: RoPE, RMSNorm, MQA
# (4 query heads, 1 KV head), QK-norm, logit softcap, **MoE ReLU²** (8
# experts, top-2 routed, capacity-based dispatch), AdamW optimizer. Auxiliary
# load balancing loss + router z-loss for training stability.

# %%

@jax.tree_util.register_static
@dataclass(kw_only=True, frozen=True)
class Config:
    # ── MoE ────────────────────────────────────────────────────
    n_experts: int = 8              # number of routed experts
    n_active_experts: int = 2       # top-k experts activated per token
    expert_mlp_dim: int = 2048      # per-expert FFN hidden dim (ReLU²)
    capacity_factor: float = 1.25   # expert buffer headroom (1.0 = exact, 1.25 = 25% extra)
    aux_loss_alpha: float = 0.01    # load balancing loss coefficient
    z_loss_alpha: float = 1e-4      # router z-loss coefficient

    # ── Architecture ───────────────────────────────────────────
    attn_impl: str = 'splash'       # 'splash' | 'einsum'
    qk_norm: bool = True            # QK-norm on queries and keys
    n_embd: int = 1024
    n_layer: int = 8
    seq_len: int = 2048
    vocab_size: int = 32768
    n_head: int = 4
    n_kv_head: int = 1
    head_dim: int = 256
    softcap: float = 15.0
    logit_dtype: str = 'bf16'       # 'bf16' or 'fp32'
    splash_block_size: int = 1024
    num_lm_head_chunks: int = 8
    batch_size: int = 64
    microbatch_size: int = 4

    # ── Training ───────────────────────────────────────────────
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    warmup_ratio: float = 0.02
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0

    # ── Eval / Data ────────────────────────────────────────────
    eval_steps: int = 10
    param_seed: int = 42

    @property
    def num_microbatches(self):
        return self.batch_size // self.microbatch_size


config = Config()
assert config.vocab_size % 256 == 0, f"vocab_size must be divisible by 256, got {config.vocab_size}"
print(f'Config: D={config.n_embd}, L={config.n_layer}, T={config.seq_len}, '
      f'V={config.vocab_size}, N={config.n_head}, K_head={config.n_kv_head}, '
      f'H={config.head_dim}')
print(f'MoE: E={config.n_experts}, top-{config.n_active_experts}, '
      f'F_expert={config.expert_mlp_dim}, '
      f'aux_alpha={config.aux_loss_alpha}, z_alpha={config.z_loss_alpha}')
mb_info = (f', microbatch={config.microbatch_size}, accum={config.num_microbatches}x'
           if config.num_microbatches > 1 else '')
print(f'Training: lr={config.learning_rate:.1e}, B={config.batch_size}{mb_info}')

# %%

def init_layer_params(config, seed=42):
    """Initialize params for one transformer layer (attention + MoE ReLU²)."""
    key = jax.random.key(seed)
    keys = jax.random.split(key, 7)
    s = (3.0 ** 0.5) * (config.n_embd ** -0.5)
    layer = dot_dict()

    # Attention projections
    layer.c_q = jax.random.uniform(keys[0], (config.n_embd, config.n_head, config.head_dim),
                                    dtype=jnp.bfloat16, minval=-s, maxval=s)
    layer.c_k = jax.random.uniform(keys[1], (config.n_embd, config.n_kv_head, config.head_dim),
                                    dtype=jnp.bfloat16, minval=-s, maxval=s)
    layer.c_v = jax.random.uniform(keys[2], (config.n_embd, config.n_kv_head, config.head_dim),
                                    dtype=jnp.bfloat16, minval=-s, maxval=s)
    layer.c_proj = jnp.zeros((config.n_head, config.head_dim, config.n_embd), dtype=jnp.bfloat16)

    # Router: small init for stable early routing
    layer.router = jax.random.normal(keys[3], (config.n_embd, config.n_experts),
                                      dtype=jnp.bfloat16) * 0.01

    # Expert ReLU² weights: stacked (E, D, F) and (E, F, D)
    E, D, F = config.n_experts, config.n_embd, config.expert_mlp_dim
    layer.expert_w_up   = jax.random.uniform(keys[4], (E, D, F),
                                              dtype=jnp.bfloat16, minval=-s, maxval=s)
    layer.expert_w_down = jnp.zeros((E, F, D), dtype=jnp.bfloat16)
    return layer


def moe_forward(config, layer, x):
    """MoE forward with capacity-based dispatch.

    1. Route: top-k expert selection with softmax weights
    2. Dispatch: scatter tokens into (E, C, D) expert buffer via cumsum positions
    3. Compute: batched ReLU² via einsums over expert dimension
    4. Combine: gather outputs, weight by routing scores, sum over K experts

    Returns: (output, aux_loss, z_loss)
    """
    B, T, D = x.shape
    N = B * T
    E = config.n_experts
    K = config.n_active_experts
    x_flat = x.reshape(N, D)

    # ── Router ──
    router_logits = jnp.einsum('nd,de->ne', x_flat, layer.router)  # (N, E)

    # Z-loss: penalize large logits for numerical stability
    z_loss = jnp.mean(jax.nn.logsumexp(router_logits, axis=-1) ** 2)

    # Top-k selection
    router_probs = jax.nn.softmax(router_logits, axis=-1)          # (N, E)
    top_k_logits, top_k_idx = jax.lax.top_k(router_logits, K)     # (N, K)
    top_k_weights = jax.nn.softmax(top_k_logits, axis=-1)         # (N, K)

    # Auxiliary load balancing loss (Switch Transformer formulation)
    expert_mask = jax.nn.one_hot(top_k_idx, E)                     # (N, K, E)
    f = jnp.sum(expert_mask, axis=(0, 1)) / (N * K)                # token fraction per expert
    P = jnp.mean(router_probs, axis=0)                             # mean prob per expert
    aux_loss = E * jnp.sum(f * P)

    # ── Dispatch ──
    # Flatten K selections: each token appears K times
    expert_flat = top_k_idx.reshape(N * K)                          # (N*K,) expert IDs
    weight_flat = top_k_weights.reshape(N * K)                     # (N*K,) routing weights
    x_rep = jnp.repeat(x_flat, K, axis=0)                          # (N*K, D)

    # Position within each expert via cumsum on one-hot mask
    expert_oh = jax.nn.one_hot(expert_flat, E)                      # (N*K, E)
    cumpos = jnp.cumsum(expert_oh, axis=0) * expert_oh
    pos = (jnp.sum(cumpos, axis=-1) - 1).astype(jnp.int32)        # 0-indexed

    # Capacity: ceil(N*K / E) × capacity_factor
    C = int(((N * K + E - 1) // E) * config.capacity_factor)
    valid = pos < C
    pos_clipped = jnp.clip(pos, 0, C - 1)

    # Scatter into expert buffer
    expert_input = jnp.zeros((E, C, D), dtype=x_flat.dtype)
    expert_input = expert_input.at[expert_flat, pos_clipped].add(
        x_rep * valid[:, None])

    # ── Expert ReLU² (batched over E dimension) ──
    up = jnp.einsum('ecd,edf->ecf', expert_input, layer.expert_w_up)
    up = jax.nn.relu(up) ** 2
    expert_out = jnp.einsum('ecf,efd->ecd', up, layer.expert_w_down)  # (E, C, D)

    # ── Combine ──
    gathered = expert_out[expert_flat, pos_clipped]                  # (N*K, D)
    weighted = gathered * weight_flat[:, None] * valid[:, None]
    output = weighted.reshape(N, K, D).sum(axis=1)                  # (N, D)

    return output.reshape(B, T, D), aux_loss, z_loss


def single_layer_forward(config, layer, x, cos, sin, layer_idx=0):
    """Forward pass for one transformer layer (attention + MoE)."""
    h = rms_norm(x)

    with jax.named_scope(f'layer_{layer_idx}/attention'):
        q = jnp.einsum('btd,dhk->bhtk', h, layer.c_q)
        k = jnp.einsum('btd,dhk->bhtk', h, layer.c_k)
        v = jnp.einsum('btd,dhk->bhtk', h, layer.c_v)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if config.qk_norm:
            q = rms_norm(q)
            k = rms_norm(k)

        seq_len = x.shape[1]

        if config.attn_impl == 'splash':
            from jax.experimental.pallas.ops.tpu.splash_attention import (
                splash_attention_mask, splash_attention_kernel)

            smask = splash_attention_mask.CausalMask(shape=(seq_len, seq_len))
            mh_mask = splash_attention_mask.MultiHeadMask(
                masks=[smask] * config.n_head)
            bs = min(config.splash_block_size, seq_len)
            block_sizes = splash_attention_kernel.BlockSizes(
                block_q=bs, block_kv=bs,
                block_q_dkv=bs, block_kv_dkv=bs,
                block_q_dq=bs, block_kv_dq=bs)
            kernel = splash_attention_kernel.make_splash_mha(
                mask=mh_mask, head_shards=1, q_seq_shards=1,
                block_sizes=block_sizes)
            attn_out = jax.vmap(kernel)(q, k, v)

        elif config.attn_impl == 'einsum':
            k_exp, v_exp = _expand_kv(k, v, config.n_head, config.n_kv_head)
            scale = config.head_dim ** -0.5
            scores = jnp.einsum('bhtd,bhsd->bhts', q, k_exp) * scale
            rows = jnp.arange(seq_len)[:, None]
            cols = jnp.arange(seq_len)[None, :]
            mask = cols <= rows
            scores = jnp.where(mask[None, None, :, :], scores,
                               jnp.finfo(scores.dtype).min)
            attn_weights = jax.nn.softmax(scores, axis=-1)
            attn_out = jnp.einsum('bhts,bhsd->bhtd', attn_weights, v_exp)

        attn_out = jnp.einsum('bhtd,hde->bte', attn_out, layer.c_proj)

    x = x + attn_out

    with jax.named_scope(f'layer_{layer_idx}/moe'):
        h2 = rms_norm(x)
        moe_out, aux_loss, z_loss = moe_forward(config, layer, h2)

    x = x + moe_out
    return x, aux_loss, z_loss


def init_all_layers(config, n_layers, seed=42):
    layers = dot_dict()
    for i in range(n_layers):
        layers[i] = init_layer_params(config, seed=seed + i * 7)
    return layers


def init_full_model(config, seed=42):
    """Initialize all model params (embed + layers + lm_head)."""
    key = jax.random.key(seed)
    params = dot_dict()
    key, k1, k2 = jax.random.split(key, 3)
    params.wte = jax.random.normal(k1, (config.vocab_size, config.n_embd),
                                    dtype=jnp.bfloat16)
    params.lm_head = jax.random.normal(k2, (config.n_embd, config.vocab_size),
                                        dtype=jnp.bfloat16) * 0.001
    params.layers = init_all_layers(config, config.n_layer, seed=seed + 100)
    return params


def model_forward(config, params, tokens):
    """Full forward: embed -> layers -> final_norm.

    Returns: (hidden, avg_aux_loss, avg_z_loss)
    """
    B, T = tokens.shape
    cos, sin = precompute_rope(T, config.head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    with jax.named_scope('embedding'):
        x = rms_norm(params.wte[tokens])
    total_aux = 0.0
    total_z = 0.0
    for i in range(config.n_layer):
        x, aux, z = single_layer_forward(config, params.layers[i], x, cos, sin, layer_idx=i)
        total_aux += aux
        total_z += z
    return rms_norm(x), total_aux / config.n_layer, total_z / config.n_layer


def _logit_dtype(config):
    return jnp.float32 if config.logit_dtype == 'fp32' else jnp.bfloat16


# Chunked LM head loss, reduces HBM usage by the last matmul (hidden_dim -> vocab_dim).
# Extracted from maxtext.


def _logits_from_chunk(h_chunk, lm_head, config):
    logits = jnp.einsum('td,dv->tv', h_chunk, lm_head,
                        preferred_element_type=_logit_dtype(config))
    return config.softcap * jnp.tanh(logits / config.softcap)


@ft.partial(jax.custom_vjp, nondiff_argnums=(3,))
def chunked_lm_head_loss(hidden, lm_head, labels, config):
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


def _chunked_loss_fwd(hidden, lm_head, labels, config):
    loss = chunked_lm_head_loss(hidden, lm_head, labels, config)
    return loss, (hidden, lm_head, labels)


def _chunked_loss_bwd(config, residuals, g):
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

    d_lm_head_init = jnp.zeros_like(lm_head)
    d_lm_head, d_hidden_chunks = jax.lax.scan(
        bwd_body, d_lm_head_init, (hidden_chunks, labels_chunks))
    return d_hidden_chunks.reshape(B, T, D), d_lm_head, jnp.zeros_like(labels)


chunked_lm_head_loss.defvjp(_chunked_loss_fwd, _chunked_loss_bwd)


def make_train_step(optimizer):
    """Create a JIT-compiled train step with MoE auxiliary losses.

    Total loss = LM loss + aux_loss_alpha * aux_loss + z_loss_alpha * z_loss.
    """
    @jax.jit
    def train_step(config, params, opt_state, x, y, _opt=optimizer):
        num_mb = config.num_microbatches

        # Reshape full batch into microbatches: (B,T) -> (num_mb, mb_size, T)
        x_micro = x.reshape(num_mb, config.microbatch_size, config.seq_len)
        y_micro = y.reshape(num_mb, config.microbatch_size, config.seq_len)

        def loss_fn(params, x_mb, y_mb):
            hidden, aux_loss, z_loss = model_forward(config, params, x_mb)
            lm_loss = chunked_lm_head_loss(hidden, params.lm_head, y_mb, config)
            return lm_loss + config.aux_loss_alpha * aux_loss + config.z_loss_alpha * z_loss

        def microbatch_step(grad_acc, data):
            x_mb, y_mb = data
            loss, grads = jax.value_and_grad(loss_fn)(params, x_mb, y_mb)
            grad_acc = jax.tree.map(jax.lax.add, grad_acc, grads)
            return grad_acc, loss

        with jax.named_scope('forward_backward'):
            grad_init = jax.tree.map(jnp.zeros_like, params)
            grads, losses = jax.lax.scan(microbatch_step, grad_init,
                                         (x_micro, y_micro))
            grads = jax.tree.map(lambda g: g / num_mb, grads)
            loss = jnp.mean(losses)

        with jax.named_scope('optimizer'):
            updates, new_opt_state = _opt.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

        return loss, new_params, new_opt_state
    return train_step


@jax.jit
def eval_step(config, params, x, y):
    """JIT-compiled eval: returns LM loss only (no aux/z losses)."""
    hidden, _, _ = model_forward(config, params, x)
    return chunked_lm_head_loss(hidden, params.lm_head, y, config)


@jax.jit
def predict_step(config, params, x):
    """JIT-compiled single step inference: returns logits."""
    hidden, _, _ = model_forward(config, params, x)
    with jax.named_scope('lm_head'):
        logits = jnp.einsum('btd,dv->btv', hidden, params.lm_head,
                            preferred_element_type=_logit_dtype(config))
        logits = config.softcap * jnp.tanh(logits / config.softcap)
    return logits


def generate(config, params, enc, prompt, max_new_tokens=64,
             temperature=0.8, top_k=50):
    """Generate text from a prompt using top-k + temperature sampling."""
    bos_id = enc.encode_single_token('<|bos|>')
    ids = [bos_id] + enc.encode_ordinary(prompt)
    key = jax.random.key(42)

    for _ in range(max_new_tokens):
        context = ids[-config.seq_len:]
        pad_len = config.seq_len - len(context)
        x = jnp.array([context + [0] * pad_len], dtype=jnp.int32)
        logits = predict_step(config, params, x)
        logits.block_until_ready()
        next_logits = logits[0, len(context) - 1, :]

        if temperature == 0:
            next_id = int(jnp.argmax(next_logits))
        else:
            next_logits = next_logits / temperature
            if top_k > 0:
                top_vals = jax.lax.top_k(next_logits, top_k)[0]
                next_logits = jnp.where(next_logits >= top_vals[-1],
                                        next_logits, -1e10)
            key, subkey = jax.random.split(key)
            next_id = int(jax.random.categorical(subkey, next_logits))
        ids.append(next_id)

    return enc.decode(ids)

# %% [markdown]
# ## Quick Training (XProf)
#
# Trains for 300 steps — outputs MFU and throughput. Also runs the TPU profiler
# XProf for a few steps and shows the profiling results.

# %%

NUM_QUICK_STEPS = 300
EVAL_EVERY = 100
XPROF_START, XPROF_END = 15, 20
LOG_DIR = '/content/log_dir'

# Init model + optimizer
params = init_full_model(config, seed=config.param_seed)
total_p = count_params(params)
non_embed_p = count_non_embed_params(params)
# Active non-embed: non-embed params minus inactive expert params
expert_params_per_layer = config.n_experts * (2 * config.n_embd * config.expert_mlp_dim)
inactive_expert_p = config.n_layer * expert_params_per_layer * (config.n_experts - config.n_active_experts) / config.n_experts
active_non_embed_p = non_embed_p - int(inactive_expert_p)
print(f'Params: {total_p/1e6:.1f}M total, {non_embed_p/1e6:.1f}M non-embed, '
      f'{active_non_embed_p/1e6:.1f}M active non-embed')
print(f'Batch: {config.batch_size} x {config.seq_len} = '
      f'{config.batch_size * config.seq_len:,} tokens/step')

optimizer = make_optimizer(config, NUM_QUICK_STEPS)
opt_state = optimizer.init(params)
train_step = make_train_step(optimizer)

# Data
raw_train = tokenize_shards(train_shard_indices, config.batch_size, config.seq_len)
train_loader = PrefetchDataLoader(raw_train, capacity=4)
val_loader_fn = lambda: tokenize_shards(val_shard_indices, config.batch_size, config.seq_len)

# FLOP counting (MoE: counts only K active experts)
fwd_flops = (config.n_layer * moe_layer_flops(
    config.batch_size, config.seq_len, config.n_embd,
    config.n_head, config.n_kv_head, config.head_dim,
    config.n_experts, config.n_active_experts, config.expert_mlp_dim)
    + matmul_flops(config.batch_size * config.seq_len,
                   config.vocab_size, config.n_embd))
step_flops = 3 * fwd_flops  # fwd + 2x bwd

smooth_loss = 0.0
window_t0 = None
flops_per_tok = step_flops / (config.batch_size * config.seq_len)
ideal_tok_s = PEAK_TFLOPS * 1e12 / flops_per_tok

def report_mfu(label, t0, n_steps):
    wall = time.time() - t0
    tok_per_s = int(n_steps * config.batch_size * config.seq_len / wall)
    mfu_pct = (tok_per_s * flops_per_tok) / (PEAK_TFLOPS * 1e12) * 100
    print(f'{label} MFU: {mfu_pct:.1f}% | tok/s: {tok_per_s:,} | '
          f'ideal tok/s (100% MFU): {int(ideal_tok_s):,}')

print(f'\n=== Quick Training: {NUM_QUICK_STEPS} steps ===\n')

for step in range(NUM_QUICK_STEPS + 1):
    last_step = (step == NUM_QUICK_STEPS)

    # --- Eval ---
    if step % EVAL_EVERY == 0 or last_step:
        val_loader = val_loader_fn()
        val_losses = []
        for _ in range(config.eval_steps):
            vx, vy = next(val_loader)
            vx, vy = jnp.array(vx), jnp.array(vy)
            vl = eval_step(config, params, vx, vy)
            val_losses.append(float(vl))
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f'step {step:05d} | Val loss: {avg_val_loss:.4f}')
        window_t0 = time.time()

    if last_step:
        break

    # --- XProf ---
    if step == XPROF_START:
        jax.profiler.start_trace(LOG_DIR)
        print("XProf started...")
    if step == XPROF_END:
        jax.profiler.stop_trace()
        print(f"XProf stopped. Trace saved to '{LOG_DIR}'.")

    # --- Train step ---
    x_batch, y_batch = next(train_loader)
    loss, params, opt_state = train_step(config, params, opt_state,
                                          x_batch, y_batch)
    loss.block_until_ready()

    loss_val = float(loss)
    ema_beta = 0.9
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
    debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))

    if (step + 1) % 50 == 0:
        if step >= 50:  # skip first 50 (XProf + warmup)
            report_mfu(f'steps {step - 48}-{step}', window_t0, 50)
        window_t0 = time.time()
        print(f'step {step:05d}/{NUM_QUICK_STEPS} | loss: {debiased_loss:.4f}')

train_loader.stop()

# --- Sample text ---
print('\n--- Samples ---')
for prompt in ['The capital of France is', 'Machine learning is']:
    text = generate(config, params, enc, prompt, max_new_tokens=64)
    print(f'Prompt: {prompt}\nOutput: {text}\n')

# %%
# === View Profiling Results ===
# Run this cell to load TensorBoard and view the trace captured in steps 15-20.

%load_ext tensorboard
%tensorboard --logdir /content/log_dir

# %% [markdown]
# ## Sweep (wandb)
#
# Bayesian LR search for the MoE config. Feel free to add `aux_loss_alpha`
# or `expert_mlp_dim` to the sweep parameters.

# %%
# === wandb LR sweep ===
import wandb
from google.colab import userdata

wandb.login(key=userdata.get("WANDB_TOKEN"))

sweep_config = {
    "name": f"moe-E{config.n_experts}-K{config.n_active_experts}-F{config.expert_mlp_dim}",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values",
                          "min": 5e-4, "max": 1e-2},
    },
}

SWEEP_PROJECT = "tpuchat-moe"
SWEEP_ID = None            # set to existing sweep ID to continue
SWEEP_STEPS = 2_500
SWEEP_EVAL_EVERY = 250


def sweep_train_fn():
    """Single training run within a wandb sweep."""
    run = wandb.init()
    lr = wandb.config.learning_rate

    cfg = Config(learning_rate=lr)
    print(f'Sweep run: lr={lr:.2e}, E={cfg.n_experts}, '
          f'K={cfg.n_active_experts}, F={cfg.expert_mlp_dim}')

    wandb.define_metric("train/loss", step_metric="step")
    wandb.define_metric("train/tok_per_sec", step_metric="step")
    wandb.define_metric("val/loss", step_metric="step")
    wandb.define_metric("val_loss", step_metric="step")

    # Init
    params = init_full_model(cfg, seed=cfg.param_seed)
    total_p = count_params(params)
    non_embed_p = count_non_embed_params(params)
    print(f'Params: {total_p/1e6:.1f}M total, {non_embed_p/1e6:.1f}M non-embed')

    sweep_opt = make_optimizer(cfg, SWEEP_STEPS)
    opt_state = sweep_opt.init(params)
    sweep_train_step = make_train_step(sweep_opt)

    raw_train = tokenize_shards(train_shard_indices, cfg.batch_size, cfg.seq_len)
    train_loader = PrefetchDataLoader(raw_train, capacity=4)
    val_loader_fn = lambda: tokenize_shards(val_shard_indices, cfg.batch_size, cfg.seq_len)

    total_batch_size = cfg.batch_size * cfg.seq_len
    smooth_loss = 0.0
    best_val_loss = float('inf')

    print(f'\n=== Sweep run: {SWEEP_STEPS} steps ===\n')

    try:
        for step in range(SWEEP_STEPS + 1):
            last_step = (step == SWEEP_STEPS)

            # --- Eval ---
            if step % SWEEP_EVAL_EVERY == 0 or last_step:
                val_loader = val_loader_fn()
                val_losses = []
                for _ in range(cfg.eval_steps):
                    vx, vy = next(val_loader)
                    vx, vy = jnp.array(vx), jnp.array(vy)
                    vl = eval_step(cfg, params, vx, vy)
                    val_losses.append(float(vl))
                avg_val_loss = sum(val_losses) / len(val_losses)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss

                wandb.log({
                    "step": step,
                    "val/loss": avg_val_loss,
                    "val_loss": avg_val_loss,
                })
                print(f'step {step:05d} | Val loss: {avg_val_loss:.4f} '
                      f'(best: {best_val_loss:.4f})')

            if last_step:
                break

            # --- Train ---
            t0 = time.time()
            x_batch, y_batch = next(train_loader)
            loss, params, opt_state = sweep_train_step(cfg, params, opt_state,
                                                        x_batch, y_batch)
            loss.block_until_ready()
            dt = time.time() - t0

            loss_val = float(loss)
            ema_beta = 0.9
            smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
            debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))

            if step % 50 == 0:
                tok_per_sec = int(total_batch_size / dt) if dt > 0 else 0
                wandb.log({
                    "step": step,
                    "train/loss": debiased_loss,
                    "train/tok_per_sec": tok_per_sec,
                })
                print(f'step {step:05d}/{SWEEP_STEPS} | loss: {debiased_loss:.4f} '
                      f'| tok/s: {tok_per_sec:,}')

    finally:
        train_loader.stop()

    wandb.finish()
    print(f'Run complete. Best val loss: {best_val_loss:.4f}')


sweep_id = SWEEP_ID or wandb.sweep(sweep_config, project=SWEEP_PROJECT)
print(f"{'Continuing' if SWEEP_ID else 'New'} sweep: {sweep_id}")
wandb.agent(sweep_id, function=sweep_train_fn, count=10, project=SWEEP_PROJECT)

# --- Disconnect runtime to stop billing ---
from google.colab import runtime
runtime.unassign()

# %%
# === Plot sweep results: LR vs val_loss ===
import wandb
import plotly.graph_objects as go

from google.colab import userdata

wandb.login(key=userdata.get("WANDB_TOKEN"))

SWEEP_PROJECT_AND_ID = "tpuchat-moe/REPLACE_WITH_SWEEP_ID"

api = wandb.Api()
sweep = api.sweep(SWEEP_PROJECT_AND_ID)
runs = [r for r in sweep.runs if r.state == "finished"]

lrs = [r.config["learning_rate"] for r in runs]
val_losses = [r.summary["val_loss"] for r in runs]
names = [r.name for r in runs]

best_idx = val_losses.index(min(val_losses))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=lrs, y=val_losses, mode='markers',
    marker=dict(size=10, color='steelblue'),
    text=names, hovertemplate='%{text}<br>LR: %{x:.2e}<br>val_loss: %{y:.4f}<extra></extra>',
    name='runs',
))
fig.add_trace(go.Scatter(
    x=[lrs[best_idx]], y=[val_losses[best_idx]], mode='markers',
    marker=dict(size=14, color='crimson', symbol='star'),
    hovertemplate=f'{names[best_idx]}<br>LR: {lrs[best_idx]:.2e}<br>val_loss: {val_losses[best_idx]:.4f}<extra></extra>',
    name=f'best (LR={lrs[best_idx]:.2e})',
))
fig.update_layout(
    title='Sweep: Learning Rate vs Val Loss',
    xaxis=dict(title='Learning Rate', type='log'),
    yaxis=dict(title='Val Loss'),
    template='plotly_white',
    showlegend=True,
    width=700, height=450,
)
fig.show()

# %% [markdown]
# ## Hero Run (20 tok/param)
#
# Full training with MoE. Uses 20 tokens per non-embed parameter.

# %%
# === Hero run: 20 tok/param ===
import wandb
from google.colab import userdata

SAVE_CHECKPOINTS = False
CHECKPOINT_DIR = '/content/checkpoints'

# Compute steps from tok/param ratio
hero_config = Config()
hero_params = init_full_model(hero_config, seed=hero_config.param_seed)
hero_non_embed = count_non_embed_params(hero_params)
hero_total_p = count_params(hero_params)

target_tokens = int(20 * hero_non_embed)
total_batch_size = hero_config.batch_size * hero_config.seq_len
HERO_STEPS = target_tokens // total_batch_size
HERO_EVAL_EVERY = 1000

print(f'Params: {hero_total_p/1e6:.1f}M total, {hero_non_embed/1e6:.1f}M non-embed')
print(f'Target tokens: {target_tokens:,} (20 tok/param)')
print(f'Steps: {HERO_STEPS:,} ({total_batch_size:,} tok/step)')

# Init optimizer
hero_opt = make_optimizer(hero_config, HERO_STEPS)
opt_state = hero_opt.init(hero_params)
hero_train_step = make_train_step(hero_opt)

# Data
raw_train = tokenize_shards(train_shard_indices, hero_config.batch_size, hero_config.seq_len)
train_loader = PrefetchDataLoader(raw_train, capacity=4)
val_loader_fn = lambda: tokenize_shards(val_shard_indices, hero_config.batch_size, hero_config.seq_len)

# FLOP counting for MFU
fwd_flops = (hero_config.n_layer * moe_layer_flops(
    hero_config.batch_size, hero_config.seq_len, hero_config.n_embd,
    hero_config.n_head, hero_config.n_kv_head, hero_config.head_dim,
    hero_config.n_experts, hero_config.n_active_experts, hero_config.expert_mlp_dim)
    + matmul_flops(hero_config.batch_size * hero_config.seq_len,
                   hero_config.vocab_size, hero_config.n_embd))
step_flops = 3 * fwd_flops

# wandb
wandb.login(key=userdata.get("WANDB_TOKEN"))
wandb.init(project="tpuchat-moe",
           name=f"hero-moe-E{hero_config.n_experts}-K{hero_config.n_active_experts}-F{hero_config.expert_mlp_dim}",
           config={
               "n_experts": hero_config.n_experts,
               "n_active_experts": hero_config.n_active_experts,
               "expert_mlp_dim": hero_config.expert_mlp_dim,
               "aux_loss_alpha": hero_config.aux_loss_alpha,
               "z_loss_alpha": hero_config.z_loss_alpha,
               "attn_impl": hero_config.attn_impl,
               "qk_norm": hero_config.qk_norm,
               "learning_rate": hero_config.learning_rate,
               "non_embed_params": hero_non_embed,
               "target_tokens": target_tokens, "steps": HERO_STEPS,
           })
wandb.define_metric("train/loss", step_metric="step")
wandb.define_metric("train/tok_per_sec", step_metric="step")
wandb.define_metric("train/mfu_pct", step_metric="step")
wandb.define_metric("val/loss", step_metric="step")

smooth_loss = 0.0
debiased_loss = 0.0
best_val_loss = float('inf')
total_training_time = 0.0
params = hero_params

if SAVE_CHECKPOINTS:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f'\n=== Hero Run: {HERO_STEPS:,} steps ===\n')

try:
    for step in range(HERO_STEPS + 1):
        last_step = (step == HERO_STEPS)

        # --- Eval ---
        if step % HERO_EVAL_EVERY == 0 or last_step:
            val_loader = val_loader_fn()
            val_losses = []
            for _ in range(hero_config.eval_steps):
                vx, vy = next(val_loader)
                vx, vy = jnp.array(vx), jnp.array(vy)
                vl = eval_step(hero_config, params, vx, vy)
                val_losses.append(float(vl))
            avg_val_loss = sum(val_losses) / len(val_losses)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

            eval_log = {"step": step, "val/loss": avg_val_loss}
            if last_step:
                eval_log["train/loss"] = debiased_loss
            wandb.log(eval_log)
            print(f'step {step:06d}/{HERO_STEPS} | Val loss: {avg_val_loss:.4f} '
                  f'(best: {best_val_loss:.4f})')

        # --- Checkpoint ---
        if SAVE_CHECKPOINTS and step > 0 and step % 50_000 == 0:
            import pickle as pkl
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'params_step{step}.pkl')
            params_np = jax.tree.map(
                lambda x: np.array(x) if isinstance(x, jax.Array) else x, params)
            with open(ckpt_path, 'wb') as f:
                pkl.dump(params_np, f)
            print(f'  Checkpoint saved: {ckpt_path}')

        if last_step:
            break

        # --- Train step ---
        t0 = time.time()
        x_batch, y_batch = next(train_loader)
        loss, params, opt_state = hero_train_step(hero_config, params, opt_state,
                                                   x_batch, y_batch)
        loss.block_until_ready()
        dt = time.time() - t0

        if step > 20:
            total_training_time += dt

        loss_val = float(loss)
        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
        debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))

        if step % 1000 == 0:
            tok_per_sec = int(total_batch_size / dt) if dt > 0 else 0
            mfu_pct = step_flops / (PEAK_TFLOPS * 1e12 * dt) * 100 if dt > 0 else 0
            pct = 100 * step / HERO_STEPS
            eta = ''
            if step > 20 and total_training_time > 0:
                avg_dt = total_training_time / (step - 20)
                remaining = (HERO_STEPS - step) * avg_dt
                eta = f' | eta: {remaining/60:.0f}m'

            wandb.log({
                "step": step,
                "train/loss": debiased_loss,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu_pct": mfu_pct,
            })
            print(f'step {step:06d}/{HERO_STEPS} ({pct:.1f}%) | '
                  f'loss: {debiased_loss:.4f} | MFU: {mfu_pct:.1f}% | '
                  f'tok/s: {tok_per_sec:,}{eta}')

finally:
    train_loader.stop()

wandb.finish()
print(f'\nHero run complete. Best val loss: {best_val_loss:.4f}')
print(f'Total training time: {total_training_time/3600:.1f}h')

# --- Sample text ---
print('\n--- Samples ---')
for prompt in ['The capital of France is', 'In a distant galaxy, scientists discovered',
               'Machine learning is']:
    text = generate(hero_config, params, enc, prompt, max_new_tokens=100)
    print(f'Prompt: {prompt}\nOutput: {text}\n')

# --- Upload checkpoint to HF Hub ---
import json

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

params_np = jax.tree.map(
    lambda x: np.array(x) if isinstance(x, jax.Array) else x, params)
with open(os.path.join(CHECKPOINT_DIR, 'params.pkl'), 'wb') as f:
    pickle.dump(params_np, f)

config_dict = {k: v for k, v in hero_config.__dict__.items()
               if not k.startswith('_')}
config_dict['revision'] = REVISION
config_dict['best_val_loss'] = best_val_loss
config_dict['total_steps'] = HERO_STEPS
config_dict['total_training_time_hours'] = round(total_training_time / 3600, 2)
with open(os.path.join(CHECKPOINT_DIR, 'config.json'), 'w') as f:
    json.dump(config_dict, f, indent=2, default=str)

from huggingface_hub import HfApi
api = HfApi()
api.create_repo(HF_REPO_ID, repo_type='model', exist_ok=True)
ckpt_name = f'checkpoint_09_rev{REVISION}'
api.upload_folder(
    folder_path=CHECKPOINT_DIR,
    repo_id=HF_REPO_ID,
    path_in_repo=ckpt_name,
    commit_message=f'09 MoE hero run rev{REVISION}: '
                   f'val_loss={best_val_loss:.4f}, {HERO_STEPS} steps',
)
print(f'\nUploaded to https://huggingface.co/{HF_REPO_ID}/tree/main/{ckpt_name}')

# --- Disconnect runtime to stop billing ---
from google.colab import runtime
runtime.unassign()

# %% [markdown]
# ## Load from HF checkpoint & Sample

# %%
# === Load hero checkpoint from HuggingFace ===
import pickle
import jax
import jax.numpy as jnp
from huggingface_hub import hf_hub_download

HF_REPO_ID = "vorushin/tpuchat"
CHECKPOINT_NAME = "checkpoint_09_rev1"  # update to match your upload

# Download params and config
params_path = hf_hub_download(HF_REPO_ID, f"{CHECKPOINT_NAME}/params.pkl")
config_path = hf_hub_download(HF_REPO_ID, f"{CHECKPOINT_NAME}/config.json")

import json
with open(config_path) as f:
    config_dict = json.load(f)
print(f"Checkpoint: {CHECKPOINT_NAME}")
print(f"Val loss: {config_dict.get('best_val_loss', 'N/A')}")
print(f"Steps: {config_dict.get('total_steps', 'N/A')}")

# Reconstruct Config and load params
sample_config = Config(
    learning_rate=config_dict['learning_rate'],
    n_embd=config_dict['n_embd'],
    n_layer=config_dict['n_layer'],
    n_head=config_dict['n_head'],
    n_kv_head=config_dict['n_kv_head'],
    head_dim=config_dict['head_dim'],
    n_experts=config_dict['n_experts'],
    n_active_experts=config_dict['n_active_experts'],
    expert_mlp_dim=config_dict['expert_mlp_dim'],
)

with open(params_path, 'rb') as f:
    params_np = pickle.load(f)
sample_params = jax.tree.map(jnp.array, params_np)

print(f'\nParams loaded: {count_params(sample_params)/1e6:.1f}M')

# %%
# === Generate samples ===
prompts = [
    'The capital of France is',
    'In a distant galaxy, scientists discovered',
    'Machine learning is',
    'The most important invention of the 20th century',
    'Once upon a time, in a small village',
    'The theory of relativity states that',
]

print('--- Samples ---\n')
for prompt in prompts:
    text = generate(sample_config, sample_params, enc, prompt, max_new_tokens=100)
    print(f'Prompt: {prompt}')
    print(f'Output: {text}\n')
