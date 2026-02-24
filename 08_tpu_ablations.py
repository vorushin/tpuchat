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
#     gpuType: V6E1
#     machine_shape: hm
# ---

# %% [markdown]
# <a href="https://colab.research.google.com/github/vorushin/tpuchat/blob/master/08_tpu_ablations.ipynb?flush_caches=true" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 08 — TPU Ablation Lab (rev 38)
#
# Companion notebook for
# [LLM pretraining on TPU v6e with a $50 budget](https://vorushin.github.io/blog/llm-pretraining-tpu-v6e-50usd).
# Based on Karpathy's [nanochat](https://github.com/karpathy/nanochat) — ported
# to JAX for a single TPU v6e on Google Colab Pro+. Also works
# on TPU v5e (available on the free Colab plan, ~3× slower).
#
# **Runtime type:** In Colab, go to *Runtime → Change runtime type* and select
# **TPU v6e** (or v5e).
#
# **Three modes:**
# 1. **Quick Training** (~300 steps) — XProf capture, MFU measurement
# 2. **Sweep** (wandb, ~13 min/run) — Bayesian LR search per architecture config
# 3. **Hero Run** (20 tok/param, ~1.9h) — Full training with eval + HuggingFace upload
#
# ### Architecture: D=1024, N=4, K=1, H=256, F=3072, L=8, B=64 (16×4), T=2048, V=32768
# | Metric | Value |
# |--------|-------|
# | Total params | 163.6M |
# | Non-embed params | 130.0M |
# | Tokens/step | 131,072 (64 × 2048) |
# | Step time (splash) | ~302ms (16 microbatches) |
# | Throughput | ~435k tok/s |
# | MFU | ~46.5% |
# | Hero run (20 tok/param) | ~2.6B tokens, ~19.8k steps, ~1.7h |
#
# ### Setup: Colab Secrets
#
# This notebook reads API tokens from
# [Colab Secrets](https://colab.research.google.com/) (🔑 key icon in the left
# sidebar). Add two secrets before running:
#
# | Secret | Where to get it | Used by |
# |--------|----------------|---------|
# | `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access is enough) | Utilities — downloads the tokenizer from `vorushin/tpuchat` |
# | `WANDB_TOKEN` | [wandb.ai/authorize](https://wandb.ai/authorize) | Sweep and Hero Run — logs metrics to Weights & Biases |

# %%
# !pip install -q "jax[tpu]" optax huggingface_hub tiktoken pyarrow requests wandb tensorboard tensorboard-plugin-profile plotly

# %% [markdown]
# ## Prerequisites
#
# Loads the data and the tokenizer (trained in
# [01_tokenizer.ipynb](https://github.com/vorushin/tpuchat/blob/master/01_tokenizer.ipynb)).
# You may need to change the number of data shards downloaded here if you train
# bigger models or train them for longer.

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

REVISION = 38

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
#   K=n_kv_heads, H=head_dim, F=d_ff, L=n_layers, V=vocab_size

def matmul_flops(M, N, K, batch=1):
    """FLOPs for [M,K] @ [K,N].  2*M*N*K per batch element."""
    return 2 * batch * M * N * K


def attention_flops(B, N, T, H):
    """FLOPs for QK^T + AV (full T×T, not causal-halved)."""
    return 2 * (2 * B * N * T * T * H)


def layer_flops(B, T, D, N, K, H, F, mlp_type='glu'):
    """MXU-relevant FLOPs for one transformer layer.

    Counts only matmul FLOPs (projections + attention core + MLP).
    mlp_type='glu': 3 MLP matmuls (gate + up + down).
    mlp_type='plain': 2 MLP matmuls (up + down).
    """
    tok = B * T
    q    = 2 * tok * D * N * H          # Q projection
    k    = 2 * tok * D * K * H          # K projection
    v    = 2 * tok * D * K * H          # V projection
    att  = attention_flops(B, N, T, H)  # core attention
    proj = 2 * tok * N * H * D          # output projection
    if mlp_type == 'glu':
        mlp = 3 * (2 * tok * D * F)     # gate + up + down
    else:
        mlp = 2 * (2 * tok * D * F)     # up + down
    return q + k + v + att + proj + mlp

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
# The core of the notebook: transformer architecture and training loop.
# Architecture: RoPE, RMSNorm, MQA (4 query heads, 1 KV head), QK-norm, logit
# softcap, SwiGLU MLP, AdamW optimizer. Tensor dimensions are aligned to the
# TPU's 256×256 MXU block size. The `Config` dataclass lets you experiment with
# model variants and tune hyperparameters — edit it as needed.

# %%

@jax.tree_util.register_static
@dataclass(kw_only=True, frozen=True)
class Config:
    # ── Ablation knobs ─────────────────────────────────────────
    attn_impl: str = 'splash'       # 'splash' | 'einsum'
    mlp_type: str = 'glu'           # 'glu' (SwiGLU, F=3072) | 'plain' (ReLU², F=4096)
    qk_norm: bool = True            # QK-norm on queries and keys

    # ── Architecture (D1024, 130M non-embed) ───────────────────
    n_embd: int = 1024
    n_layer: int = 8
    seq_len: int = 2048
    vocab_size: int = 32768
    n_head: int = 4
    n_kv_head: int = 1
    head_dim: int = 256
    mlp_dim: int = 3072             # 3072 for glu, 4096 for plain
    softcap: float = 15.0
    logit_dtype: str = 'bf16'       # 'bf16' or 'fp32' — bf16 is ~4% faster
    splash_block_size: int = 1024
    num_lm_head_chunks: int = 8
    batch_size: int = 64
    microbatch_size: int = 4    # gradient accumulation when < batch_size

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
      f'V={config.vocab_size}, N={config.n_head}, K={config.n_kv_head}, '
      f'H={config.head_dim}, F={config.mlp_dim}')
print(f'Ablations: attn_impl={config.attn_impl}, mlp_type={config.mlp_type}, '
      f'qk_norm={config.qk_norm}')
mb_info = (f', microbatch={config.microbatch_size}, accum={config.num_microbatches}x'
           if config.num_microbatches > 1 else '')
print(f'Training: lr={config.learning_rate:.1e}, B={config.batch_size}{mb_info}')

# %%

def init_layer_params(config, seed=42):
    """Initialize params for one transformer layer."""
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

    # MLP — shape depends on mlp_type
    if config.mlp_type == 'glu':
        # SwiGLU: gate (D,F) + up (D,F) + down (F,D)
        layer.w_gate = jax.random.uniform(keys[3], (config.n_embd, config.mlp_dim),
                                           dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.w_up = jax.random.uniform(keys[4], (config.n_embd, config.mlp_dim),
                                         dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.w_down = jnp.zeros((config.mlp_dim, config.n_embd), dtype=jnp.bfloat16)
    else:
        # Plain (ReLU²): up (D,F) + down (F,D)
        layer.w_up = jax.random.uniform(keys[3], (config.n_embd, config.mlp_dim),
                                         dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.w_down = jnp.zeros((config.mlp_dim, config.n_embd), dtype=jnp.bfloat16)
    return layer


def single_layer_forward(config, layer, x, cos, sin, layer_idx=0):
    """Forward pass for one transformer layer."""
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

    with jax.named_scope(f'layer_{layer_idx}/mlp'):
        h2 = rms_norm(x)
        if config.mlp_type == 'glu':
            gate = jax.nn.silu(jnp.einsum('btd,dh->bth', h2, layer.w_gate))
            up = jnp.einsum('btd,dh->bth', h2, layer.w_up)
            mlp_out = jnp.einsum('bth,hd->btd', gate * up, layer.w_down)
        else:  # plain (ReLU²)
            mlp_out = jnp.einsum('btd,dh->bth', h2, layer.w_up)
            mlp_out = jax.nn.relu(mlp_out) ** 2
            mlp_out = jnp.einsum('bth,hd->btd', mlp_out, layer.w_down)

    x = x + mlp_out
    return x


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
    """Full forward: embed -> layers -> final_norm. Returns hidden (B,T,D)."""
    B, T = tokens.shape
    cos, sin = precompute_rope(T, config.head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    with jax.named_scope('embedding'):
        x = rms_norm(params.wte[tokens])
    for i in range(config.n_layer):
        x = single_layer_forward(config, params.layers[i], x, cos, sin, layer_idx=i)
    return rms_norm(x)


def _logit_dtype(config):
    return jnp.float32 if config.logit_dtype == 'fp32' else jnp.bfloat16


# Chunked LM head loss, reduces HMB usage by the last matmul (hidden_dim -> vocab_dim).
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
    """Create a JIT-compiled train step with the given optax optimizer.

    Supports gradient accumulation: when config.microbatch_size < config.batch_size,
    uses jax.lax.scan to process microbatches sequentially, accumulating gradients
    before a single optimizer update.
    """
    @jax.jit
    def train_step(config, params, opt_state, x, y, _opt=optimizer):
        num_mb = config.num_microbatches

        # Reshape full batch into microbatches: (B,T) → (num_mb, mb_size, T)
        x_micro = x.reshape(num_mb, config.microbatch_size, config.seq_len)
        y_micro = y.reshape(num_mb, config.microbatch_size, config.seq_len)

        def loss_fn(params, x_mb, y_mb):
            hidden = model_forward(config, params, x_mb)
            return chunked_lm_head_loss(hidden, params.lm_head, y_mb, config)

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
    """JIT-compiled eval: returns loss for a single batch."""
    hidden = model_forward(config, params, x)
    return chunked_lm_head_loss(hidden, params.lm_head, y, config)


@jax.jit
def predict_step(config, params, x):
    """JIT-compiled single step inference: returns logits."""
    hidden = model_forward(config, params, x)
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
# XProf for a few steps and shows the profiling results. Check that your loss
# goes down, that the MFU is reasonable (~50% is achievable on v6e), and use
# tok/s to estimate time of your longer runs. In the XProf trace, tightly packed
# "Device Execution" blocks = compute-bound (good); large gaps = input-bound
# (something needs fixing).

# %%

NUM_QUICK_STEPS = 300
EVAL_EVERY = 100
XPROF_START, XPROF_END = 15, 20
LOG_DIR = '/content/log_dir'

# Init model + optimizer
params = init_full_model(config, seed=config.param_seed)
total_p = count_params(params)
non_embed_p = count_non_embed_params(params)
print(f'Params: {total_p/1e6:.1f}M total, {non_embed_p/1e6:.1f}M non-embed')
print(f'Batch: {config.batch_size} x {config.seq_len} = '
      f'{config.batch_size * config.seq_len:,} tokens/step')

optimizer = make_optimizer(config, NUM_QUICK_STEPS)
opt_state = optimizer.init(params)
train_step = make_train_step(optimizer)

# Data
raw_train = tokenize_shards(train_shard_indices, config.batch_size, config.seq_len)
train_loader = PrefetchDataLoader(raw_train, capacity=4)
val_loader_fn = lambda: tokenize_shards(val_shard_indices, config.batch_size, config.seq_len)

# FLOP counting
fwd_flops = (config.n_layer * layer_flops(
    config.batch_size, config.seq_len, config.n_embd,
    config.n_head, config.n_kv_head, config.head_dim,
    config.mlp_dim, config.mlp_type)
    + matmul_flops(config.batch_size * config.seq_len,
                   config.vocab_size, config.n_embd))
step_flops = 3 * fwd_flops  # fwd + 2x bwd

smooth_loss = 0.0
mfu_t0 = None
mfu_tokens = 0
mfu_eval_time = 0.0
report_t0 = time.time()
report_tokens = 0
report_eval_time = 0.0

print(f'\n=== Quick Training: {NUM_QUICK_STEPS} steps ===\n')

for step in range(NUM_QUICK_STEPS + 1):
    last_step = (step == NUM_QUICK_STEPS)

    # --- Eval ---
    if step % EVAL_EVERY == 0 or last_step:
        eval_t0 = time.time()
        val_loader = val_loader_fn()
        val_losses = []
        for _ in range(config.eval_steps):
            vx, vy = next(val_loader)
            vx, vy = jnp.array(vx), jnp.array(vy)
            vl = eval_step(config, params, vx, vy)
            val_losses.append(float(vl))
        avg_val_loss = sum(val_losses) / len(val_losses)
        eval_dt = time.time() - eval_t0
        if mfu_t0 is not None:
            mfu_eval_time += eval_dt
        report_eval_time += eval_dt
        print(f'step {step:05d} | Val loss: {avg_val_loss:.4f}')

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

    t0 = time.time()
    loss, params, opt_state = train_step(config, params, opt_state,
                                          x_batch, y_batch)
    loss.block_until_ready()
    dt = time.time() - t0

    if step > XPROF_END:
        if mfu_t0 is None:
            mfu_t0 = time.time()
        mfu_tokens += config.batch_size * config.seq_len
    report_tokens += config.batch_size * config.seq_len

    loss_val = float(loss)
    ema_beta = 0.9
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
    debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))

    if step % 50 == 0:
        report_wall = time.time() - report_t0 - report_eval_time
        tok_per_sec = int(report_tokens / report_wall) if report_wall > 0 else 0
        print(f'step {step:05d}/{NUM_QUICK_STEPS} | loss: {debiased_loss:.4f} '
              f'| tok/s: {tok_per_sec:,}')
        report_t0 = time.time()
        report_tokens = 0
        report_eval_time = 0.0

train_loader.stop()

# --- MFU report (wall clock excl. eval, steps 21-300) ---
if mfu_t0 is not None:
    mfu_wall = time.time() - mfu_t0 - mfu_eval_time
    tok_per_s = int(mfu_tokens / mfu_wall)
    flops_per_tok = step_flops / (config.batch_size * config.seq_len)
    mfu_pct = (tok_per_s * flops_per_tok) / (PEAK_TFLOPS * 1e12) * 100
    ideal_tok_s = PEAK_TFLOPS * 1e12 / flops_per_tok
    print(f'\nMFU: {mfu_pct:.1f}% | tok/s: {tok_per_s:,} | '
          f'ideal tok/s (100% MFU): {int(ideal_tok_s):,}')

# --- Sample text ---
print('\n--- Samples ---')
for prompt in ['The capital of France is', 'Machine learning is']:
    text = generate(config, params, enc, prompt, max_new_tokens=64)
    print(f'Prompt: {prompt}\nOutput: {text}\n')

# %%
# === View Profiling Results ===
# Run this cell to load TensorBoard and view the trace captured in steps 15-20.
# - If you see large gaps between "Device Execution", you are INPUT BOUND.
# - If "Device Execution" blocks are packed tightly, you are COMPUTE BOUND (good!).

%load_ext tensorboard
%tensorboard --logdir /content/log_dir

# %% [markdown]
# ## Sweep (wandb)
#
# When everything looks good with your new transformer variant — run a quick
# (1–2 hour) hyperparameter search. Uses Bayesian optimization to search
# learning rate in log-uniform [5e-5, 1e-3] by default; feel free to add more
# parameters to `sweep_config`. At the very least, find a good learning rate —
# that's the most important hyperparameter to tune. You can run multiple copies
# of this Colab in parallel: "Save a copy in Drive" and set `SWEEP_ID` to the
# wandb sweep ID printed in the output of your first Colab.

# %%
# === wandb LR sweep ===
# Workflow: set attn_impl, mlp_type, qk_norm in Config above,
# re-run Model cells, then run this cell to sweep LR for that config.
import wandb
from google.colab import userdata

wandb.login(key=userdata.get("WANDB_TOKEN"))

sweep_config = {
    "name": f"ablation-{config.mlp_type}-{config.attn_impl}-qknorm{config.qk_norm}",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values",
                          "min": 5e-5, "max": 1e-3},
    },
}

SWEEP_PROJECT = "tpuchat-ablations"
SWEEP_ID = None            # set to existing sweep ID to continue, e.g. 'abc123'
SWEEP_STEPS = 2_500        # ~13 min at ~300ms/step (same ~328M token budget)
SWEEP_EVAL_EVERY = 250


def sweep_train_fn():
    """Single training run within a wandb sweep."""
    run = wandb.init()
    lr = wandb.config.learning_rate

    cfg = Config(learning_rate=lr)
    print(f'Sweep run: lr={lr:.2e}, attn={cfg.attn_impl}, '
          f'mlp={cfg.mlp_type}, qk_norm={cfg.qk_norm}')

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

    except Exception as e:
        import traceback
        print(f"\nSweep run crashed at step {step}: {e}")
        print(traceback.format_exc())
        raise
    finally:
        train_loader.stop()
        try:
            wandb.finish()
        except Exception:
            pass

    print(f'Run complete. Best val loss: {best_val_loss:.4f}')


sweep_id = SWEEP_ID or wandb.sweep(sweep_config, project=SWEEP_PROJECT)
print(f"{'Continuing' if SWEEP_ID else 'New'} sweep: {sweep_id}")
wandb.agent(sweep_id, function=sweep_train_fn, count=5, project=SWEEP_PROJECT)

# --- Disconnect runtime to stop billing ---
from google.colab import runtime
runtime.unassign()

# %%
# === Plot sweep results: LR vs val_loss ===
import wandb
import plotly.graph_objects as go

from google.colab import userdata

wandb.login(key=userdata.get("WANDB_TOKEN"))

SWEEP_PROJECT_AND_ID = "tpuchat-ablations/q2c790hc"

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
# You found the good hyperparameters — update them in
# `hero_config = Config(learning_rate=...)` and train your model for 20 tokens
# per parameter (Chinchilla-optimal ratio). Training config: AdamW, warmup 2%,
# warmdown 50% (cosine to 0), weight_decay=0.1. The model is exported to
# HuggingFace and the Colab kernel is terminated afterwards (to avoid paying for
# an idle TPU).

# %%
# === Hero run: 20 tok/param, ~19.8k steps ===
import wandb
from google.colab import userdata

RESUME_CHECKPOINT = ''  # set to e.g. 'checkpoint_08_rev12' to resume from HF
CHECKPOINT_EVERY = 5_000
CHECKPOINT_DIR = '/content/checkpoints'


def save_checkpoint_to_hf(params, opt_state, config, step, best_val_loss,
                          total_training_time, revision, hf_repo_id, notebook_id):
    """Save training state to HF Hub. Survives Colab preemption."""
    import json
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    params_np = jax.tree.map(
        lambda x: np.array(x) if isinstance(x, jax.Array) else x, params)
    with open(os.path.join(CHECKPOINT_DIR, 'params.pkl'), 'wb') as f:
        pickle.dump(params_np, f)
    opt_state_np = jax.tree.map(
        lambda x: np.array(x) if isinstance(x, jax.Array) else x, opt_state)
    with open(os.path.join(CHECKPOINT_DIR, 'opt_state.pkl'), 'wb') as f:
        pickle.dump(opt_state_np, f)
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    config_dict['revision'] = revision
    config_dict['best_val_loss'] = best_val_loss
    config_dict['checkpoint_step'] = step
    config_dict['total_training_time_hours'] = round(total_training_time / 3600, 2)
    with open(os.path.join(CHECKPOINT_DIR, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(hf_repo_id, repo_type='model', exist_ok=True)
    ckpt_name = f'checkpoint_{notebook_id}_rev{revision}'
    api.upload_folder(
        folder_path=CHECKPOINT_DIR, repo_id=hf_repo_id, path_in_repo=ckpt_name,
        commit_message=f'{notebook_id} step {step}: val_loss={best_val_loss:.4f}',
    )
    print(f'  Checkpoint uploaded to HF: {ckpt_name} (step {step})')


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
print(f'Estimated time: {HERO_STEPS * 0.302 / 3600:.1f} hours (at ~302ms/step, 16 microbatches)')

# Init optimizer
hero_opt = make_optimizer(hero_config, HERO_STEPS)
hero_train_step = make_train_step(hero_opt)

# Resume from HF checkpoint or start fresh
start_step = 0
if RESUME_CHECKPOINT:
    from huggingface_hub import hf_hub_download
    import json as _json
    _p = hf_hub_download(HF_REPO_ID, f"{RESUME_CHECKPOINT}/params.pkl")
    _o = hf_hub_download(HF_REPO_ID, f"{RESUME_CHECKPOINT}/opt_state.pkl")
    _c = hf_hub_download(HF_REPO_ID, f"{RESUME_CHECKPOINT}/config.json")
    with open(_p, 'rb') as f:
        params = jax.tree.map(jnp.array, pickle.load(f))
    with open(_o, 'rb') as f:
        opt_state = jax.tree.map(jnp.array, pickle.load(f))
    with open(_c) as f:
        _meta = _json.load(f)
    start_step = _meta['checkpoint_step']
    best_val_loss = _meta.get('best_val_loss', float('inf'))
    print(f'Resumed from {RESUME_CHECKPOINT} at step {start_step}, '
          f'best val loss: {best_val_loss:.4f}')
else:
    params = hero_params
    opt_state = hero_opt.init(hero_params)

# Data
raw_train = tokenize_shards(train_shard_indices, hero_config.batch_size, hero_config.seq_len)
train_loader = PrefetchDataLoader(raw_train, capacity=4)
val_loader_fn = lambda: tokenize_shards(val_shard_indices, hero_config.batch_size, hero_config.seq_len)

# FLOP counting for MFU
fwd_flops = (hero_config.n_layer * layer_flops(
    hero_config.batch_size, hero_config.seq_len, hero_config.n_embd,
    hero_config.n_head, hero_config.n_kv_head, hero_config.head_dim,
    hero_config.mlp_dim, hero_config.mlp_type)
    + matmul_flops(hero_config.batch_size * hero_config.seq_len,
                   hero_config.vocab_size, hero_config.n_embd))
step_flops = 3 * fwd_flops

# wandb
wandb.login(key=userdata.get("WANDB_TOKEN"))
wandb.init(project="tpuchat-ablations",
           name=f"hero-{hero_config.mlp_type}-{hero_config.attn_impl}-qknorm{hero_config.qk_norm}",
           config={
               "mlp_type": hero_config.mlp_type, "attn_impl": hero_config.attn_impl, "qk_norm": hero_config.qk_norm,
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
if not RESUME_CHECKPOINT:
    best_val_loss = float('inf')
total_training_time = 0.0

# SIGTERM handler — Colab sends SIGTERM before preempting TPU runtimes
import signal
_sigterm_received = False
def _sigterm_handler(signum, frame):
    global _sigterm_received
    _sigterm_received = True
    print(f"\nSIGTERM received — Colab is preempting this runtime")
signal.signal(signal.SIGTERM, _sigterm_handler)

print(f'\n=== Hero Run: {HERO_STEPS:,} steps (starting from {start_step}) ===\n')

try:
    for step in range(start_step, HERO_STEPS + 1):
        last_step = (step == HERO_STEPS)

        # --- SIGTERM check (Colab preemption) ---
        if _sigterm_received:
            print("Saving checkpoint before SIGTERM exit...")
            try:
                save_checkpoint_to_hf(params, opt_state, hero_config, step,
                                      best_val_loss, total_training_time,
                                      REVISION, HF_REPO_ID, '08')
            except Exception as ckpt_err:
                print(f"Checkpoint save failed: {ckpt_err}")
            break

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

        # --- Checkpoint to HF Hub ---
        if step > 0 and step % CHECKPOINT_EVERY == 0:
            save_checkpoint_to_hf(params, opt_state, hero_config, step,
                                  best_val_loss, total_training_time,
                                  REVISION, HF_REPO_ID, '08')

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

except Exception as e:
    import traceback
    print(f"\nHERO RUN CRASHED at step {step}: {e}")
    print(traceback.format_exc())
    try:
        wandb.log({"step": step, "error": type(e).__name__})
        wandb.alert(title=f"Hero run crashed at step {step}",
                    text=f"{type(e).__name__}: {e}\n\nLast loss: {debiased_loss:.4f}",
                    level=wandb.AlertLevel.ERROR)
    except Exception:
        pass
    raise
finally:
    train_loader.stop()
    try:
        wandb.finish()
    except Exception:
        pass

print(f'\nHero run complete. Best val loss: {best_val_loss:.4f}')
print(f'Total training time: {total_training_time/3600:.1f}h')

# --- Sample text ---
print('\n--- Samples ---')
for prompt in ['The capital of France is', 'In a distant galaxy, scientists discovered',
               'Machine learning is']:
    text = generate(hero_config, params, enc, prompt, max_new_tokens=100)
    print(f'Prompt: {prompt}\nOutput: {text}\n')

# --- Final checkpoint upload to HF Hub ---
save_checkpoint_to_hf(params, opt_state, hero_config, HERO_STEPS,
                      best_val_loss, total_training_time,
                      REVISION, HF_REPO_ID, '08')

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
CHECKPOINT_NAME = "checkpoint_08_rev4"  # update to match your upload

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
    mlp_dim=config_dict['mlp_dim'],
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
