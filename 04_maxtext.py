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
# <a href="https://colab.research.google.com/github/vorushin/tpuchat/blob/master/04_maxtext.ipynb?flush_caches=true" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 04 — MaxText-inspired GPT (~370M params)
#
# Self-contained notebook that:
# 1. Downloads tokenizer from HuggingFace Hub
# 2. Downloads ~50 data shards from FineWeb-Edu-100B-Shuffle
# 3. Defines GPT model in raw JAX (no Flax) with MaxText/Gemma3 best practices
# 4. Trains the model on a single TPU v6e-1
# 5. Saves checkpoint to HuggingFace Hub
#
# Changes from 02_train.py:
# - **SwiGLU MLP** (replaces ReLU^2) — gate/up/down projections
# - **Explicit dimensions** — n_embd=1024, n_layer=24, mlp_dim=3072 (all 256-aligned for MXU)
# - **Standard pre-norm residual** — no x0 connection (matches Gemma3/Llama3)
# - **LR 7e-4** — from wandb sweep results
# - **~370M params** — smaller than 02's ~1.5B for faster iteration

# %%
# Install dependencies (uncomment for Colab)
# !pip install -q "jax[tpu]" optax huggingface_hub tiktoken pyarrow requests torch tensorboard tensorboard-plugin-profile

# %%
# === Config ===
import functools as ft
import itertools as it
import time
import os
import math
import queue
import threading
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tiktoken

@jax.tree_util.register_static
@dataclass(kw_only=True, frozen=True)
class Config:
    # Data
    num_shards: int = 50
    hf_repo_id: str = 'vorushin/tpuchat'

    # Model architecture — all matrix dims are multiples of 256 for v6e MXU
    n_head: int = 4
    n_kv_head: int = 2  # GQA ratio 2:1 (matches Gemma3-4B)
    head_dim: int = 256  # exact MXU tile size
    n_embd: int = 1024   # = 4 x 256
    n_layer: int = 24
    mlp_dim: int = 3072  # = 12 x 256, 3x expansion for SwiGLU
    vocab_size: int = 32768
    seq_len: int = 2048
    window_pattern: str = 'LLLL'
    softcap: float = 15.0
    attn_impl: str = 'splash'  # 'einsum', 'jax', 'splash', 'pallas'
    splash_block_size: int = 1024  # block size for splash kernel
    num_lm_head_chunks: int = 8    # chunk sequence for memory-efficient lm_head loss

    # Training
    num_iterations: int = 1000  # set to -1 for auto from target_param_data_ratio
    target_param_data_ratio: float = 10.5
    device_batch_size: int = 8
    total_batch_size: int = -1  # -1 = auto (device_batch_size * seq_len)
    max_chars_per_doc: int = 10_000

    # Optimizer (AdamW)
    learning_rate: float = 7e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    warmup_ratio: float = 0.02
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0

    # Eval / Logging
    eval_every: int = 100
    eval_steps: int = 10
    log_every: int = 10
    save_every: int = -1  # -1 = only at end
    sample_every: int = 250

    # Seed
    param_seed: int = 42

config = Config()
assert config.vocab_size % 256 == 0, f"vocab_size must be divisible by 256, got {config.vocab_size}"
assert config.n_head % config.n_kv_head == 0, \
    f'n_head ({config.n_head}) must be divisible by n_kv_head ({config.n_kv_head})'
assert config.n_embd == config.n_head * config.head_dim, \
    f'n_embd ({config.n_embd}) must equal n_head * head_dim ({config.n_head * config.head_dim})'
assert config.n_embd % 256 == 0, f'n_embd ({config.n_embd}) must be 256-aligned'
assert config.mlp_dim % 256 == 0, f'mlp_dim ({config.mlp_dim}) must be 256-aligned'
assert config.head_dim % 256 == 0, f'head_dim ({config.head_dim}) must be 256-aligned'
assert (config.device_batch_size * config.seq_len) % config.num_lm_head_chunks == 0, \
    f'device_batch_size * seq_len ({config.device_batch_size * config.seq_len}) must be divisible by num_lm_head_chunks ({config.num_lm_head_chunks})'
print(f'Model: n_layer={config.n_layer}, n_embd={config.n_embd}, n_head={config.n_head}, '
      f'n_kv_head={config.n_kv_head}, head_dim={config.head_dim}, '
      f'mlp_dim={config.mlp_dim}, vocab={config.vocab_size}')

# %%
# === HuggingFace Hub login + download tokenizer ===
from huggingface_hub import login, HfApi, hf_hub_download
from google.colab import userdata
login(token=userdata.get("HF_TOKEN"))

# Download tokenizer files from HF Hub
import pickle
TOKENIZER_DIR = '/content/tokenizer'
os.makedirs(TOKENIZER_DIR, exist_ok=True)

tok_pkl_path = hf_hub_download(
    repo_id=config.hf_repo_id,
    filename='tokenizer/tokenizer.pkl',
    local_dir=TOKENIZER_DIR,
)
token_bytes_path = hf_hub_download(
    repo_id=config.hf_repo_id,
    filename='tokenizer/token_bytes.pt',
    local_dir=TOKENIZER_DIR,
)
print(f'Downloaded tokenizer to {TOKENIZER_DIR}')

# Load tokenizer (tiktoken encoding object)
import tiktoken
with open(os.path.join(TOKENIZER_DIR, 'tokenizer', 'tokenizer.pkl'), 'rb') as f:
    enc = pickle.load(f)
print(f'Loaded tokenizer: vocab_size={enc.n_vocab}')

# Load token_bytes for BPB evaluation
import torch
with open(os.path.join(TOKENIZER_DIR, 'tokenizer', 'token_bytes.pt'), 'rb') as f:
    token_bytes_pt = torch.load(f, map_location='cpu')
token_bytes_np = token_bytes_pt.numpy().astype(np.int32)
print(f'Loaded token_bytes: shape={token_bytes_np.shape}')

# %%
# === Download data shards ===
import requests
from multiprocessing import Pool

BASE_URL = 'https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main'
DATA_DIR = '/content/base_data'
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

# Download training shards (0..num_shards-1) + a few val shards
NUM_TRAIN_SHARDS = config.num_shards
NUM_VAL_SHARDS = 2
total_shards = NUM_TRAIN_SHARDS + NUM_VAL_SHARDS

t0 = time.time()
with Pool(8) as pool:
    results = pool.map(download_shard, range(total_shards))
print(f'\nDownloaded {sum(results)}/{total_shards} shards in {time.time()-t0:.1f}s')

# %%
# === Data pipeline: tokenize parquet shards into (x, y) batches ===
import pyarrow.parquet as pq

def tokenize_shards(shard_indices, batch_size, seq_len):
    """Yield (x, y) batches by tokenizing parquet shards on the fly.

    Uses BOS-aligned packing: each document starts with BOS, documents are
    concatenated into a token buffer, and batches are sliced from the buffer.
    """
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
                    if len(doc) > config.max_chars_per_doc:
                        doc = doc[:config.max_chars_per_doc]
                    tokens = [bos_id] + enc.encode_ordinary(doc)
                    buf.extend(tokens)

                    # Yield batches whenever we have enough tokens
                    tokens_per_batch = batch_size * (seq_len + 1)
                    while len(buf) >= tokens_per_batch:
                        batch_tokens = np.array(buf[:tokens_per_batch], dtype=np.int32)
                        batch_tokens = batch_tokens.reshape(batch_size, seq_len + 1)
                        x = batch_tokens[:, :-1]  # input
                        y = batch_tokens[:, 1:]   # target
                        buf = buf[tokens_per_batch:]
                        yield x, y

train_shard_indices = list(range(NUM_TRAIN_SHARDS))
val_shard_indices = list(range(NUM_TRAIN_SHARDS, NUM_TRAIN_SHARDS + NUM_VAL_SHARDS))

print(f'Train shards: {len(train_shard_indices)}, Val shards: {len(val_shard_indices)}')
print(f'Batch: {config.device_batch_size} x {config.seq_len} = {config.device_batch_size * config.seq_len:,} tokens/step')

# %%
# === dot_dict: JAX-compatible mutable dictionary (from training cookbook) ===

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
# === Model: GPT in raw JAX ===

def rms_norm(x):
    """RMSNorm with no learnable parameters."""
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def precompute_rope(seq_len, head_dim, base=10000):
    """Precompute rotary embedding cos/sin tables."""
    channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)  # (seq_len, head_dim/2)
    cos = jnp.cos(freqs).astype(jnp.bfloat16)
    sin = jnp.sin(freqs).astype(jnp.bfloat16)
    return cos, sin  # (seq_len, head_dim/2)


def apply_rope(x, cos, sin):
    """Apply rotary embeddings. x: (B, H, T, D), cos/sin: (1, 1, T, D/2)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1)




def init_param_state(config: Config) -> dot_dict:
    """Initialize all model parameters."""
    key = jax.random.key(config.param_seed)
    n_embd = config.n_embd
    n_head = config.n_head
    n_kv_head = config.n_kv_head
    head_dim = config.head_dim
    n_layer = config.n_layer
    mlp_dim = config.mlp_dim
    vocab_size = config.vocab_size
    def split_key():
        nonlocal key
        key, subkey = jax.random.split(key)
        return subkey

    # Uniform init bound (matches nanochat: sqrt(3) * std = sqrt(3) / sqrt(n_embd))
    s = (3.0 ** 0.5) * (n_embd ** -0.5)

    params = dot_dict()

    # Token embedding: normal(0, 1)
    params.wte = jax.random.normal(split_key(), (vocab_size, n_embd), dtype=jnp.bfloat16)

    # LM head: normal(0, 0.001)
    params.lm_head = jax.random.normal(split_key(), (n_embd, vocab_size), dtype=jnp.bfloat16) * 0.001

    # Precompute RoPE
    params.rope_cos, params.rope_sin = precompute_rope(config.seq_len, head_dim)

    # Layers
    params.layers = dot_dict()
    for i in range(n_layer):
        layer = dot_dict()
        # Attention projections: uniform(-s, s), proj=zeros
        # Weights shaped with head dim for reshape-free einsums
        layer.c_q = jax.random.uniform(split_key(), (n_embd, n_head, head_dim),
                                        dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.c_k = jax.random.uniform(split_key(), (n_embd, n_kv_head, head_dim),
                                        dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.c_v = jax.random.uniform(split_key(), (n_embd, n_kv_head, head_dim),
                                        dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.c_proj = jnp.zeros((n_head, head_dim, n_embd), dtype=jnp.bfloat16)
        # SwiGLU MLP: gate + up uniform(-s, s), down=zeros
        layer.w_gate = jax.random.uniform(split_key(), (n_embd, mlp_dim),
                                           dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.w_up = jax.random.uniform(split_key(), (n_embd, mlp_dim),
                                         dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.w_down = jnp.zeros((mlp_dim, n_embd), dtype=jnp.bfloat16)

        params.layers[i] = layer

    return params


def compute_window_sizes(config: Config):
    """Compute per-layer sliding window sizes."""
    pattern = config.window_pattern.upper()
    long_w = config.seq_len
    short_w = long_w // 2
    char_to_w = {'L': long_w, 'S': short_w}
    sizes = []
    for i in range(config.n_layer):
        c = pattern[i % len(pattern)]
        sizes.append(char_to_w[c])
    sizes[-1] = long_w  # last layer always full
    return sizes


def _expand_kv(k, v, n_head, n_kv_head):
    """Repeat KV heads to match Q head count for non-splash backends."""
    if n_kv_head == n_head:
        return k, v
    ratio = n_head // n_kv_head
    return jnp.repeat(k, ratio, axis=1), jnp.repeat(v, ratio, axis=1)


def model_apply(config: Config, params: dot_dict, tokens: jax.Array) -> jax.Array:
    """Forward pass: tokens (B, T) -> hidden (B, T, n_embd) after final norm."""
    B, T = tokens.shape
    n_head = config.n_head
    n_kv_head = config.n_kv_head
    head_dim = config.head_dim
    n_layer = config.n_layer
    window_sizes = compute_window_sizes(config)

    # RoPE: (T, head_dim/2) -> (1, 1, T, head_dim/2) for (B, H, T, D) broadcasting
    cos = params.rope_cos[:T][None, None, :, :]  # (1, 1, T, D/2)
    sin = params.rope_sin[:T][None, None, :, :]

    # Token embedding + norm
    with jax.named_scope('embedding'):
        x = params.wte[tokens]  # (B, T, n_embd)
        x = rms_norm(x)

    for i in range(n_layer):
        layer = params.layers[i]

        # Pre-norm
        h = rms_norm(x)

        with jax.named_scope(f'layer_{i}/attention'):
            # === Attention ===
            # Native layout for splash/pallas is (B, H, T, D)
            # We want to avoid transposes, so let's produce q/k/v in (B, H, T, D) directly
            q = jnp.einsum('btd,dhk->bhtk', h, layer.c_q)  # (B, H, T, D)
            k = jnp.einsum('btd,dhk->bhtk', h, layer.c_k)  # (B, KV, T, D)
            v = jnp.einsum('btd,dhk->bhtk', h, layer.c_v)  # (B, KV, T, D)

            # Apply RoPE (T-dim aligned with (B, H, T, D))
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            # QK-norm
            q = rms_norm(q)
            k = rms_norm(k)

            # Attention — dispatch to selected implementation
            # Splash handles GQA natively (no KV repeat needed).
            # Other backends need explicit KV head expansion.
            w = window_sizes[i]

            if config.attn_impl == 'einsum':
                # Manual einsum attention (supports sliding window)
                # GQA: expand KV heads to match Q heads
                k_exp, v_exp = _expand_kv(k, v, n_head, n_kv_head)
                scale = head_dim ** -0.5
                scores = jnp.einsum('bhtd,bhsd->bhts', q, k_exp) * scale
                rows = jnp.arange(T)[:, None]
                cols = jnp.arange(T)[None, :]
                if w < T:
                    mask = (cols <= rows) & (cols >= rows - w + 1)
                else:
                    mask = cols <= rows
                scores = jnp.where(mask[None, None, :, :], scores,
                                   jnp.finfo(scores.dtype).min)
                attn_weights = jax.nn.softmax(scores, axis=-1)
                attn_out = jnp.einsum('bhts,bhsd->bhtd', attn_weights, v_exp)

            elif config.attn_impl == 'jax':
                # jax.nn.dot_product_attention expects (B, H, T, D)
                # GQA: expand KV heads to match Q heads
                k_exp, v_exp = _expand_kv(k, v, n_head, n_kv_head)
                if w < T:
                    rows = jnp.arange(T)[:, None]
                    cols = jnp.arange(T)[None, :]
                    mask = (cols <= rows) & (cols >= rows - w + 1)
                    # mask needs to be broadcastable to (B, H, T, S) -> (1, 1, T, S)
                    attn_out = jax.nn.dot_product_attention(
                        q, k_exp, v_exp, mask=mask[None, None, :, :], implementation='xla')
                else:
                    attn_out = jax.nn.dot_product_attention(
                        q, k_exp, v_exp, is_causal=True, implementation='xla')

            elif config.attn_impl == 'splash':
                # Splash Attention — Pallas kernel
                # Handles GQA natively: pass K/V with n_kv_head heads,
                # kernel reshapes Q and uses index mapping internally.
                from jax.experimental.pallas.ops.tpu.splash_attention import (
                    splash_attention_mask, splash_attention_kernel)

                smask = splash_attention_mask.CausalMask(shape=(T, T))
                if w < T:
                    smask = smask & splash_attention_mask.LocalMask(
                        shape=(T, T), window_size=(w, w), offset=0)

                # Mask uses Q head count — kernel maps Q heads to KV heads
                mh_mask = splash_attention_mask.MultiHeadMask(masks=[smask] * n_head)

                bs = config.splash_block_size
                block_sizes = splash_attention_kernel.BlockSizes(
                    block_q=bs, block_kv=bs,
                    block_q_dkv=bs, block_kv_dkv=bs,
                    block_q_dq=bs, block_kv_dq=bs,
                )

                kernel = splash_attention_kernel.make_splash_mha(
                    mask=mh_mask,
                    head_shards=1,
                    q_seq_shards=1,
                    block_sizes=block_sizes
                )

                # Q: (B, n_head, T, D), K/V: (B, n_kv_head, T, D)
                attn_out = jax.vmap(kernel)(q, k, v)

            elif config.attn_impl == 'pallas':
                # Pallas Flash Attention
                # GQA: expand KV heads to match Q heads
                k_exp, v_exp = _expand_kv(k, v, n_head, n_kv_head)
                from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
                attn_out = flash_attention(
                    q, k_exp, v_exp,
                    causal=True,
                    sm_scale=head_dim ** -0.5,
                )

            # Output projection
            # attn_out is (B, H, T, D) -> needs (B, T, E)
            attn_out = jnp.einsum('bhtd,hde->bte', attn_out, layer.c_proj)

        # Standard pre-norm residual (no x0 connection)
        x = x + attn_out

        with jax.named_scope(f'layer_{i}/mlp'):
            # === SwiGLU MLP ===
            h2 = rms_norm(x)
            gate = jax.nn.silu(jnp.einsum('btd,dh->bth', h2, layer.w_gate))
            up = jnp.einsum('btd,dh->bth', h2, layer.w_up)
            mlp_out = jnp.einsum('bth,hd->btd', gate * up, layer.w_down)
        x = x + mlp_out

    # Final norm only — lm_head applied separately
    x = rms_norm(x)

    return x  # (B, T, n_embd)


def apply_lm_head(config: Config, params: dot_dict, hidden: jax.Array) -> jax.Array:
    """Apply lm_head to get logits. Used for eval and generation."""
    with jax.named_scope('lm_head'):
        logits = jnp.einsum('btd,dv->btv', hidden, params.lm_head,
                            preferred_element_type=jnp.float32)
        logits = config.softcap * jnp.tanh(logits / config.softcap)
    return logits


def _logits_from_chunk(h_chunk: jax.Array, lm_head: jax.Array, config: Config) -> jax.Array:
    """Compute logits for a single (chunk_size, n_embd) slice."""
    logits = jnp.einsum('td,dv->tv', h_chunk, lm_head,
                        preferred_element_type=jnp.float32)
    return config.softcap * jnp.tanh(logits / config.softcap)


@ft.partial(jax.custom_vjp, nondiff_argnums=(3,))
def chunked_lm_head_loss(hidden: jax.Array, lm_head: jax.Array,
                         labels: jax.Array, config: Config) -> jax.Array:
    """Cross-entropy loss without materialising full (B, T, vocab) logits.

    Tiles the B×T dimension into config.num_lm_head_chunks slices, computes
    logits and loss per slice via jax.lax.scan, then sums.  The custom VJP
    recomputes logits during the backward pass (gradient checkpointing).
    """
    B, T, D = hidden.shape
    N = config.num_lm_head_chunks
    S = B * T // N  # tokens per chunk

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
    return loss, (hidden, lm_head, labels)   # residuals: small tensors, no logits


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
        d_h, d_w = vjp_fn(g / (B * T))   # scale by upstream g / total_tokens
        return d_lm_head_acc + d_w, d_h

    d_lm_head_init = jnp.zeros_like(lm_head)
    d_lm_head, d_hidden_chunks = jax.lax.scan(
        bwd_body, d_lm_head_init, (hidden_chunks, labels_chunks))

    return d_hidden_chunks.reshape(B, T, D), d_lm_head, jnp.zeros_like(labels)


chunked_lm_head_loss.defvjp(_chunked_loss_fwd, _chunked_loss_bwd)


# Test model initialization
params = init_param_state(config)
num_params = sum(p.size for p in jax.tree.leaves(params) if isinstance(p, jax.Array))
print(f'Model parameters: {num_params:,}')

# %%
# === Optimizer: AdamW with warmup + linear warmdown ===

def init_adam_state(param: jax.Array) -> dot_dict:
    """Initialize Adam optimizer state for a single parameter."""
    return dot_dict(
        mu=jnp.zeros_like(param),
        nu=jnp.zeros_like(param),
        count=jnp.array(0, dtype=jnp.int32),
    )


def adamw_step(config, lr_mult, param, grad, state):
    """AdamW update. Returns (new_param, new_state)."""
    new_count = state.count + 1
    new_mu = config.beta1 * state.mu + (1 - config.beta1) * grad
    new_nu = config.beta2 * state.nu + (1 - config.beta2) * grad ** 2

    mu_hat = new_mu / (1 - config.beta1 ** new_count)
    nu_hat = new_nu / (1 - config.beta2 ** new_count)

    lr = config.learning_rate * lr_mult
    update = mu_hat / (jnp.sqrt(nu_hat) + config.eps)

    # Weight decay for 2D+ params
    wd = jnp.where(param.ndim >= 2, config.weight_decay, 0.0)
    new_param = param - lr * (update + wd * param)

    new_state = dot_dict(mu=new_mu, nu=new_nu, count=new_count)
    return new_param, new_state


def get_lr_multiplier(step, num_iterations, config: Config):
    """Linear warmup, constant, linear warmdown schedule."""
    warmup_iters = int(config.warmup_ratio * num_iterations)
    warmdown_iters = int(config.warmdown_ratio * num_iterations)

    if step < warmup_iters:
        return (step + 1) / max(warmup_iters, 1)
    elif step <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - step) / max(warmdown_iters, 1)
        return progress * 1.0 + (1 - progress) * config.final_lr_frac

# %%
# === Training setup ===

# Count parameters for scaling laws
def count_matrix_params(params):
    """Count parameters that contribute to scaling."""
    attn = 0
    mlp = 0
    for i in range(config.n_layer):
        layer = params.layers[i]
        for name in ['c_q', 'c_k', 'c_v', 'c_proj']:
            attn += layer[name].size
        for name in ['w_gate', 'w_up', 'w_down']:
            mlp += layer[name].size
    emb = params.wte.size
    lm = params.lm_head.size
    return emb, attn, mlp, lm

emb_params, attn_params, mlp_params, lm_params = count_matrix_params(params)
non_emb_params = attn_params + mlp_params + lm_params
scaling_params = emb_params + non_emb_params
print(f'Params: {scaling_params/1e6:.1f}M total '
      f'(embed: {emb_params/1e6:.1f}M, attn: {attn_params/1e6:.1f}M, '
      f'mlp: {mlp_params/1e6:.1f}M, lm_head: {lm_params/1e6:.1f}M)')
target_tokens = int(config.target_param_data_ratio * scaling_params)

# Batch size
total_batch_size = config.total_batch_size
if total_batch_size == -1:
    total_batch_size = config.device_batch_size * config.seq_len
    print(f'Total batch size: {total_batch_size:,} tokens/step '
          f'({config.device_batch_size}×{config.seq_len})')

# Number of iterations
if config.num_iterations > 0:
    num_iterations = config.num_iterations
else:
    num_iterations = target_tokens // total_batch_size

print(f'Scaling params: {scaling_params:,}')
print(f'Target tokens: {target_tokens:,}')
print(f'Num iterations: {num_iterations:,}')
print(f'Estimated training tokens: {total_batch_size * num_iterations:,}')

# %%
# === JIT-compiled train step ===

# Filter out non-trainable params (rope_cos, rope_sin)
def split_trainable(params):
    """Split params into trainable and static (non-differentiable)."""
    trainable = dot_dict()
    static = dot_dict()
    for k, v in params.items():
        if k in ('rope_cos', 'rope_sin'):
            static[k] = v
        else:
            trainable[k] = v
    return trainable, static


def merge_params(trainable, static):
    """Merge trainable and static params back together."""
    merged = dot_dict()
    merged.update(trainable)
    merged.update(static)
    return merged


@jax.jit
def train_step(config: Config, params: dot_dict, opt_state: dot_dict,
               x: jax.Array, y: jax.Array, lr_mult: jax.Array):
    """Single training step: forward, backward, optimizer update."""
    trainable, static = split_trainable(params)

    def loss_fn(trainable_params):
        full_params = merge_params(trainable_params, static)
        hidden = model_apply(config, full_params, x)
        return chunked_lm_head_loss(hidden, full_params.lm_head, y, config)

    with jax.named_scope('forward_backward'):
        loss, grads = jax.value_and_grad(loss_fn)(trainable)

    # Apply AdamW update
    with jax.named_scope('optimizer'):
        is_opt_leaf = lambda x: isinstance(x, dot_dict) and 'mu' in x
        t_leaves, t_treedef = jax.tree.flatten(trainable)
        g_leaves, _ = jax.tree.flatten(grads)
        o_leaves, o_treedef = jax.tree.flatten(opt_state, is_leaf=is_opt_leaf)

        new_t_leaves, new_o_leaves = [], []
        for p, g, s in zip(t_leaves, g_leaves, o_leaves):
            new_p, new_s = adamw_step(config, lr_mult, p, g, s)
            new_t_leaves.append(new_p)
            new_o_leaves.append(new_s)

        new_trainable = t_treedef.unflatten(new_t_leaves)
        new_opt_state = o_treedef.unflatten(new_o_leaves)
        new_params = merge_params(new_trainable, static)

    return loss, new_params, new_opt_state


# Initialize optimizer state (only for trainable params)
trainable_params, static_params = split_trainable(params)
opt_state = jax.tree.map(init_adam_state, trainable_params)

print('Optimizer state initialized.')
print(f'Trainable param arrays: {len(jax.tree.leaves(trainable_params))}')

# %%
# === Eval + inference helpers ===

@jax.jit
def eval_step(config: Config, params: dot_dict, x: jax.Array, y: jax.Array):
    """JIT-compiled eval: returns loss for a single batch."""
    hidden = model_apply(config, params, x)
    logits = apply_lm_head(config, params, hidden)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))


@jax.jit
def predict_step(config: Config, params: dot_dict, x: jax.Array):
    """JIT-compiled single step inference."""
    hidden = model_apply(config, params, x)
    return apply_lm_head(config, params, hidden)


def generate(config, params, enc, prompt, max_new_tokens=100,
             temperature=0.8, top_k=200, top_p=0.95, seed=None):
    """Generate text from a prompt using temperature + top-k + top-p sampling.

    Args:
        config: model config
        params: model parameters
        enc: tiktoken encoding
        prompt: text prompt string
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature (0 = greedy, higher = more random)
        top_k: keep only top-k logits before sampling (0 = no filtering)
        top_p: nucleus sampling cumulative probability (1.0 = no filtering)
        seed: random seed (None = use current time)
    """
    if seed is None:
        seed = int(time.time() * 1000) % (2**31)
    key = jax.random.key(seed)

    prompt_ids = enc.encode_ordinary(prompt)
    bos_id = enc.encode_single_token('<|bos|>')
    ids = [bos_id] + prompt_ids

    for _ in range(max_new_tokens):
        # Truncate context to max seq_len if needed
        context = ids[-config.seq_len:]

        # Pad context to full seq_len to avoid JIT recompilation
        # (JIT cache depends on shape, so variable shape = recompile every step)
        pad_len = config.seq_len - len(context)
        padded_context = context + [0] * pad_len
        x = jnp.array([padded_context], dtype=jnp.int32)

        # We need the logits at the last valid token position
        logits = predict_step(config, params, x)
        logits.block_until_ready()  # Wait for computation (fix for Colab debugger exception)
        next_logits = logits[0, len(context) - 1, :]  # (vocab_size,)
        next_logits = next_logits.astype(jnp.float32)

        if temperature == 0:
            # Greedy
            next_id = int(jnp.argmax(next_logits))
        else:
            # Temperature scaling
            next_logits = next_logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = jax.lax.top_k(next_logits, top_k)
                # Set everything outside top-k to -inf
                mask_k = jnp.full_like(next_logits, -jnp.inf)
                next_logits = mask_k.at[top_k_indices].set(top_k_logits)

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                # Sort logits in descending order
                sorted_logits, sorted_indices = jax.lax.top_k(next_logits, len(next_logits))
                sorted_probs = jax.nn.softmax(sorted_logits)
                cumulative_probs = jnp.cumsum(sorted_probs)

                # Mask tokens after cumulative probability > top_p
                # We want to keep the first token that exceeds top_p, so we shift mask
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep at least one token
                sorted_indices_to_remove = jnp.concatenate([jnp.array([False]), sorted_indices_to_remove[:-1]])

                # Scatter -inf back to original indices
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits = next_logits.at[indices_to_remove].set(-jnp.inf)

            # Sample
            key, subkey = jax.random.split(key)
            next_id = int(jax.random.categorical(subkey, next_logits))

        ids.append(next_id)

    return enc.decode(ids)


# Quick test prompts (re-run this cell after training)
PROMPTS = [
    'The capital of France is',
    'In a distant galaxy, scientists discovered',
    'The quick brown fox',
    'Machine learning is',
]

# %%
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
                # Transfer to HBM in background thread — overlaps with compute
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
# === Training loop ===
use_random_data = False  # @param {type:"boolean"}

raw_train_loader = tokenize_shards(train_shard_indices, config.device_batch_size, config.seq_len)
train_loader = PrefetchDataLoader(raw_train_loader, capacity=4)
val_loader_fn = lambda: tokenize_shards(val_shard_indices, config.device_batch_size, config.seq_len)

# Pre-generate random data on HBM for profiling (excludes data loading time)
if use_random_data:
    rng_data = jax.random.key(0)
    random_x = jax.random.randint(rng_data, (config.device_batch_size, config.seq_len),
                                   0, config.vocab_size, dtype=jnp.int32)
    random_y = jax.random.randint(rng_data, (config.device_batch_size, config.seq_len),
                                   0, config.vocab_size, dtype=jnp.int32)
    # Force to device
    random_x.block_until_ready()
    random_y.block_until_ready()
    print(f'Using random data on HBM: x={random_x.shape}, y={random_y.shape}')

# History for plotting
train_loss_history = []  # (step, smoothed_loss)
val_loss_history = []    # (step, val_loss)

smooth_loss = 0.0
total_training_time = 0.0
best_val_loss = float('inf')

print(f'\n=== Starting training for {num_iterations:,} steps ===\n')

for step in range(num_iterations + 1):
    last_step = (step == num_iterations)

    # === Eval ===
    if config.eval_every > 0 and (last_step or step % config.eval_every == 0):
        val_loader = val_loader_fn()
        val_losses = []
        for ei in range(config.eval_steps):
            vx, vy = next(val_loader)
            vx, vy = jnp.array(vx), jnp.array(vy)
            vl = eval_step(config, params, vx, vy)
            val_losses.append(float(vl))
        avg_val_loss = sum(val_losses) / len(val_losses)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        val_loss_history.append((step, avg_val_loss))
        print(f'Step {step:05d} | Val loss: {avg_val_loss:.4f} (best: {best_val_loss:.4f})')

    # === Sample ===
    if config.sample_every > 0 and step > 0 and (last_step or step % config.sample_every == 0):
        print(f"\n--- Samples (step {step}) ---")
        for prompt in PROMPTS:
            sample_text = generate(config, params, enc, prompt, max_new_tokens=64, top_p=0.95)
            print(f"Prompt: {prompt}\nOutput: {sample_text}\n")
        print("----------------------------")

    if last_step:
        break

    # === Profiling ===
    if step == 15:
        jax.profiler.start_trace('log_dir')
        print("Profiling started...")
    if step == 20:
        jax.profiler.stop_trace()
        print("Profiling stopped. Trace saved to 'log_dir'.")

    # === Train step ===
    lr_mult = jnp.array(get_lr_multiplier(step, num_iterations, config), dtype=jnp.float32)
    t0 = time.time()

    if use_random_data:
        x_batch, y_batch = random_x, random_y
    else:
        x_batch, y_batch = next(train_loader)  # already on HBM from prefetch worker

    loss, params, opt_state = train_step(config, params, opt_state, x_batch, y_batch, lr_mult)
    loss.block_until_ready()
    dt = time.time() - t0

    if step > 10:
        total_training_time += dt

    loss_val = float(loss)
    ema_beta = 0.9
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
    debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))

    # Record every step
    train_loss_history.append((step, debiased_loss))

    if step % config.log_every == 0:
        tok_per_sec = int(total_batch_size / dt) if dt > 0 else 0
        pct = 100 * step / num_iterations
        eta = ''
        if step > 10 and total_training_time > 0:
            avg_dt = total_training_time / (step - 10)
            remaining = (num_iterations - step) * avg_dt
            eta = f' | eta: {remaining/60:.1f}m'
        print(f'step {step:05d}/{num_iterations:05d} ({pct:.1f}%) | loss: {debiased_loss:.4f} '
              f'| lr_mult: {float(lr_mult):.3f} | dt: {dt*1000:.0f}ms '
              f'| tok/s: {tok_per_sec:,}{eta}')

print(f'\nTraining complete. Total time: {total_training_time/60:.1f}m')
print(f'Best val loss: {best_val_loss:.4f}')

# %%
# === View Profiling Results ===
# Run this cell to load TensorBoard and view the trace captured in steps 15-20.
# - If you see large gaps between "Device Execution", you are INPUT BOUND.
# - If "Device Execution" blocks are packed tightly, you are COMPUTE BOUND (good!).

%load_ext tensorboard
%tensorboard --logdir log_dir

# %%
# === Plot training curves ===
# Re-run this cell anytime to see the latest curves
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(12, 5))

# Train loss
if train_loss_history:
    steps, losses = zip(*train_loss_history)
    ax.plot(steps, losses, label='Train loss (smoothed)', alpha=0.8, linewidth=1)

# Val loss
if val_loss_history:
    steps, losses = zip(*val_loss_history)
    ax.plot(steps, losses, 'ro-', label='Val loss', markersize=5)

ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title(f'Training curves — n_layer={config.n_layer}, n_embd={config.n_embd}, batch={total_batch_size:,} tok/step')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



# %%
# === Save checkpoint to HuggingFace Hub ===
save_checkpoint = False  # @param {type:"boolean"}

if save_checkpoint:
    import pickle
    import json

    CHECKPOINT_DIR = '/content/checkpoint'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Save params as pickle (convert JAX arrays to numpy)
    params_np = jax.tree.map(lambda x: np.array(x) if isinstance(x, jax.Array) else x,
                              params)
    with open(os.path.join(CHECKPOINT_DIR, 'params.pkl'), 'wb') as f:
        pickle.dump(params_np, f)
    print(f'Saved params to {CHECKPOINT_DIR}/params.pkl')

    # Save config
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    with open(os.path.join(CHECKPOINT_DIR, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

    # Upload to HF Hub
    api = HfApi()
    api.create_repo(config.hf_repo_id, repo_type='model', exist_ok=True)
    api.upload_folder(
        folder_path=CHECKPOINT_DIR,
        repo_id=config.hf_repo_id,
        path_in_repo=f'checkpoint_L{config.n_layer}',
        commit_message=f'Upload checkpoint (n_layer={config.n_layer}, {num_iterations} steps)',
    )
    print(f'\nUploaded to https://huggingface.co/{config.hf_repo_id}/tree/main/checkpoint_L{config.n_layer}')
else:
    print('Skipping checkpoint save. Set save_checkpoint=True to upload.')
