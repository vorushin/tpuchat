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
# <a href="https://colab.research.google.com/github/vorushin/tpuchat/blob/master/03_worker.ipynb?flush_caches=true" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 03 — Hyperparameter Sweep Worker (wandb)
#
# wandb-coordinated hyperparameter sweep for tpuchat.
#
# **Usage:**
# 1. Run cells 1-7 to set up environment and create a sweep (once)
# 2. Copy the `sweep_id` printed in cell 7
# 3. On each Colab Pro+ instance: run cells 1-6, paste `sweep_id` into cell 9, run cells 8-9

# %%
# Install dependencies (uncomment for Colab)
# !pip install -q "jax[tpu]" optax huggingface_hub tiktoken pyarrow requests torch wandb

# %%
# === Imports + Config ===
import functools as ft
import itertools as it
import time
import os
import math
import queue
import threading
import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tiktoken
import wandb

@jax.tree_util.register_static
@dataclass(kw_only=True, frozen=True)
class Config:
    # Data
    num_shards: int = 50
    hf_repo_id: str = 'vorushin/tpuchat'

    # Model architecture — n_head is the primary scaling knob
    n_head: int = 8
    n_kv_head: int = 2  # GQA: must divide n_head evenly
    aspect_ratio: int = 64
    head_dim: int = 256
    vocab_size: int = 32768
    seq_len: int = 2048
    window_pattern: str = 'LLLL'
    softcap: float = 15.0
    attn_impl: str = 'splash'  # 'einsum', 'jax', 'splash', 'pallas'
    splash_block_size: int = 1024  # block size for splash kernel

    # Training
    num_iterations: int = 1000
    target_param_data_ratio: float = 10.5
    device_batch_size: int = 8
    total_batch_size: int = -1  # -1 = auto (device_batch_size * seq_len)
    max_chars_per_doc: int = 10_000

    # Optimizer (AdamW)
    learning_rate: float = 3e-4
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

    # Derived
    @property
    def n_embd(self):
        return self.n_head * self.head_dim

    @property
    def depth(self):
        return self.n_embd // self.aspect_ratio

    @property
    def n_layer(self):
        return self.depth

# %%
# === wandb + HuggingFace login ===
from google.colab import userdata
wandb.login(key=userdata.get("WANDB_TOKEN"))

from huggingface_hub import login, HfApi, hf_hub_download
login(token=userdata.get("HF_TOKEN"))

# %%
# === Download tokenizer + data shards ===
import pickle

config_default = Config()
assert config_default.vocab_size % 256 == 0, f"vocab_size must be divisible by 256, got {config_default.vocab_size}"

TOKENIZER_DIR = '/content/tokenizer'
os.makedirs(TOKENIZER_DIR, exist_ok=True)

tok_pkl_path = hf_hub_download(
    repo_id=config_default.hf_repo_id,
    filename='tokenizer/tokenizer.pkl',
    local_dir=TOKENIZER_DIR,
)
token_bytes_path = hf_hub_download(
    repo_id=config_default.hf_repo_id,
    filename='tokenizer/token_bytes.pt',
    local_dir=TOKENIZER_DIR,
)
print(f'Downloaded tokenizer to {TOKENIZER_DIR}')

# Load tokenizer (tiktoken encoding object)
with open(os.path.join(TOKENIZER_DIR, 'tokenizer', 'tokenizer.pkl'), 'rb') as f:
    enc = pickle.load(f)
print(f'Loaded tokenizer: vocab_size={enc.n_vocab}')

# Download data shards
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

NUM_TRAIN_SHARDS = config_default.num_shards
NUM_VAL_SHARDS = 2
total_shards = NUM_TRAIN_SHARDS + NUM_VAL_SHARDS

t0 = time.time()
with Pool(8) as pool:
    results = pool.map(download_shard, range(total_shards))
print(f'\nDownloaded {sum(results)}/{total_shards} shards in {time.time()-t0:.1f}s')

# %%
# === Model definition ===
import pyarrow.parquet as pq

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


def _expand_kv(k, v, n_head, n_kv_head):
    """Repeat KV heads to match Q head count for non-splash backends."""
    if n_kv_head == n_head:
        return k, v
    ratio = n_head // n_kv_head
    return jnp.repeat(k, ratio, axis=1), jnp.repeat(v, ratio, axis=1)


def init_param_state(config: Config) -> dot_dict:
    """Initialize all model parameters."""
    key = jax.random.key(config.param_seed)
    n_embd = config.n_embd
    n_head = config.n_head
    n_kv_head = config.n_kv_head
    head_dim = config.head_dim
    n_layer = config.n_layer
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

    # Per-layer scalars
    params.resid_lambdas = jnp.ones(n_layer, dtype=jnp.bfloat16)
    params.x0_lambdas = jnp.full(n_layer, 0.1, dtype=jnp.bfloat16)

    # Precompute RoPE
    params.rope_cos, params.rope_sin = precompute_rope(config.seq_len, head_dim)

    # Layers
    params.layers = dot_dict()
    for i in range(n_layer):
        layer = dot_dict()
        # Attention projections: uniform(-s, s), proj=zeros
        layer.c_q = jax.random.uniform(split_key(), (n_embd, n_head, head_dim),
                                        dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.c_k = jax.random.uniform(split_key(), (n_embd, n_kv_head, head_dim),
                                        dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.c_v = jax.random.uniform(split_key(), (n_embd, n_kv_head, head_dim),
                                        dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.c_proj = jnp.zeros((n_head, head_dim, n_embd), dtype=jnp.bfloat16)
        # MLP: uniform(-s, s), proj=zeros
        layer.c_fc = jax.random.uniform(split_key(), (n_embd, 4 * n_embd),
                                         dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.mlp_proj = jnp.zeros((4 * n_embd, n_embd), dtype=jnp.bfloat16)

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


def model_apply(config: Config, params: dot_dict, tokens: jax.Array) -> jax.Array:
    """Forward pass: tokens (B, T) -> logits (B, T, vocab_size)."""
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
        x0 = x  # save for x0 residual connection

    for i in range(n_layer):
        layer = params.layers[i]

        # Pre-norm
        h = rms_norm(x)

        with jax.named_scope(f'layer_{i}/attention'):
            # === Attention ===
            q = jnp.einsum('btd,dhk->bhtk', h, layer.c_q)  # (B, H, T, D)
            k = jnp.einsum('btd,dhk->bhtk', h, layer.c_k)  # (B, KV, T, D)
            v = jnp.einsum('btd,dhk->bhtk', h, layer.c_v)  # (B, KV, T, D)

            # Apply RoPE
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            # QK-norm
            q = rms_norm(q)
            k = rms_norm(k)

            # Attention — dispatch to selected implementation
            w = window_sizes[i]

            if config.attn_impl == 'einsum':
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
                k_exp, v_exp = _expand_kv(k, v, n_head, n_kv_head)
                if w < T:
                    rows = jnp.arange(T)[:, None]
                    cols = jnp.arange(T)[None, :]
                    mask = (cols <= rows) & (cols >= rows - w + 1)
                    attn_out = jax.nn.dot_product_attention(
                        q, k_exp, v_exp, mask=mask[None, None, :, :], implementation='xla')
                else:
                    attn_out = jax.nn.dot_product_attention(
                        q, k_exp, v_exp, is_causal=True, implementation='xla')

            elif config.attn_impl == 'splash':
                from jax.experimental.pallas.ops.tpu.splash_attention import (
                    splash_attention_mask, splash_attention_kernel)

                smask = splash_attention_mask.CausalMask(shape=(T, T))
                if w < T:
                    smask = smask & splash_attention_mask.LocalMask(
                        shape=(T, T), window_size=(w, w), offset=0)

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

                attn_out = jax.vmap(kernel)(q, k, v)

            elif config.attn_impl == 'pallas':
                k_exp, v_exp = _expand_kv(k, v, n_head, n_kv_head)
                from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
                attn_out = flash_attention(
                    q, k_exp, v_exp,
                    causal=True,
                    sm_scale=head_dim ** -0.5,
                )

            # Output projection
            attn_out = jnp.einsum('bhtd,hde->bte', attn_out, layer.c_proj)

        # Residual with per-layer scaling
        x = params.resid_lambdas[i] * x + params.x0_lambdas[i] * x0
        x = x + attn_out

        with jax.named_scope(f'layer_{i}/mlp'):
            # === MLP ===
            h2 = rms_norm(x)
            mlp_out = jnp.einsum('btd,dh->bth', h2, layer.c_fc)
            mlp_out = jax.nn.relu(mlp_out) ** 2  # ReLU^2
            mlp_out = jnp.einsum('bth,hd->btd', mlp_out, layer.mlp_proj)
        x = x + mlp_out

    # Final norm + lm_head
    with jax.named_scope('lm_head'):
        x = rms_norm(x)
        logits = jnp.einsum('btd,dv->btv', x, params.lm_head,
                            preferred_element_type=jnp.float32)

        # Logit softcap
        logits = config.softcap * jnp.tanh(logits / config.softcap)

    return logits

# %%
# === Optimizer + training utilities ===

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
        logits = model_apply(config, full_params, x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
        return loss

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


@jax.jit
def eval_step(config: Config, params: dot_dict, x: jax.Array, y: jax.Array):
    """JIT-compiled eval: returns loss for a single batch."""
    logits = model_apply(config, params, x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))


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
                    if len(doc) > config_default.max_chars_per_doc:
                        doc = doc[:config_default.max_chars_per_doc]
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
# === Define sweep config ===
sweep_config = {
    "name": "tpuchat-hparam-sweep",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "early_terminate": {"type": "hyperband", "min_iter": 3, "eta": 3},
    "run_cap": 30,
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 5e-5, "max": 1e-3},
        "n_head": {"values": [4, 8]},
        "n_kv_head": {"values": [1, 2]},
        "head_dim": {"values": [128, 256]},
        "device_batch_size": {"values": [4, 8]},
        "warmup_ratio": {"distribution": "uniform", "min": 0.01, "max": 0.1},
        "warmdown_ratio": {"distribution": "uniform", "min": 0.3, "max": 0.7},
    },
}

# %%
# === Create sweep (run once) ===
sweep_id = wandb.sweep(sweep_config, project="tpuchat")
print(f"Sweep ID: {sweep_id}")

# %%
# === Train function (called by wandb agent) ===

def train_fn():
    """Single training run within a wandb sweep."""
    run = wandb.init()

    # Map wandb.config to Config dataclass — only override fields that exist
    config_fields = {f.name for f in dataclasses.fields(Config)}
    overrides = {k: v for k, v in dict(wandb.config).items() if k in config_fields}
    config = Config(**overrides)

    # Validate constraints
    if config.n_head % config.n_kv_head != 0:
        print(f'SKIP: n_head={config.n_head} not divisible by n_kv_head={config.n_kv_head}')
        wandb.log({"val_loss": 999.0})
        wandb.finish()
        return

    print(f'Config: n_head={config.n_head}, n_kv_head={config.n_kv_head}, '
          f'head_dim={config.head_dim}, n_embd={config.n_embd}, depth={config.depth}, '
          f'lr={config.learning_rate:.2e}, device_batch_size={config.device_batch_size}')

    # Sweep-specific settings
    num_iterations = config.num_iterations  # 1000 by default
    eval_every = config.eval_every          # 100 by default
    eval_steps = config.eval_steps          # 10 by default

    # Batch size
    total_batch_size = config.device_batch_size * config.seq_len

    # wandb metric definitions
    wandb.define_metric("train/loss", step_metric="step")
    wandb.define_metric("train/tok_per_sec", step_metric="step")
    wandb.define_metric("val/loss", step_metric="step")
    wandb.define_metric("val_loss", step_metric="step")

    # Init model + optimizer
    params = init_param_state(config)
    num_params = sum(p.size for p in jax.tree.leaves(params) if isinstance(p, jax.Array))
    print(f'Model parameters: {num_params:,}')

    trainable_params, static_params = split_trainable(params)
    opt_state = jax.tree.map(init_adam_state, trainable_params)

    # Data loaders
    train_shard_indices = list(range(NUM_TRAIN_SHARDS))
    val_shard_indices = list(range(NUM_TRAIN_SHARDS, NUM_TRAIN_SHARDS + NUM_VAL_SHARDS))

    raw_train_loader = tokenize_shards(train_shard_indices, config.device_batch_size, config.seq_len)
    train_loader = PrefetchDataLoader(raw_train_loader, capacity=4)
    val_loader_fn = lambda: tokenize_shards(val_shard_indices, config.device_batch_size, config.seq_len)

    smooth_loss = 0.0
    best_val_loss = float('inf')

    print(f'\n=== Starting sweep run for {num_iterations} steps ===\n')

    try:
        for step in range(num_iterations + 1):
            last_step = (step == num_iterations)

            # === Eval ===
            if eval_every > 0 and (last_step or step % eval_every == 0):
                val_loader = val_loader_fn()
                val_losses = []
                for ei in range(eval_steps):
                    vx, vy = next(val_loader)
                    vx, vy = jnp.array(vx), jnp.array(vy)
                    vl = eval_step(config, params, vx, vy)
                    val_losses.append(float(vl))
                avg_val_loss = sum(val_losses) / len(val_losses)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss

                wandb.log({
                    "step": step,
                    "val/loss": avg_val_loss,
                    "val_loss": avg_val_loss,  # sweep metric
                })
                print(f'Step {step:05d} | Val loss: {avg_val_loss:.4f} (best: {best_val_loss:.4f})')

            if last_step:
                break

            # === Train step ===
            lr_mult = jnp.array(get_lr_multiplier(step, num_iterations, config), dtype=jnp.float32)
            t0 = time.time()

            x_batch, y_batch = next(train_loader)
            loss, params, opt_state = train_step(config, params, opt_state, x_batch, y_batch, lr_mult)
            loss.block_until_ready()
            dt = time.time() - t0

            loss_val = float(loss)
            ema_beta = 0.9
            smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
            debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))

            if step % config.log_every == 0:
                tok_per_sec = int(total_batch_size / dt) if dt > 0 else 0
                wandb.log({
                    "step": step,
                    "train/loss": debiased_loss,
                    "train/tok_per_sec": tok_per_sec,
                })
                print(f'step {step:05d}/{num_iterations} | loss: {debiased_loss:.4f} | '
                      f'tok/s: {tok_per_sec:,}')

    finally:
        train_loader.stop()

    wandb.finish()
    print(f'Run complete. Best val loss: {best_val_loss:.4f}')

# %%
# === Run agent ===
SWEEP_ID = "PASTE_SWEEP_ID_HERE"  # @param {type:"string"}
NUM_RUNS = 5  # @param {type:"integer"}
wandb.agent(SWEEP_ID, function=train_fn, count=NUM_RUNS, project="tpuchat")
