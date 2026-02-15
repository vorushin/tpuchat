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
# # 02 — Pretrain GPT Model (JAX, Single TPU)
#
# Self-contained notebook that:
# 1. Downloads tokenizer from HuggingFace Hub
# 2. Downloads ~50 data shards from FineWeb-Edu-100B-Shuffle
# 3. Defines GPT model in raw JAX (no Flax) following the JAX training cookbook pattern
# 4. Trains the model on a single TPU v6e-1
# 5. Saves checkpoint to HuggingFace Hub
#
# Ported from [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy.

# %%
# Install dependencies (uncomment for Colab)
# !pip install -q jax[tpu] optax huggingface_hub tiktoken pyarrow requests torch

# %%
# === Config ===
import functools as ft
import itertools as it
import time
import os
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

@jax.tree_util.register_static
@dataclass(kw_only=True, frozen=True)
class Config:
    # Data
    num_shards: int = 50
    hf_repo_id: str = 'vorushin/tpuchat'

    # Model architecture
    depth: int = 12
    aspect_ratio: int = 64
    head_dim: int = 128
    vocab_size: int = 32768
    seq_len: int = 2048
    window_pattern: str = 'SSSL'
    softcap: float = 15.0

    # Training
    num_iterations: int = 1000  # set to -1 for auto from target_param_data_ratio
    target_param_data_ratio: float = 10.5
    device_batch_size: int = 8
    total_batch_size: int = -1  # -1 = auto
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

    # Derived (computed in __post_init__)
    @property
    def n_embd(self):
        base_dim = self.depth * self.aspect_ratio
        return ((base_dim + self.head_dim - 1) // self.head_dim) * self.head_dim

    @property
    def n_head(self):
        return self.n_embd // self.head_dim

    @property
    def n_kv_head(self):
        return self.n_head  # no GQA for simplicity

    @property
    def n_layer(self):
        return self.depth

    @property
    def padded_vocab(self):
        return ((self.vocab_size + 63) // 64) * 64

config = Config()
print(f'Model: depth={config.depth}, n_embd={config.n_embd}, n_head={config.n_head}, '
      f'head_dim={config.head_dim}, vocab={config.vocab_size} (padded={config.padded_vocab})')

# %%
# === HuggingFace Hub login + download tokenizer ===
from huggingface_hub import login, HfApi, hf_hub_download
login()  # will prompt for your HF token

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
    """Apply rotary embeddings. x: (B, T, H, D)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have value embedding (alternating, last always)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def init_param_state(config: Config) -> dot_dict:
    """Initialize all model parameters."""
    key = jax.random.key(config.param_seed)
    n_embd = config.n_embd
    n_head = config.n_head
    n_kv_head = config.n_kv_head
    head_dim = config.head_dim
    n_layer = config.n_layer
    padded_vocab = config.padded_vocab
    kv_dim = n_kv_head * head_dim

    def split_key():
        nonlocal key
        key, subkey = jax.random.split(key)
        return subkey

    # Uniform init bound (matches nanochat: sqrt(3) * std = sqrt(3) / sqrt(n_embd))
    s = (3.0 ** 0.5) * (n_embd ** -0.5)

    params = dot_dict()

    # Token embedding: normal(0, 1)
    params.wte = jax.random.normal(split_key(), (padded_vocab, n_embd), dtype=jnp.bfloat16)

    # LM head: normal(0, 0.001)
    params.lm_head = jax.random.normal(split_key(), (n_embd, padded_vocab), dtype=jnp.bfloat16) * 0.001

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
        layer.c_q = jax.random.uniform(split_key(), (n_embd, n_head * head_dim),
                                        dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.c_k = jax.random.uniform(split_key(), (n_embd, n_kv_head * head_dim),
                                        dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.c_v = jax.random.uniform(split_key(), (n_embd, n_kv_head * head_dim),
                                        dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.c_proj = jnp.zeros((n_head * head_dim, n_embd), dtype=jnp.bfloat16)
        # MLP: uniform(-s, s), proj=zeros
        layer.c_fc = jax.random.uniform(split_key(), (n_embd, 4 * n_embd),
                                         dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.mlp_proj = jnp.zeros((4 * n_embd, n_embd), dtype=jnp.bfloat16)

        # Value embedding (alternating layers)
        if has_ve(i, n_layer):
            layer.ve_embed = jax.random.uniform(split_key(), (padded_vocab, kv_dim),
                                                 dtype=jnp.bfloat16, minval=-s, maxval=s)
            layer.ve_gate = jnp.zeros((32, n_kv_head), dtype=jnp.bfloat16)

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

    # RoPE: (T, head_dim/2) -> (1, T, 1, head_dim/2) for broadcasting
    cos = params.rope_cos[:T][None, :, None, :]  # (1, T, 1, D/2)
    sin = params.rope_sin[:T][None, :, None, :]

    # Token embedding + norm
    x = params.wte[tokens]  # (B, T, n_embd)
    x = rms_norm(x)
    x0 = x  # save for x0 residual connection

    for i in range(n_layer):
        layer = params.layers[i]

        # Pre-norm
        h = rms_norm(x)

        # === Attention ===
        q = jnp.einsum('btd,dh->bth', h, layer.c_q)  # (B, T, n_head*head_dim)
        k = jnp.einsum('btd,dh->bth', h, layer.c_k)  # (B, T, n_kv*head_dim)
        v = jnp.einsum('btd,dh->bth', h, layer.c_v)

        # Reshape to (B, T, H, D)
        q = q.reshape(B, T, n_head, head_dim)
        k = k.reshape(B, T, n_kv_head, head_dim)
        v = v.reshape(B, T, n_kv_head, head_dim)

        # Value embedding (ResFormer-style)
        if has_ve(i, n_layer):
            ve = layer.ve_embed[tokens]  # (B, T, kv_dim)
            ve = ve.reshape(B, T, n_kv_head, head_dim)
            # Gate: input-dependent, per head
            gate = 2.0 * jax.nn.sigmoid(jnp.einsum('btd,dh->bth', h[:, :, :32], layer.ve_gate))
            v = v + gate[:, :, :, None] * ve

        # Apply RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # QK-norm
        q = rms_norm(q)
        k = rms_norm(k)

        # Attention via jax.nn.dot_product_attention
        # JAX expects (B, T, H, D) — no transpose needed

        # GQA: repeat k,v heads if needed
        if n_kv_head < n_head:
            repeats = n_head // n_kv_head
            k = jnp.repeat(k, repeats, axis=2)
            v = jnp.repeat(v, repeats, axis=2)

        # Sliding window: create mask if window < seq_len
        w = window_sizes[i]
        if w < T:
            # Create sliding window causal mask: (T, T)
            rows = jnp.arange(T)[:, None]
            cols = jnp.arange(T)[None, :]
            mask = (cols <= rows) & (cols >= rows - w + 1)
            attn_out = jax.nn.dot_product_attention(q, k, v, mask=mask)
        else:
            attn_out = jax.nn.dot_product_attention(q, k, v, is_causal=True)

        # Already (B, T, H, D), reshape to (B, T, n_head*head_dim)
        attn_out = attn_out.reshape(B, T, -1)

        # Output projection
        attn_out = jnp.einsum('btd,de->bte', attn_out, layer.c_proj)

        # Residual with per-layer scaling
        x = params.resid_lambdas[i] * x + params.x0_lambdas[i] * x0
        x = x + attn_out

        # === MLP ===
        h2 = rms_norm(x)
        mlp_out = jnp.einsum('btd,dh->bth', h2, layer.c_fc)
        mlp_out = jax.nn.relu(mlp_out) ** 2  # ReLU^2
        mlp_out = jnp.einsum('bth,hd->btd', mlp_out, layer.mlp_proj)
        x = x + mlp_out

    # Final norm + lm_head
    x = rms_norm(x)
    logits = jnp.einsum('btd,dv->btv', x, params.lm_head)
    logits = logits[:, :, :config.vocab_size]  # remove padding

    # Logit softcap
    logits = logits.astype(jnp.float32)
    logits = config.softcap * jnp.tanh(logits / config.softcap)

    return logits


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
    """Count parameters that contribute to scaling (matrices + lm_head)."""
    count = 0
    for i in range(config.n_layer):
        layer = params.layers[i]
        for name in ['c_q', 'c_k', 'c_v', 'c_proj', 'c_fc', 'mlp_proj']:
            count += layer[name].size
        if has_ve(i, config.n_layer):
            count += layer['ve_gate'].size
    count += params.lm_head.size
    return count

scaling_params = count_matrix_params(params)
target_tokens = int(config.target_param_data_ratio * scaling_params)

# Batch size
total_batch_size = config.total_batch_size
if total_batch_size == -1:
    total_batch_size = config.device_batch_size * config.seq_len
    print(f'Total batch size: {total_batch_size:,} tokens/step')

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
    """Single training step with proper functional updates."""
    trainable, static = split_trainable(params)

    def loss_fn(trainable_params):
        full_params = merge_params(trainable_params, static)
        logits = model_apply(config, full_params, x)
        loss = jnp.mean(
            jax.vmap(jax.vmap(lambda logit, target: -jax.nn.log_softmax(logit)[target]))(logits, y)
        )
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(trainable)

    # Flatten trees to apply optimizer per-leaf
    # (can't use jax.tree.map because adamw_step returns a tuple,
    #  and tuples are pytree nodes that JAX would try to traverse)
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

    return loss, merge_params(new_trainable, static), new_opt_state


# Initialize optimizer state (only for trainable params)
trainable_params, static_params = split_trainable(params)
opt_state = jax.tree.map(init_adam_state, trainable_params)

print('Optimizer state initialized.')
print(f'Trainable param arrays: {len(jax.tree.leaves(trainable_params))}')

# %%
# === Training loop ===

train_loader = tokenize_shards(train_shard_indices, config.device_batch_size, config.seq_len)
val_loader_fn = lambda: tokenize_shards(val_shard_indices, config.device_batch_size, config.seq_len)

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
        for eval_step in range(config.eval_steps):
            vx, vy = next(val_loader)
            vx, vy = jnp.array(vx), jnp.array(vy)
            trainable, static = split_trainable(params)
            def val_loss_fn(tp):
                full_p = merge_params(tp, static)
                logits = model_apply(config, full_p, vx)
                return jnp.mean(
                    jax.vmap(jax.vmap(lambda l, t: -jax.nn.log_softmax(l)[t]))(logits, vy)
                )
            vl = val_loss_fn(trainable)
            val_losses.append(float(vl))
        avg_val_loss = sum(val_losses) / len(val_losses)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        val_loss_history.append((step, avg_val_loss))
        print(f'Step {step:05d} | Val loss: {avg_val_loss:.4f} (best: {best_val_loss:.4f})')

    # === Sample ===
    if config.sample_every > 0 and step > 0 and (last_step or step % config.sample_every == 0):
        prompt = 'The capital of France is'
        prompt_ids = enc.encode_ordinary(prompt)
        bos_id = enc.encode_single_token('<|bos|>')
        ids = jnp.array([[bos_id] + prompt_ids], dtype=jnp.int32)
        for _ in range(50):
            logits = model_apply(config, params, ids)
            next_logit = logits[0, -1, :]
            next_id = jnp.argmax(next_logit)
            ids = jnp.concatenate([ids, next_id[None, None]], axis=1)
        sample_text = enc.decode(ids[0].tolist())
        print(f'Sample: {sample_text}')

    if last_step:
        break

    # === Train step ===
    x_np, y_np = next(train_loader)
    x_batch = jnp.array(x_np)
    y_batch = jnp.array(y_np)
    lr_mult = jnp.array(get_lr_multiplier(step, num_iterations, config), dtype=jnp.float32)

    t0 = time.time()
    loss, params, opt_state = train_step(config, params, opt_state, x_batch, y_batch, lr_mult)
    jax.block_until_ready(loss)
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
ax.set_title(f'Training curves — depth={config.depth}, batch={total_batch_size:,} tok/step')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# === Save checkpoint to HuggingFace Hub ===
import pickle

CHECKPOINT_DIR = '/content/checkpoint'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Save params as pickle (simple, works for raw JAX arrays)
# Convert to numpy for serialization
params_np = jax.tree.map(lambda x: np.array(x) if isinstance(x, jax.Array) else x,
                          params)
with open(os.path.join(CHECKPOINT_DIR, 'params.pkl'), 'wb') as f:
    pickle.dump(params_np, f)
print(f'Saved params to {CHECKPOINT_DIR}/params.pkl')

# Save config
import json
config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
with open(os.path.join(CHECKPOINT_DIR, 'config.json'), 'w') as f:
    json.dump(config_dict, f, indent=2, default=str)

# Upload to HF Hub
api = HfApi()
api.create_repo(config.hf_repo_id, repo_type='model', exist_ok=True)
api.upload_folder(
    folder_path=CHECKPOINT_DIR,
    repo_id=config.hf_repo_id,
    path_in_repo=f'checkpoint_d{config.depth}',
    commit_message=f'Upload checkpoint (depth={config.depth}, {num_iterations} steps)',
)
print(f'\nUploaded to https://huggingface.co/{config.hf_repo_id}/tree/main/checkpoint_d{config.depth}')
