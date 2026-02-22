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
# # 07 — Train ~100M Transformer on Apple M4 Pro (MLX)
#
# Self-contained notebook that:
# 1. Downloads tokenizer from HuggingFace Hub
# 2. Downloads ~50 data shards from FineWeb-Edu-100B-Shuffle
# 3. Defines a ~99M param transformer in raw MLX (no nn.Module)
# 4. Trains for 1B tokens (~61K steps) on Apple M4 Pro
# 5. Saves checkpoints locally
#
# Architecture: E=768, L=8, D=128, H=6, KV=1, MLP=2048 (SwiGLU)
# Chosen based on 06_apple_silicon_perf.py benchmarks:
# - Bigger matmuls → higher GPU% (768×2048 > 512×1536)
# - Fewer layers = less bandwidth-bound overhead (8 vs 22)
# - SDPA ~2x faster than einsum, mx.fast.rope 2.3x speedup
# - GQA n_kv=1 marginally fastest
#
# Expected: ~9K-10K tok/s (~34% MFU), ~25 hours wall clock, final loss ~3.5-4.0

# %%
import functools as ft
import itertools as it
import time
import os
import math
import queue
import threading
import pickle
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np

# %% [markdown]
# ## Config

# %%
@dataclass(kw_only=True, frozen=True)
class Config:
    # Data
    num_shards: int = 50
    hf_repo_id: str = 'vorushin/tpuchat'
    max_chars_per_doc: int = 10_000
    data_dir: str = os.path.expanduser('~/tpuchat_data/shards')

    # Model architecture — ~99M params
    n_layer: int = 8
    n_embd: int = 768
    n_head: int = 6
    n_kv_head: int = 1
    head_dim: int = 128
    mlp_dim: int = 2048        # 8/3 × E ≈ SwiGLU standard ratio
    vocab_size: int = 32768
    softcap: float = 15.0
    num_lm_head_chunks: int = 2

    # Training
    seq_len: int = 2048
    device_batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    warmup_ratio: float = 0.02
    warmdown_ratio: float = 0.5
    max_grad_norm: float = 1.0
    use_remat: bool = False
    target_tokens: int = 1_000_000_000  # 1B tokens

    # Eval / Logging
    eval_every: int = 500
    eval_steps: int = 10
    log_every: int = 10
    sample_every: int = 2000
    save_every: int = 10000

    # Seed
    param_seed: int = 42

    @property
    def total_batch_tokens(self):
        return self.device_batch_size * self.seq_len

    @property
    def num_iterations(self):
        return self.target_tokens // self.total_batch_tokens


cfg = Config()
assert cfg.vocab_size % 256 == 0, f"vocab_size must be divisible by 256, got {cfg.vocab_size}"
assert cfg.n_embd == cfg.n_head * cfg.head_dim, \
    f'n_embd ({cfg.n_embd}) must equal n_head * head_dim ({cfg.n_head * cfg.head_dim})'
assert cfg.n_head % cfg.n_kv_head == 0, \
    f'n_head ({cfg.n_head}) must be divisible by n_kv_head ({cfg.n_kv_head})'

print(f"Config: B={cfg.device_batch_size}, T={cfg.seq_len}, E={cfg.n_embd}, "
      f"H={cfg.n_head}, KV={cfg.n_kv_head}, D={cfg.head_dim}, "
      f"MLP={cfg.mlp_dim}, V={cfg.vocab_size}, L={cfg.n_layer}")
print(f"Tokens/step: {cfg.total_batch_tokens:,}, "
      f"Total steps: {cfg.num_iterations:,}, "
      f"Target tokens: {cfg.target_tokens:,}")

# %% [markdown]
# ## Tokenizer + Data

# %%
from huggingface_hub import hf_hub_download
import tiktoken
import pyarrow.parquet as pq
import requests

# Download tokenizer
TOKENIZER_DIR = os.path.expanduser('~/tpuchat_data/tokenizer')
os.makedirs(TOKENIZER_DIR, exist_ok=True)

tok_pkl_path = hf_hub_download(
    repo_id=cfg.hf_repo_id,
    filename='tokenizer/tokenizer.pkl',
    local_dir=TOKENIZER_DIR,
)
print(f'Downloaded tokenizer to {TOKENIZER_DIR}')

with open(os.path.join(TOKENIZER_DIR, 'tokenizer', 'tokenizer.pkl'), 'rb') as f:
    enc = pickle.load(f)
print(f'Loaded tokenizer: vocab_size={enc.n_vocab}')

# %%
# Download data shards
os.makedirs(cfg.data_dir, exist_ok=True)

BASE_URL = 'https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main'

def download_shard(index):
    filename = f'shard_{index:05d}.parquet'
    filepath = os.path.join(cfg.data_dir, filename)
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

NUM_TRAIN_SHARDS = cfg.num_shards
NUM_VAL_SHARDS = 2
total_shards = NUM_TRAIN_SHARDS + NUM_VAL_SHARDS

t0 = time.time()
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(download_shard, range(total_shards)))
print(f'\nDownloaded {sum(results)}/{total_shards} shards in {time.time()-t0:.1f}s')

# %%
# Data pipeline: tokenize parquet shards into (x, y) batches
def tokenize_shards(shard_indices, batch_size, seq_len):
    """Yield (x, y) batches by tokenizing parquet shards on the fly.

    Uses BOS-aligned packing: each document starts with BOS, documents are
    concatenated into a token buffer, and batches are sliced from the buffer.
    """
    bos_id = enc.encode_single_token('<|bos|>')
    buf = []

    while True:  # loop over epochs
        for shard_idx in shard_indices:
            filepath = os.path.join(cfg.data_dir, f'shard_{shard_idx:05d}.parquet')
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                texts = rg.column('text').to_pylist()
                for doc in texts:
                    if len(doc) > cfg.max_chars_per_doc:
                        doc = doc[:cfg.max_chars_per_doc]
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
print(f'Batch: {cfg.device_batch_size} x {cfg.seq_len} = {cfg.total_batch_tokens:,} tokens/step')

# %%
# Background data prefetch (unified memory — no device_put needed)
@dataclass
class PrefetchDataLoader:
    """Wraps an iterator and prefetches items in a background thread."""
    iterator: any
    capacity: int = 4

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
                # Unified memory: no host->device copy needed
                x, y = item
                self.queue.put((mx.array(x), mx.array(y)))
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

# %% [markdown]
# ## Model

# %%
def init_layer_params(cfg, seed=42):
    """Initialize params for one transformer layer."""
    mx.random.seed(seed)
    s = (3.0 ** 0.5) * (cfg.n_embd ** -0.5)
    return dict(
        c_q=mx.random.uniform(-s, s, (cfg.n_embd, cfg.n_head, cfg.head_dim)).astype(mx.bfloat16),
        c_k=mx.random.uniform(-s, s, (cfg.n_embd, cfg.n_kv_head, cfg.head_dim)).astype(mx.bfloat16),
        c_v=mx.random.uniform(-s, s, (cfg.n_embd, cfg.n_kv_head, cfg.head_dim)).astype(mx.bfloat16),
        c_proj=mx.zeros((cfg.n_head, cfg.head_dim, cfg.n_embd), dtype=mx.bfloat16),
        w_gate=mx.random.uniform(-s, s, (cfg.n_embd, cfg.mlp_dim)).astype(mx.bfloat16),
        w_up=mx.random.uniform(-s, s, (cfg.n_embd, cfg.mlp_dim)).astype(mx.bfloat16),
        w_down=mx.zeros((cfg.mlp_dim, cfg.n_embd), dtype=mx.bfloat16),
        attn_norm_w=mx.ones((cfg.n_embd,), dtype=mx.bfloat16),
        mlp_norm_w=mx.ones((cfg.n_embd,), dtype=mx.bfloat16),
        qk_norm_w=mx.ones((cfg.head_dim,), dtype=mx.bfloat16),
    )


def single_layer_forward(cfg, layer, x):
    """Forward pass for one transformer layer."""
    h = mx.fast.rms_norm(x, layer['attn_norm_w'], 1e-6)

    # Attention projections -> (B, H, T, D)
    q = mx.einsum('btd,dhk->bhtk', h, layer['c_q'])
    k = mx.einsum('btd,dhk->bhtk', h, layer['c_k'])
    v = mx.einsum('btd,dhk->bhtk', h, layer['c_v'])

    # RoPE
    q = mx.fast.rope(q, dims=cfg.head_dim, traditional=False,
                     base=10000.0, scale=1.0, offset=0)
    k = mx.fast.rope(k, dims=cfg.head_dim, traditional=False,
                     base=10000.0, scale=1.0, offset=0)

    # QK norm
    q = mx.fast.rms_norm(q, layer['qk_norm_w'], 1e-6)
    k = mx.fast.rms_norm(k, layer['qk_norm_w'], 1e-6)

    # SDPA (handles GQA natively)
    T = x.shape[1]
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=q.dtype), k=1)
    attn_out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=cfg.head_dim ** -0.5, mask=mask)

    # Output projection
    attn_out = mx.einsum('bhtd,hde->bte', attn_out, layer['c_proj'])
    x = x + attn_out

    # SwiGLU MLP
    h2 = mx.fast.rms_norm(x, layer['mlp_norm_w'], 1e-6)
    gate = nn.silu(mx.einsum('btd,dh->bth', h2, layer['w_gate']))
    up = mx.einsum('btd,dh->bth', h2, layer['w_up'])
    mlp_out = mx.einsum('bth,hd->btd', gate * up, layer['w_down'])
    x = x + mlp_out
    return x


def init_full_model(cfg, seed=42):
    """Initialize all model params (embed + layers + lm_head)."""
    mx.random.seed(seed)
    params = {}
    params['wte'] = mx.random.normal((cfg.vocab_size, cfg.n_embd)).astype(mx.bfloat16)
    params['lm_head'] = mx.random.normal((cfg.n_embd, cfg.vocab_size)).astype(mx.bfloat16) * 0.001
    params['emb_norm_w'] = mx.ones((cfg.n_embd,), dtype=mx.bfloat16)
    params['final_norm_w'] = mx.ones((cfg.n_embd,), dtype=mx.bfloat16)
    params['layers'] = {}
    for i in range(cfg.n_layer):
        params['layers'][i] = init_layer_params(cfg, seed=seed + 100 + i * 7)
    return params


def model_forward(cfg, params, tokens):
    """Full forward: tokens (B, T) -> hidden (B, T, E)."""
    x = mx.fast.rms_norm(params['wte'][tokens], params['emb_norm_w'], 1e-6)
    for i in range(cfg.n_layer):
        if cfg.use_remat:
            layer_fn = ft.partial(single_layer_forward, cfg)
            x = mx.checkpoint(layer_fn)(params['layers'][i], x)
        else:
            x = single_layer_forward(cfg, params['layers'][i], x)
    return mx.fast.rms_norm(x, params['final_norm_w'], 1e-6)


def chunked_lm_head_loss(hidden, lm_head, labels, config):
    """Chunked LM head to reduce peak memory from full vocab logits."""
    B, T, D = hidden.shape
    N = config.num_lm_head_chunks
    S = B * T // N
    hidden_flat = hidden.reshape(N, S, D)
    labels_flat = labels.reshape(N, S)

    total_loss = mx.array(0.0)
    for i in range(N):
        logits = hidden_flat[i] @ lm_head          # (S, vocab_size)
        logits = logits.astype(mx.float32)
        logits = config.softcap * mx.tanh(logits / config.softcap)
        total_loss = total_loss + mx.sum(nn.losses.cross_entropy(logits, labels_flat[i]))
    return total_loss / (B * T)


# Initialize model
params = init_full_model(cfg, seed=cfg.param_seed)
mx.eval(params)

# Count params
num_params = sum(p.size for _, p in mlx.utils.tree_flatten(params))
print(f'Model parameters: {num_params:,} ({num_params/1e6:.1f}M)')
print(f'Model memory (bf16): {num_params * 2 / 1e6:.0f} MB')

# FLOP counting for MFU (matching 08_tpu_ablations methodology)
PEAK_TFLOPS = 17.2  # Apple M4 Pro bf16 peak

def matmul_flops(M, N, K, batch=1):
    return 2 * batch * M * N * K

def attention_flops(B, N, T, H):
    return 2 * (2 * B * N * T * T * H)

def compute_layer_flops(B, T, D, N, K, H, F):
    """GLU MLP: 3 matmuls (gate + up + down)."""
    tok = B * T
    q    = 2 * tok * D * N * H
    k    = 2 * tok * D * K * H
    v    = 2 * tok * D * K * H
    att  = attention_flops(B, N, T, H)
    proj = 2 * tok * N * H * D
    mlp  = 3 * (2 * tok * D * F)
    return q + k + v + att + proj + mlp

fwd_flops = (cfg.n_layer * compute_layer_flops(
    cfg.device_batch_size, cfg.seq_len, cfg.n_embd,
    cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim)
    + matmul_flops(cfg.device_batch_size * cfg.seq_len,
                   cfg.vocab_size, cfg.n_embd))
step_flops = 3 * fwd_flops
flops_per_tok = step_flops / cfg.total_batch_tokens
ideal_tok_s = PEAK_TFLOPS * 1e12 / flops_per_tok
print(f'Step FLOPs: {step_flops/1e12:.3f} TFLOP | '
      f'FLOPs/tok: {flops_per_tok/1e6:.1f}M | '
      f'Ideal tok/s (100% MFU): {int(ideal_tok_s):,}')

# %% [markdown]
# ## Optimizer

# %%
import mlx.optimizers as optim

# LR schedule: linear warmup + constant + linear warmdown
num_iterations = cfg.num_iterations
warmup_steps = int(cfg.warmup_ratio * num_iterations)
warmdown_steps = int(cfg.warmdown_ratio * num_iterations)
constant_steps = num_iterations - warmup_steps - warmdown_steps

warmup = optim.linear_schedule(0, cfg.learning_rate, warmup_steps)
constant = optim.linear_schedule(cfg.learning_rate, cfg.learning_rate, constant_steps)
warmdown = optim.linear_schedule(cfg.learning_rate, 0, warmdown_steps)
lr_schedule = optim.join_schedules(
    [warmup, constant, warmdown],
    [warmup_steps, warmup_steps + constant_steps]
)

optimizer = optim.AdamW(
    learning_rate=lr_schedule,
    betas=[cfg.beta1, cfg.beta2],
    eps=cfg.eps,
    weight_decay=cfg.weight_decay,
)

print(f'Optimizer: AdamW (lr={cfg.learning_rate}, wd={cfg.weight_decay}, '
      f'betas=({cfg.beta1}, {cfg.beta2}))')
print(f'Schedule: warmup {warmup_steps} + constant {constant_steps} + warmdown {warmdown_steps} = {num_iterations} steps')

# %% [markdown]
# ## Training Step

# %%
@mx.compile
def compiled_loss_and_grads(params, x, y):
    """Compiled forward + backward + grad clip (compute-heavy part)."""
    def loss_fn(p):
        hidden = model_forward(cfg, p, x)
        return chunked_lm_head_loss(hidden, p['lm_head'], y, cfg)
    loss, grads = mx.value_and_grad(loss_fn)(params)
    grads, gnorm = optim.clip_grad_norm(grads, max_norm=cfg.max_grad_norm)
    return loss, grads, gnorm


def train_step(cfg, params, optimizer, x, y):
    """Single training step: compiled fwd/bwd + optimizer update."""
    loss, grads, gnorm = compiled_loss_and_grads(params, x, y)
    new_params = optimizer.apply_gradients(grads, params)
    return loss, new_params, gnorm

# %% [markdown]
# ## Eval + Generation

# %%
def eval_loss(cfg, params, val_loader_fn, num_steps):
    """Evaluate average loss over num_steps batches."""
    val_loader = val_loader_fn()
    losses = []
    for _ in range(num_steps):
        vx, vy = next(val_loader)
        vx, vy = mx.array(vx), mx.array(vy)
        hidden = model_forward(cfg, params, vx)
        loss = chunked_lm_head_loss(hidden, params['lm_head'], vy, cfg)
        mx.eval(loss)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def generate(cfg, params, enc, prompt, max_new_tokens=100,
             temperature=0.8, top_k=200, top_p=0.95, seed=None):
    """Generate text from a prompt using temperature + top-k + top-p sampling."""
    if seed is None:
        seed = int(time.time() * 1000) % (2**31)

    prompt_ids = enc.encode_ordinary(prompt)
    bos_id = enc.encode_single_token('<|bos|>')
    ids = [bos_id] + prompt_ids

    for _ in range(max_new_tokens):
        context = ids[-cfg.seq_len:]
        # Pad to full seq_len to avoid recompilation
        pad_len = cfg.seq_len - len(context)
        padded_context = context + [0] * pad_len
        x = mx.array([padded_context], dtype=mx.int32)

        hidden = model_forward(cfg, params, x)
        # Get logits at last valid position
        h = hidden[0, len(context) - 1, :]  # (E,)
        logits = h @ params['lm_head']  # (vocab_size,)
        logits = logits.astype(mx.float32)
        logits = cfg.softcap * mx.tanh(logits / cfg.softcap)
        mx.eval(logits)

        if temperature == 0:
            next_id = int(mx.argmax(logits))
        else:
            logits = logits / temperature

            # Top-k
            if top_k > 0:
                top_k_vals = mx.sort(logits)[-top_k]
                logits = mx.where(logits < top_k_vals, float('-inf'), logits)

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_indices = mx.argsort(logits)[::-1]
                sorted_logits = logits[sorted_indices]
                sorted_probs = mx.softmax(sorted_logits)
                cumulative_probs = mx.cumsum(sorted_probs)
                # Remove tokens with cumulative probability above threshold
                remove_mask = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                remove_mask = mx.concatenate([mx.array([False]), remove_mask[:-1]])
                sorted_logits = mx.where(remove_mask, float('-inf'), sorted_logits)
                # Scatter back
                logits = mx.zeros_like(logits)
                logits[sorted_indices] = sorted_logits

            # Sample
            mx.random.seed(seed + len(ids))
            probs = mx.softmax(logits)
            next_id = int(mx.random.categorical(mx.log(probs + 1e-10)))

        ids.append(next_id)

    return enc.decode(ids)


PROMPTS = [
    'The capital of France is',
    'In a distant galaxy, scientists discovered',
    'The quick brown fox',
    'Machine learning is',
]

# %% [markdown]
# ## Checkpoint

# %%
CHECKPOINT_DIR = os.path.expanduser('~/tpuchat_data/checkpoints_mlx')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(params, opt_state, opt_step, step):
    """Save checkpoint: params + optimizer state + step."""
    path = os.path.join(CHECKPOINT_DIR, f'step_{step:06d}.npz')
    flat_params = {f'param.{k}': v for k, v in mlx.utils.tree_flatten(params)}
    flat_opt = {f'opt.{k}': v for k, v in mlx.utils.tree_flatten(opt_state)}
    all_arrays = {**flat_params, **flat_opt, 'opt_step': mx.array(opt_step)}
    mx.savez(path, **all_arrays)
    print(f'Saved checkpoint to {path}')
    return path


def load_checkpoint(path):
    """Load checkpoint and return (flat_params, flat_opt, opt_step, step)."""
    data = dict(mx.load(path))
    flat_params = [(k[len('param.'):], v) for k, v in data.items() if k.startswith('param.')]
    flat_opt = [(k[len('opt.'):], v) for k, v in data.items() if k.startswith('opt.')]
    params = mlx.utils.tree_unflatten(flat_params)
    opt_state = mlx.utils.tree_unflatten(flat_opt)
    opt_step = int(data['opt_step'])
    # Extract step from filename
    basename = os.path.basename(path)
    step = int(basename.split('_')[1].split('.')[0])
    return params, opt_state, opt_step, step

# %% [markdown]
# ## Training Loop

# %%
raw_train_loader = tokenize_shards(train_shard_indices, cfg.device_batch_size, cfg.seq_len)
train_loader = PrefetchDataLoader(raw_train_loader, capacity=4)
val_loader_fn = lambda: tokenize_shards(val_shard_indices, cfg.device_batch_size, cfg.seq_len)

# History for plotting
train_loss_history = []
val_loss_history = []

smooth_loss = 0.0
total_training_time = 0.0
best_val_loss = float('inf')
MFU_START_STEP = 20
mfu_t0 = None
mfu_tokens = 0
mfu_eval_time = 0.0

print(f'\n=== Starting training for {num_iterations:,} steps ===')
print(f'Expected: ~{num_iterations * cfg.total_batch_tokens / 1e9:.1f}B tokens')
print()

for step in range(num_iterations + 1):
    last_step = (step == num_iterations)

    # === Eval ===
    if cfg.eval_every > 0 and (last_step or step % cfg.eval_every == 0):
        eval_t0 = time.time()
        avg_val_loss = eval_loss(cfg, params, val_loader_fn, cfg.eval_steps)
        if mfu_t0 is not None:
            mfu_eval_time += time.time() - eval_t0
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        val_loss_history.append((step, avg_val_loss))
        print(f'Step {step:06d} | Val loss: {avg_val_loss:.4f} (best: {best_val_loss:.4f})')

    # === Sample ===
    if cfg.sample_every > 0 and step > 0 and (last_step or step % cfg.sample_every == 0):
        print(f"\n--- Samples (step {step}) ---")
        for prompt in PROMPTS:
            sample_text = generate(cfg, params, enc, prompt, max_new_tokens=64, top_p=0.95)
            print(f"Prompt: {prompt}\nOutput: {sample_text}\n")
        print("----------------------------\n")

    # === Save ===
    if cfg.save_every > 0 and step > 0 and (last_step or step % cfg.save_every == 0):
        save_checkpoint(params, optimizer.state, optimizer.step.item(), step)

    if last_step:
        break

    # === Metal GPU profiling on steps 15-20 ===
    if step == 15:
        try:
            mx.metal.start_capture("mlx_trace.gputrace")
            print("Metal GPU capture started...")
        except Exception as e:
            print(f"Metal capture not available: {e}")
    if step == 20:
        try:
            mx.metal.stop_capture()
            print("Metal GPU capture stopped. Saved to mlx_trace.gputrace")
        except Exception:
            pass

    # === Train step ===
    t0 = time.time()

    x_batch, y_batch = next(train_loader)
    loss, params, gnorm = train_step(cfg, params, optimizer, x_batch, y_batch)
    mx.eval(loss, params)
    dt = time.time() - t0

    # MFU tracking
    if step == MFU_START_STEP:
        mfu_t0 = time.time()
    if step > MFU_START_STEP:
        mfu_tokens += cfg.total_batch_tokens

    if step > 10:
        total_training_time += dt

    loss_val = loss.item()
    ema_beta = 0.9
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
    debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))

    train_loss_history.append((step, debiased_loss))

    if step % cfg.log_every == 0:
        tok_per_sec = int(cfg.total_batch_tokens / dt) if dt > 0 else 0
        pct = 100 * step / num_iterations
        eta = ''
        if step > 10 and total_training_time > 0:
            avg_dt = total_training_time / (step - 10)
            remaining = (num_iterations - step) * avg_dt
            hrs = remaining / 3600
            eta = f' | eta: {hrs:.1f}h' if hrs >= 1 else f' | eta: {remaining/60:.1f}m'
        gnorm_val = gnorm.item()
        cur_lr = optimizer.learning_rate.item()
        mfu_str = ''
        if step > MFU_START_STEP and mfu_t0 is not None:
            step_mfu = step_flops / (PEAK_TFLOPS * 1e12 * dt) * 100 if dt > 0 else 0
            mfu_str = f' | MFU: {step_mfu:.1f}%'
        print(f'step {step:06d}/{num_iterations:06d} ({pct:5.1f}%) | '
              f'loss: {debiased_loss:.4f} | lr: {cur_lr:.2e} | '
              f'gnorm: {gnorm_val:.2f} | dt: {dt*1000:.0f}ms | '
              f'tok/s: {tok_per_sec:,}{mfu_str}{eta}')

print(f'\nTraining complete. Total time: {total_training_time/3600:.1f}h')
print(f'Best val loss: {best_val_loss:.4f}')

if mfu_t0 is not None:
    mfu_wall = time.time() - mfu_t0 - mfu_eval_time
    tok_per_s = int(mfu_tokens / mfu_wall)
    mfu_pct = (tok_per_s * flops_per_tok) / (PEAK_TFLOPS * 1e12) * 100
    print(f'MFU: {mfu_pct:.1f}% | tok/s: {tok_per_s:,} | '
          f'ideal tok/s (100% MFU): {int(ideal_tok_s):,}')

# %%
# === Plot training curves ===
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(12, 5))

if train_loss_history:
    steps, losses = zip(*train_loss_history)
    ax.plot(steps, losses, label='Train loss (smoothed)', alpha=0.8, linewidth=1)

if val_loss_history:
    steps, losses = zip(*val_loss_history)
    ax.plot(steps, losses, 'ro-', label='Val loss', markersize=5)

ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title(f'Training curves — E={cfg.n_embd}, L={cfg.n_layer}, {num_params/1e6:.0f}M params')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.expanduser('~/tpuchat_data/training_curves_mlx.png'), dpi=150)
print('Saved training curves to ~/tpuchat_data/training_curves_mlx.png')
plt.close()
