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
#   accelerator: GPU
# ---

# %% [markdown]
# <a href="https://colab.research.google.com/github/vorushin/tpuchat/blob/master/11_gpu_jax.ipynb?flush_caches=true" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 11 — GPU Lab: JAX on NVIDIA G4 (rev 3)
#
# JAX side of the JAX-vs-PyTorch pretraining comparison on a single Colab **G4**
# GPU (NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB, sm_120). Mirror
# notebook: [12_gpu_torch.ipynb](https://github.com/vorushin/tpuchat/blob/master/12_gpu_torch.ipynb).
# Ported from [08_tpu_ablations.ipynb](https://github.com/vorushin/tpuchat/blob/master/08_tpu_ablations.ipynb)
# with the head layout re-tuned for cuDNN flash attention (H=128 instead of
# TPU's H=256). Param count and FLOPs/step are **bit-identical** to the TPU
# config, so loss curves and MFU are directly comparable.
#
# ### Architecture: D=1024, N=8, K=2, H=128, F=3072, L=8, B=64, T=2048, V=32768
# | Metric | Value |
# |--------|-------|
# | Total params | 163.6M |
# | Non-embed params | 130.0M |
# | Tokens/step | 131,072 (64 × 2048, no grad accumulation) |
# | Precision | pure bf16 (params, grads, Adam moments) — same as TPU recipe |
#
# **Sections:** A — matmul peak calibration · B — component benchmarks ·
# C — quick training (300 steps, XProf) · D — profiler summary ·
# E — metrics.json export
#
# **Running via Colab CLI** (no browser needed):
# `colab new --gpu G4 -s gpu-bench && colab exec -s gpu-bench -f 11_gpu_jax.py`
# — every cell is pure Python (no `!`/`%` magics), so the jupytext `.py` runs
# as-is. Secrets: HF_TOKEN is optional (the tokenizer repo is public); see the
# secrets cell for the fallback chain.

# %%
# === Install dependencies (pure Python — safe under colab exec) ===
import importlib.util
import subprocess
import sys


def _pip(*args):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *args])


# CUDA-enabled JAX: Colab GPU runtimes usually preinstall it; if the CUDA
# plugin is missing, install before the first `import jax`. Use jax[cuda13]
# instead if the runtime ships CUDA 13 drivers only.
if importlib.util.find_spec('jax_cuda12_plugin') is None and \
        importlib.util.find_spec('jax_cuda13_plugin') is None:
    print('No JAX CUDA plugin found — installing jax[cuda12]...')
    _pip('-U', 'jax[cuda12]')

for mod, pkg in [('optax', 'optax'), ('tiktoken', 'tiktoken'),
                 ('pyarrow', 'pyarrow'), ('huggingface_hub', 'huggingface_hub'),
                 ('requests', 'requests')]:
    if importlib.util.find_spec(mod) is None:
        _pip(pkg)

# %%
# === Imports, GPU constants, dot_dict ===
import functools as ft
import json
import os
import pickle
import queue
import threading
import time
from dataclasses import dataclass

# XLA flags must be set before importing jax
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_latency_hiding_scheduler=true')

import jax
import jax.numpy as jnp
import numpy as np
import optax

# NVIDIA RTX PRO 6000 Blackwell Server Edition (Colab G4) constants
GPU_PEAK_TFLOPS_QUOTED = 960   # Colab's quoted bf16 figure — likely sparse; Section A measures dense
HBM_BW_GBS = 1600              # ~1.6 TB/s GDDR7
MEASURED_PEAK_TFLOPS = None    # set by Section A (matmul calibration)

REVISION = 3

# Persistent compile cache speeds up the exec-iterate loop on a warm runtime
try:
    jax.config.update('jax_compilation_cache_dir', '/content/jax_cache')
except Exception as e:
    print(f'Compile cache not enabled: {e}')

print(f"JAX version : {jax.__version__}")
print(f"Devices     : {jax.devices()}")
print(f"Peak TFLOPS : {GPU_PEAK_TFLOPS_QUOTED} (bf16, Colab-quoted — calibrated in Section A)")
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
# === GPU environment probe ===
# Fails fast if the runtime has no CUDA device or the wheel lacks sm_120 kernels.

try:
    smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                          '--format=csv,noheader'], capture_output=True, text=True, timeout=30)
    GPU_INFO = smi.stdout.strip()
    print(f'nvidia-smi  : {GPU_INFO}')
except Exception as e:
    GPU_INFO = 'unknown'
    print(f'nvidia-smi unavailable: {e}')

devices = jax.devices()
assert any(d.platform == 'gpu' for d in devices), \
    f'No GPU device visible to JAX: {devices}. Wrong runtime type or CPU-only jaxlib.'

# Tiny bf16 matmul — raises here (not mid-training) if kernels are missing for this arch
_a = jnp.ones((128, 128), dtype=jnp.bfloat16)
_probe = (_a @ _a).block_until_ready()
print(f'bf16 matmul probe OK on {devices[0].device_kind}')

# %% [markdown]
# ## Prerequisites
#
# Loads the tokenizer (trained in
# [01_tokenizer.ipynb](https://github.com/vorushin/tpuchat/blob/master/01_tokenizer.ipynb))
# and a small slice of FineWeb-Edu-100B-Shuffle — 2 train shards + the same val
# shard the TPU notebooks use, enough for a 300-step benchmark run.

# %%
# === Secrets + HF login + tokenizer ===
import requests
from multiprocessing import Pool
import pyarrow.parquet as pq
from huggingface_hub import login, hf_hub_download

HF_REPO_ID = 'vorushin/tpuchat'
DATA_DIR = '/content/base_data'
TOKENIZER_DIR = '/content/tokenizer'
MAX_CHARS_PER_DOC = 10_000


def get_secret(name):
    """Colab Secrets → env var → /content/secrets.json (uploaded via colab CLI) → None."""
    try:
        from google.colab import userdata
        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    if os.environ.get(name):
        return os.environ[name]
    try:
        with open('/content/secrets.json') as f:
            return json.load(f).get(name)
    except Exception:
        return None


hf_token = get_secret('HF_TOKEN')
if hf_token:
    login(token=hf_token)
else:
    print('No HF_TOKEN found — proceeding anonymously (tokenizer repo is public)')

os.makedirs(TOKENIZER_DIR, exist_ok=True)
hf_hub_download(repo_id=HF_REPO_ID, filename='tokenizer/tokenizer.pkl',
                local_dir=TOKENIZER_DIR)
print(f'Downloaded tokenizer to {TOKENIZER_DIR}')

with open(os.path.join(TOKENIZER_DIR, 'tokenizer', 'tokenizer.pkl'), 'rb') as f:
    enc = pickle.load(f)
print(f'Loaded tokenizer: vocab_size={enc.n_vocab}')

# %%
# === Download data shards ===
BASE_URL = 'https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main'
os.makedirs(DATA_DIR, exist_ok=True)

# 2 train shards (~104M tokens) cover 300 steps × 131k tokens with margin.
# Val shard 50 = first val shard of the TPU notebooks → identical eval batches.
TRAIN_SHARD_INDICES = [0, 1]
VAL_SHARD_INDICES = [50]


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


all_shard_indices = TRAIN_SHARD_INDICES + VAL_SHARD_INDICES
t0 = time.time()
with Pool(len(all_shard_indices)) as pool:
    results = pool.map(download_shard, all_shard_indices)
print(f'Downloaded {sum(results)}/{len(all_shard_indices)} shards in {time.time()-t0:.1f}s')

# %%
# === tokenize_shards + PrefetchDataLoader ===

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
# === FLOP counting helpers ===
# Dimension notation: B=batch, T=seq_len, D=d_model, N=n_heads,
#   K=n_kv_heads, H=head_dim, F=d_ff, L=n_layers, V=vocab_size

def matmul_flops(M, N, K, batch=1):
    """FLOPs for [M,K] @ [K,N].  2*M*N*K per batch element."""
    return 2 * batch * M * N * K


def attention_flops(B, N, T, H):
    """FLOPs for QK^T + AV (full T×T, not causal-halved).

    Causal kernels skip the upper triangle, so actual tensor-core work is
    ~half this — attention MFU% is overestimated by ~2x.
    """
    return 2 * (2 * B * N * T * T * H)


def layer_flops(B, T, D, N, K, H, F, mlp_type='glu'):
    """Tensor-core-relevant FLOPs for one transformer layer.

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
# === Benchmark harness ===

ALL_RESULTS = []   # global collector — every benchmark() appends here


def benchmark(fn, *args, warmup=3, repeats=10, flop_count=None,
              hbm_bytes=None, label=""):
    """Run fn repeatedly and report wall time, TFLOP/s, MFU% (vs quoted and
    measured peak), HBM bandwidth%.

    Returns dict with wall_ms, tflops, mfu_quoted_pct, mfu_measured_pct,
    hbm_bw_gbs, hbm_bw_pct.
    """
    # Warmup (triggers JIT + XLA compilation)
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    # Timed runs
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    wall_s = sum(times) / len(times)
    wall_ms = wall_s * 1000

    tflops = flop_count / (wall_s * 1e12) if flop_count else 0.0
    mfu_quoted_pct = tflops / GPU_PEAK_TFLOPS_QUOTED * 100 if flop_count else 0.0
    mfu_measured_pct = (tflops / MEASURED_PEAK_TFLOPS * 100
                        if flop_count and MEASURED_PEAK_TFLOPS else 0.0)

    hbm_bw_gbs = hbm_bytes / (wall_s * 1e9) if hbm_bytes else 0.0
    hbm_bw_pct = hbm_bw_gbs / HBM_BW_GBS * 100 if hbm_bytes else 0.0

    result = dict(label=label, wall_ms=wall_ms, tflops=tflops,
                  mfu_quoted_pct=mfu_quoted_pct, mfu_measured_pct=mfu_measured_pct,
                  hbm_bw_gbs=hbm_bw_gbs, hbm_bw_pct=hbm_bw_pct)
    ALL_RESULTS.append(result)

    mfu_q = f"{mfu_quoted_pct:5.1f}%" if flop_count else "  n/a"
    mfu_m = f"{mfu_measured_pct:5.1f}%" if flop_count and MEASURED_PEAK_TFLOPS else "  n/a"
    tflop_str = f"{tflops:6.1f}" if flop_count else "   n/a"
    print(f"  {label:<40s}  {wall_ms:8.2f} ms  {tflop_str} TFLOP/s  "
          f"MFU {mfu_q} (quoted) {mfu_m} (measured)")
    return result


def print_summary(results):
    """Print a formatted comparison table from benchmark results."""
    print(f"\n  {'Label':<40s}  {'Wall ms':>8s}  {'TFLOP/s':>8s}  "
          f"{'MFU%q':>6s}  {'MFU%m':>6s}")
    print("  " + "-" * 78)
    for r in results:
        tf = f"{r['tflops']:7.1f}" if r['tflops'] > 0 else "    n/a"
        mq = f"{r['mfu_quoted_pct']:5.1f}%" if r['tflops'] > 0 else "  n/a"
        mm = f"{r['mfu_measured_pct']:5.1f}%" if r['mfu_measured_pct'] > 0 else "  n/a"
        print(f"  {r['label']:<40s}  {r['wall_ms']:8.2f}  {tf}  {mq:>6s}  {mm:>6s}")
    print()

# %% [markdown]
# ## Section A — Matmul peak calibration
#
# Colab quotes 960 bf16 TFLOPs for the G4; NVIDIA marketing figures for
# Blackwell are often sparse (2:4 sparsity) numbers, which would put the dense
# peak near 480. We measure the achievable dense bf16 peak with large matmuls
# and report every MFU number against **both** the quoted and the measured peak.

# %%
# === Section A: matmul peak calibration ===

def _matmul_bench(M, N, K, label):
    a = jax.random.normal(jax.random.key(0), (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(jax.random.key(1), (K, N), dtype=jnp.bfloat16)
    fn = jax.jit(lambda a, b: a @ b)
    return benchmark(fn, a, b, flop_count=matmul_flops(M, N, K), label=label)


CALIBRATION_RESULTS = []
print('Square matmuls:')
for dim in [2048, 4096, 8192, 16384]:
    r = _matmul_bench(dim, dim, dim, f'matmul {dim}x{dim}x{dim} bf16')
    CALIBRATION_RESULTS.append({'shape': [dim, dim, dim], 'dtype': 'bf16',
                                'tflops': r['tflops']})

print('Model-shaped matmuls (B*T=131072 tokens):')
for (M, N, K, lbl) in [(131072, 3072, 1024, 'matmul BTxDxF (mlp up)'),
                       (131072, 32768, 1024, 'matmul BTxDxV (lm_head)')]:
    r = _matmul_bench(M, N, K, lbl)
    CALIBRATION_RESULTS.append({'shape': [M, N, K], 'dtype': 'bf16',
                                'tflops': r['tflops']})

MEASURED_PEAK_TFLOPS = max(c['tflops'] for c in CALIBRATION_RESULTS)
print(f'\nQuoted peak  : {GPU_PEAK_TFLOPS_QUOTED} TFLOP/s (Colab announcement)')
print(f'Measured peak: {MEASURED_PEAK_TFLOPS:.0f} TFLOP/s (best dense bf16 matmul)')
print(f'Ratio        : {MEASURED_PEAK_TFLOPS / GPU_PEAK_TFLOPS_QUOTED:.2f} '
      f'(≈0.5 would mean the quoted figure is sparse)')

# %% [markdown]
# ## Model
#
# Same architecture as the TPU ablation lab: RoPE, RMSNorm, GQA, QK-norm,
# logit softcap, SwiGLU MLP, AdamW. Two differences for GPU:
# head layout is N=8, K=2, H=128 (identical param count and FLOPs to the TPU's
# N=4, K=1, H=256 since N·H and K·H are unchanged), and attention runs through
# cuDNN flash attention (`jax.nn.dot_product_attention`) instead of the TPU
# splash kernel. Tensors use BTNH layout — what cuDNN expects natively.

# %%
# === Config ===

@jax.tree_util.register_static
@dataclass(kw_only=True, frozen=True)
class Config:
    # ── Ablation knobs ─────────────────────────────────────────
    attn_impl: str = 'cudnn'        # 'cudnn' | 'einsum'
    mlp_type: str = 'glu'           # 'glu' (SwiGLU, F=3072) | 'plain' (ReLU², F=4096)
    qk_norm: bool = True            # QK-norm on queries and keys

    # ── Architecture (D1024, 130M non-embed) ───────────────────
    n_embd: int = 1024
    n_layer: int = 8
    seq_len: int = 2048
    vocab_size: int = 32768
    n_head: int = 8
    n_kv_head: int = 2
    head_dim: int = 128
    mlp_dim: int = 3072             # 3072 for glu, 4096 for plain
    softcap: float = 15.0
    logit_dtype: str = 'bf16'       # 'bf16' or 'fp32' — fp32 is ~33% slower on G4
    num_lm_head_chunks: int = 1     # chunking exists for 32 GB TPUs; on 96 GB it costs ~7% step time
    batch_size: int = 64
    microbatch_size: int = 64       # == batch_size → no gradient accumulation

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
assert config.batch_size == config.microbatch_size, \
    'GPU notebooks run without gradient accumulation (96 GB HBM fits the full batch)'
print(f'Config: D={config.n_embd}, L={config.n_layer}, T={config.seq_len}, '
      f'V={config.vocab_size}, N={config.n_head}, K={config.n_kv_head}, '
      f'H={config.head_dim}, F={config.mlp_dim}')
print(f'Ablations: attn_impl={config.attn_impl}, mlp_type={config.mlp_type}, '
      f'qk_norm={config.qk_norm}')
print(f'Training: lr={config.learning_rate:.1e}, B={config.batch_size} (no accumulation)')

# %%
# === Model: RMSNorm, RoPE, attention, layers, init ===

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
    """Apply rotary embeddings. x: (B, T, N, H), cos/sin: (1, T, 1, H/2)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1)


def _expand_kv(k, v, n_head, n_kv_head):
    """Repeat KV heads (axis 2 of BTKH) to match Q head count for the einsum backend."""
    if n_kv_head == n_head:
        return k, v
    ratio = n_head // n_kv_head
    return jnp.repeat(k, ratio, axis=2), jnp.repeat(v, ratio, axis=2)


def attention(config, q, k, v):
    """Causal attention. q: (B,T,N,H), k/v: (B,T,K,H) → (B,T,N,H)."""
    if config.attn_impl == 'cudnn':
        # cuDNN flash attention — handles GQA natively, never materializes scores
        return jax.nn.dot_product_attention(q, k, v, is_causal=True,
                                            implementation='cudnn')
    elif config.attn_impl == 'einsum':
        k_exp, v_exp = _expand_kv(k, v, config.n_head, config.n_kv_head)
        seq_len = q.shape[1]
        scale = config.head_dim ** -0.5
        scores = jnp.einsum('btnh,bsnh->bnts', q, k_exp) * scale
        rows = jnp.arange(seq_len)[:, None]
        cols = jnp.arange(seq_len)[None, :]
        mask = cols <= rows
        scores = jnp.where(mask[None, None, :, :], scores,
                           jnp.finfo(scores.dtype).min)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum('bnts,bsnh->btnh', attn_weights, v_exp)
    else:
        raise ValueError(f'Unknown attn_impl: {config.attn_impl}')


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
    """Forward pass for one transformer layer. x: (B,T,D), BTNH attention layout."""
    h = rms_norm(x)

    with jax.named_scope(f'layer_{layer_idx}/attention'):
        q = jnp.einsum('btd,dnh->btnh', h, layer.c_q)
        k = jnp.einsum('btd,dnh->btnh', h, layer.c_k)
        v = jnp.einsum('btd,dnh->btnh', h, layer.c_v)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if config.qk_norm:
            q = rms_norm(q)
            k = rms_norm(k)

        attn_out = attention(config, q, k, v)
        attn_out = jnp.einsum('btnh,nhd->btd', attn_out, layer.c_proj)

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
    cos = cos[None, :, None, :]     # (1, T, 1, H/2) — broadcasts over BTNH
    sin = sin[None, :, None, :]
    with jax.named_scope('embedding'):
        x = rms_norm(params.wte[tokens])
    for i in range(config.n_layer):
        x = single_layer_forward(config, params.layers[i], x, cos, sin, layer_idx=i)
    return rms_norm(x)


def count_params(params):
    """Count total parameters."""
    return sum(p.size for p in jax.tree.leaves(params) if isinstance(p, jax.Array))


def count_non_embed_params(params):
    """Non-embedding params (unembed + layers). Excludes wte (lookup table)."""
    return count_params(params) - params.wte.size

# %%
# === Chunked LM head loss, train/eval/predict steps, optimizer ===

def _logit_dtype(config):
    return jnp.float32 if config.logit_dtype == 'fp32' else jnp.bfloat16


# Chunked LM head loss, reduces HBM usage by the last matmul (hidden_dim -> vocab_dim).
# Extracted from maxtext. On the 96 GB G4 chunking is unnecessary and the scan
# costs ~7% step time (517ms @ 8 chunks vs 481ms @ 1) — default is 1 chunk.


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


def loss_fn(config, params, x, y):
    hidden = model_forward(config, params, x)
    return chunked_lm_head_loss(hidden, params.lm_head, y, config)


def make_train_step(optimizer, donate=True):
    """Create a JIT-compiled train step (no gradient accumulation on GPU).

    donate=True frees the old params/opt_state buffers and lets XLA update
    in place. Use donate=False for benchmarking (repeated calls reuse inputs).
    """
    donate_argnums = (1, 2) if donate else ()

    @ft.partial(jax.jit, donate_argnums=donate_argnums)
    def train_step(config, params, opt_state, x, y, _opt=optimizer):
        with jax.named_scope('forward_backward'):
            loss, grads = jax.value_and_grad(
                lambda p: loss_fn(config, p, x, y))(params)

        with jax.named_scope('optimizer'):
            updates, new_opt_state = _opt.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

        return loss, new_params, new_opt_state
    return train_step


@jax.jit
def eval_step(config, params, x, y):
    """JIT-compiled eval: returns loss for a single batch."""
    return loss_fn(config, params, x, y)


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
# ## Section B — Component benchmarks
#
# Times each piece of the training step in isolation: the attention op, one
# transformer layer, the full model, the optimizer, and the assembled train
# step. Labels are string-identical to 12_gpu_torch so `gpu/compare.py` can
# join the two result sets. Includes a cudnn-vs-einsum numerical parity check.

# %%
# === Section B: correctness check (cudnn vs einsum attention) ===

_check_cfg_cudnn = Config(attn_impl='cudnn')
_check_cfg_einsum = Config(attn_impl='einsum')
_B_check = 4    # small batch: einsum path materializes (B,N,T,T) scores

_key = jax.random.key(0)
_kq, _kk, _kv = jax.random.split(_key, 3)
_q = jax.random.normal(_kq, (_B_check, config.seq_len, config.n_head, config.head_dim),
                       dtype=jnp.bfloat16)
_k = jax.random.normal(_kk, (_B_check, config.seq_len, config.n_kv_head, config.head_dim),
                       dtype=jnp.bfloat16)
_v = jax.random.normal(_kv, (_B_check, config.seq_len, config.n_kv_head, config.head_dim),
                       dtype=jnp.bfloat16)

_out_cudnn = jax.jit(lambda q, k, v: attention(_check_cfg_cudnn, q, k, v))(_q, _k, _v)
_out_einsum = jax.jit(lambda q, k, v: attention(_check_cfg_einsum, q, k, v))(_q, _k, _v)
_max_diff = float(jnp.max(jnp.abs(_out_cudnn.astype(jnp.float32)
                                  - _out_einsum.astype(jnp.float32))))
print(f'cudnn vs einsum attention max abs diff: {_max_diff:.4f} (bf16 tolerance ~0.06)')
assert _max_diff < 0.0625, 'cudnn attention diverges from einsum reference'

# %%
# === Section B: component benchmarks ===

B, T = config.batch_size, config.seq_len
D, N, K, H, F = (config.n_embd, config.n_head, config.n_kv_head,
                 config.head_dim, config.mlp_dim)

fwd_flops = (config.n_layer * layer_flops(B, T, D, N, K, H, F, config.mlp_type)
             + matmul_flops(B * T, config.vocab_size, D))
step_flops = 3 * fwd_flops  # fwd + 2x bwd

bench_params = init_full_model(config, seed=config.param_seed)
bench_layer = bench_params.layers[0]
key = jax.random.key(0)
kq, kk, kv, kx, kt = jax.random.split(key, 5)
q_full = jax.random.normal(kq, (B, T, N, H), dtype=jnp.bfloat16)
k_full = jax.random.normal(kk, (B, T, K, H), dtype=jnp.bfloat16)
v_full = jax.random.normal(kv, (B, T, K, H), dtype=jnp.bfloat16)
x_hidden = jax.random.normal(kx, (B, T, D), dtype=jnp.bfloat16)
tokens = jax.random.randint(kt, (B, T), 0, config.vocab_size, dtype=jnp.int32)
labels = jax.random.randint(kt, (B, T), 0, config.vocab_size, dtype=jnp.int32)
cos_b, sin_b = precompute_rope(T, H)
cos_b, sin_b = cos_b[None, :, None, :], sin_b[None, :, None, :]

SECTION_B_RESULTS = []
attn_fl = attention_flops(B, N, T, H)
layer_fl = layer_flops(B, T, D, N, K, H, F, config.mlp_type)

# -- attention op --
attn_fwd = jax.jit(lambda q, k, v: attention(config, q, k, v))
SECTION_B_RESULTS.append(benchmark(
    attn_fwd, q_full, k_full, v_full,
    flop_count=attn_fl, label='attention fwd'))

attn_fwd_bwd = jax.jit(jax.grad(
    lambda q, k, v: attention(config, q, k, v).astype(jnp.float32).sum(),
    argnums=(0, 1, 2)))
SECTION_B_RESULTS.append(benchmark(
    attn_fwd_bwd, q_full, k_full, v_full,
    flop_count=3 * attn_fl, label='attention fwd+bwd'))

# -- single layer --
layer_fwd = jax.jit(lambda lp, x: single_layer_forward(config, lp, x, cos_b, sin_b))
SECTION_B_RESULTS.append(benchmark(
    layer_fwd, bench_layer, x_hidden,
    flop_count=layer_fl, label='layer fwd'))

layer_fwd_bwd = jax.jit(jax.grad(
    lambda lp, x: single_layer_forward(config, lp, x, cos_b, sin_b)
                  .astype(jnp.float32).sum(), argnums=(0, 1)))
SECTION_B_RESULTS.append(benchmark(
    layer_fwd_bwd, bench_layer, x_hidden,
    flop_count=3 * layer_fl, label='layer fwd+bwd'))

# -- full model --
model_fwd = jax.jit(lambda p, t: model_forward(config, p, t))
SECTION_B_RESULTS.append(benchmark(
    model_fwd, bench_params, tokens,
    flop_count=config.n_layer * layer_fl, label='model fwd'))

model_fwd_bwd = jax.jit(jax.value_and_grad(
    lambda p: loss_fn(config, p, tokens, labels)))
SECTION_B_RESULTS.append(benchmark(
    model_fwd_bwd, bench_params,
    flop_count=step_flops, label='model fwd+bwd (loss+grads)'))

# -- optimizer step alone (fake grads) --
NUM_QUICK_STEPS = 300
bench_opt = make_optimizer(config, NUM_QUICK_STEPS)
bench_opt_state = bench_opt.init(bench_params)
fake_grads = jax.tree.map(lambda p: jnp.ones_like(p) * 1e-4, bench_params)

@jax.jit
def opt_only(g, s, p):
    updates, new_s = bench_opt.update(g, s, p)
    return optax.apply_updates(p, updates), new_s   # compute + apply, like torch opt.step()


SECTION_B_RESULTS.append(benchmark(
    opt_only, fake_grads, bench_opt_state, bench_params,
    label='optimizer step'))

# -- full train step (no donation so repeated calls reuse inputs) --
bench_train_step = make_train_step(bench_opt, donate=False)
SECTION_B_RESULTS.append(benchmark(
    bench_train_step, config, bench_params, bench_opt_state, tokens, labels,
    flop_count=step_flops, label='train step'))

print_summary(SECTION_B_RESULTS)

# Free benchmark buffers before training
del bench_params, bench_opt_state, fake_grads, q_full, k_full, v_full, x_hidden

# %% [markdown]
# ## Section C — Quick Training (XProf)
#
# Trains for 300 steps — outputs MFU and throughput, captures an XProf trace on
# steps 15-20 (perfetto format, parsed in Section D). Check that the loss goes
# down and that the trace shows tightly packed device execution (compute-bound)
# rather than gaps (input-bound).

# %%
# === Section C: quick training ===

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
train_step = make_train_step(optimizer, donate=True)

# Data
raw_train = tokenize_shards(TRAIN_SHARD_INDICES, config.batch_size, config.seq_len)
train_loader = PrefetchDataLoader(raw_train, capacity=4)
val_loader_fn = lambda: tokenize_shards(VAL_SHARD_INDICES, config.batch_size, config.seq_len)

smooth_loss = 0.0
mfu_t0 = None
mfu_tokens = 0
mfu_eval_time = 0.0
report_t0 = time.time()
report_tokens = 0
report_eval_time = 0.0
compile_s = None
step_times = []       # per-step wall time, post-warmup
tok_s_series = []     # (step, tok/s) every 50 steps
val_losses_series = []

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
        val_losses_series.append({'step': step, 'loss': avg_val_loss})
        print(f'step {step:05d} | Val loss: {avg_val_loss:.4f}')

    if last_step:
        break

    # --- XProf ---
    if step == XPROF_START:
        jax.profiler.start_trace(LOG_DIR, create_perfetto_trace=True)
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

    if step == 0:
        compile_s = dt  # first call includes JIT compilation

    if step > XPROF_END:
        if mfu_t0 is None:
            mfu_t0 = time.time()
        mfu_tokens += config.batch_size * config.seq_len
        step_times.append(dt)
    report_tokens += config.batch_size * config.seq_len

    loss_val = float(loss)
    ema_beta = 0.9
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
    debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))

    if step % 50 == 0:
        report_wall = time.time() - report_t0 - report_eval_time
        tok_per_sec = int(report_tokens / report_wall) if report_wall > 0 else 0
        tok_s_series.append({'step': step, 'tok_s': tok_per_sec})
        print(f'step {step:05d}/{NUM_QUICK_STEPS} | loss: {debiased_loss:.4f} '
              f'| tok/s: {tok_per_sec:,}')
        report_t0 = time.time()
        report_tokens = 0
        report_eval_time = 0.0

train_loader.stop()

# --- MFU report (wall clock excl. eval, steps 21-300) ---
TRAINING_METRICS = {}
if mfu_t0 is not None:
    mfu_wall = time.time() - mfu_t0 - mfu_eval_time
    tok_per_s = int(mfu_tokens / mfu_wall)
    flops_per_tok = step_flops / (config.batch_size * config.seq_len)
    mfu_quoted = (tok_per_s * flops_per_tok) / (GPU_PEAK_TFLOPS_QUOTED * 1e12) * 100
    mfu_measured = (tok_per_s * flops_per_tok) / (MEASURED_PEAK_TFLOPS * 1e12) * 100
    step_ms_median = sorted(step_times)[len(step_times) // 2] * 1000
    print(f'\nMFU: {mfu_quoted:.1f}% (vs quoted {GPU_PEAK_TFLOPS_QUOTED}) | '
          f'{mfu_measured:.1f}% (vs measured {MEASURED_PEAK_TFLOPS:.0f}) | '
          f'tok/s: {tok_per_s:,} | median step: {step_ms_median:.1f}ms | '
          f'compile: {compile_s:.1f}s')
    TRAINING_METRICS = {
        'num_steps': NUM_QUICK_STEPS,
        'tokens_per_step': config.batch_size * config.seq_len,
        'compile_s': round(compile_s, 2),
        'tok_per_sec': tok_per_s,
        'step_ms_median': round(step_ms_median, 2),
        'mfu_quoted_pct': round(mfu_quoted, 2),
        'mfu_measured_pct': round(mfu_measured, 2),
        'tok_s_series': tok_s_series,
        'val_losses': val_losses_series,
        'final_train_loss_ema': round(debiased_loss, 4),
    }

# --- Sample text ---
print('\n--- Samples ---')
for prompt in ['The capital of France is', 'Machine learning is']:
    text = generate(config, params, enc, prompt, max_new_tokens=64)
    print(f'Prompt: {prompt}\nOutput: {text}\n')

# %% [markdown]
# ## Section D — Profiler summary
#
# Pure-Python replacement for the TensorBoard cell: parses the perfetto/chrome
# trace captured in Section C, prints the top-25 GPU kernels by total time, and
# tars the trace directory for download via `colab download`.

# %%
# === Section D: profiler summary + trace tarball ===
import collections
import glob
import gzip


def parse_trace(log_dir, top_n=25):
    """Aggregate device-side trace events by kernel name from an XProf trace."""
    patterns = ['**/*.trace.json.gz', '**/perfetto_trace.json.gz', '**/*.trace.json']
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(log_dir, pat), recursive=True))
    if not paths:
        print(f'No trace files found under {log_dir}')
        return []
    path = sorted(paths)[-1]
    print(f'Parsing {path}')
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        trace = json.load(f)
    events = trace.get('traceEvents', trace if isinstance(trace, list) else [])

    # Identify device (GPU) processes from metadata events
    proc_names = {}
    for e in events:
        if e.get('ph') == 'M' and e.get('name') == 'process_name':
            proc_names[e.get('pid')] = e.get('args', {}).get('name', '')
    device_pids = {pid for pid, name in proc_names.items()
                   if any(s in name.lower() for s in ('gpu', 'device', 'stream'))}

    totals = collections.Counter()
    total_dur = 0.0
    for e in events:
        if e.get('ph') != 'X':
            continue
        if device_pids and e.get('pid') not in device_pids:
            continue
        dur = e.get('dur', 0)
        totals[e.get('name', '?')] += dur
        total_dur += dur

    top = totals.most_common(top_n)
    print(f'\n  {"Kernel":<64s}  {"Total ms":>9s}  {"%":>6s}')
    print('  ' + '-' * 84)
    result = []
    for name, dur in top:
        pct = 100 * dur / total_dur if total_dur else 0
        print(f'  {name[:64]:<64s}  {dur/1000:9.2f}  {pct:5.1f}%')
        result.append({'name': name, 'total_ms': round(dur / 1000, 3),
                       'pct_of_captured': round(pct, 2)})
    return result


TOP_KERNELS = parse_trace(LOG_DIR)

TRACE_TARBALL = '/content/trace_jax.tar.gz'
subprocess.run(['tar', '-czf', TRACE_TARBALL, '-C', '/content', 'log_dir'],
               check=True)
print(f'\nTrace tarball: {TRACE_TARBALL} '
      f'({os.path.getsize(TRACE_TARBALL) / 1e6:.1f} MB)')

# %% [markdown]
# ## Section E — metrics.json export
#
# Machine-readable results for `colab download` + `gpu/compare.py`.

# %%
# === Section E: metrics.json writer ===
import datetime

metrics = {
    'framework': 'jax',
    'notebook': '11_gpu_jax',
    'revision': REVISION,
    'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'env': {
        'gpu_name': GPU_INFO,
        'vram_gb': 96,
        'framework_version': jax.__version__,
        'python': sys.version.split()[0],
        'attn_impl': config.attn_impl,
        'compile_mode': None,
        'xla_flags': os.environ.get('XLA_FLAGS', ''),
        'notes': [],
    },
    'config': {k: getattr(config, k) for k in Config.__dataclass_fields__},
    'flops': {
        'fwd_flops': fwd_flops,
        'step_flops': step_flops,
        'flops_per_token': step_flops // (config.batch_size * config.seq_len),
    },
    'calibration': {
        'quoted_peak_tflops': GPU_PEAK_TFLOPS_QUOTED,
        'measured_peak_tflops': round(MEASURED_PEAK_TFLOPS, 1),
        'matmuls': [{**c, 'tflops': round(c['tflops'], 1)} for c in CALIBRATION_RESULTS],
    },
    'components': [
        {'label': r['label'], 'wall_ms': round(r['wall_ms'], 3),
         'tflops': round(r['tflops'], 1),
         'mfu_quoted_pct': round(r['mfu_quoted_pct'], 2),
         'mfu_measured_pct': round(r['mfu_measured_pct'], 2)}
        for r in SECTION_B_RESULTS
    ],
    'training': TRAINING_METRICS,
    'profile': {
        'top_kernels': TOP_KERNELS,
        'trace_file': os.path.basename(TRACE_TARBALL),
    },
}

with open('/content/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2, default=str)
print('Wrote /content/metrics.json')
print(json.dumps({k: metrics[k] for k in ['framework', 'training']}, indent=2, default=str))
