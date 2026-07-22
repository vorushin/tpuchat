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
# <a href="https://colab.research.google.com/github/vorushin/tpuchat/blob/master/12_gpu_torch.ipynb?flush_caches=true" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 12 — GPU Lab: PyTorch on NVIDIA G4 (rev 3)
#
# PyTorch side of the JAX-vs-PyTorch pretraining comparison on a single Colab
# **G4** GPU (NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB, sm_120).
# Mirror notebook: [11_gpu_jax.ipynb](https://github.com/vorushin/tpuchat/blob/master/11_gpu_jax.ipynb)
# — a line-by-line port: same parameter shapes, same einsum equations, same
# init distributions, same pure-bf16 precision recipe (bf16 params, grads, and
# Adam moments — no autocast, no fp32 master weights), same data pipeline.
# PyTorch idioms: `torch.compile`, SDPA (cuDNN flash attention backend),
# fused AdamW.
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
# C — quick training (300 steps, torch.profiler) · D — profiler summary ·
# E — metrics.json export
#
# **Running via Colab CLI** (no browser needed):
# `colab new --gpu G4 -s gpu-bench && colab exec -s gpu-bench -f 12_gpu_torch.py`
# — every cell is pure Python (no `!`/`%` magics), so the jupytext `.py` runs
# as-is. Secrets: HF_TOKEN is optional (the tokenizer repo is public); see the
# secrets cell for the fallback chain.

# %%
# === Install dependencies (pure Python — safe under colab exec) ===
import importlib.util
import subprocess
import sys
from importlib import metadata as _metadata


def _pip(*args):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *args])


# Blackwell (sm_120) needs torch >= 2.7 built against CUDA 12.8+. Check the
# wheel's CUDA tag BEFORE importing torch so an upgrade doesn't need a restart.
try:
    _torch_ver = _metadata.version('torch')
except _metadata.PackageNotFoundError:
    _torch_ver = None

if _torch_ver is None:
    print('torch not found — installing cu128 wheel...')
    _pip('torch', '--index-url', 'https://download.pytorch.org/whl/cu128')
elif '+cu' in _torch_ver and int(_torch_ver.split('+cu')[1]) < 128:
    print(f'torch {_torch_ver} predates Blackwell support — upgrading to cu128...')
    _pip('-U', 'torch', '--index-url', 'https://download.pytorch.org/whl/cu128')
else:
    print(f'torch {_torch_ver} present')

for mod, pkg in [('tiktoken', 'tiktoken'), ('pyarrow', 'pyarrow'),
                 ('huggingface_hub', 'huggingface_hub'), ('requests', 'requests')]:
    if importlib.util.find_spec(mod) is None:
        _pip(pkg)

# %%
# === Imports, GPU constants ===
import contextlib
import json
import math
import os
import pickle
import queue
import threading
import time
from dataclasses import dataclass

# Persistent inductor cache speeds up the exec-iterate loop on a warm runtime
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/content/inductor_cache')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# NVIDIA RTX PRO 6000 Blackwell Server Edition (Colab G4) constants
GPU_PEAK_TFLOPS_QUOTED = 960   # Colab's quoted bf16 figure — likely sparse; Section A measures dense
HBM_BW_GBS = 1600              # ~1.6 TB/s GDDR7
MEASURED_PEAK_TFLOPS = None    # set by Section A (matmul calibration)

REVISION = 3

# TF32 for any stray fp32 matmuls (mirrors XLA's default on the JAX side)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

print(f"torch version: {torch.__version__} (CUDA {torch.version.cuda})")
print(f"Peak TFLOPS  : {GPU_PEAK_TFLOPS_QUOTED} (bf16, Colab-quoted — calibrated in Section A)")
print(f"Notebook rev : {REVISION}")

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

assert torch.cuda.is_available(), 'No CUDA device visible to torch. Wrong runtime type?'
arch_list = torch.cuda.get_arch_list()
print(f'Arch list   : {arch_list}')
assert any('120' in a for a in arch_list), \
    f'torch wheel has no sm_120 kernels ({arch_list}) — install a cu128+ build and restart'

# Tiny bf16 matmul — raises here (not mid-training) if kernels are missing for this arch
_a = torch.ones(128, 128, dtype=torch.bfloat16, device='cuda')
_probe = _a @ _a
torch.cuda.synchronize()
print(f'bf16 matmul probe OK on {torch.cuda.get_device_name(0)}')

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


def _to_device(arr):
    """int32 numpy → pinned int64 tensor → async H2D copy (torch wants long indices)."""
    t = torch.from_numpy(arr.astype(np.int64))
    return t.pin_memory().to('cuda', non_blocking=True)


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
                item = (_to_device(x), _to_device(y))
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
    measured peak), HBM bandwidth%. Mirrors the JAX harness: warmup absorbs
    compilation, each timed call syncs the device.
    """
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
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
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    fn = lambda: a @ b
    return benchmark(fn, flop_count=matmul_flops(M, N, K), label=label)


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
# Line-by-line port of the JAX model: parameter shapes `c_q (D,N,H)`,
# `c_proj (N,H,D)`, `w_gate/w_up (D,F)`, `w_down (F,D)`, the same einsum
# equations, parameter-free RMSNorm, RoPE then QK-norm, GQA (N=8, K=2),
# SwiGLU, logit softcap. Attention runs through
# `F.scaled_dot_product_attention` with the backend pinned via `sdpa_kernel`
# (cuDNN flash attention by default). The whole model lives in bf16; the
# `math` attention impl mirrors the JAX `einsum` reference path.

# %%
# === Config ===

@dataclass(kw_only=True, frozen=True)
class Config:
    # ── Ablation knobs ─────────────────────────────────────────
    attn_impl: str = 'sdpa_cudnn'   # 'sdpa_cudnn' | 'sdpa_flash' | 'math'
    compile_mode: str = 'default'   # 'none' | 'default' | 'max-autotune'
                                    # max-autotune: -0.8% step time but +93s warmup,
                                    # and crashed (CUDA illegal memory access) with
                                    # the unchunked CE fusion on sm_120 — not default
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
    logit_dtype: str = 'bf16'       # 'bf16' or 'fp32'
    num_lm_head_chunks: int = 1     # torch is insensitive to chunking (Inductor fuses
                                    # the loop); 1 matches the JAX notebook's default
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
    'GPU notebooks run without gradient accumulation (96 GB VRAM fits the full batch)'
print(f'Config: D={config.n_embd}, L={config.n_layer}, T={config.seq_len}, '
      f'V={config.vocab_size}, N={config.n_head}, K={config.n_kv_head}, '
      f'H={config.head_dim}, F={config.mlp_dim}')
print(f'Ablations: attn_impl={config.attn_impl}, compile_mode={config.compile_mode}, '
      f'mlp_type={config.mlp_type}, qk_norm={config.qk_norm}')
print(f'Training: lr={config.learning_rate:.1e}, B={config.batch_size} (no accumulation)')

# %%
# === Model: RMSNorm, RoPE, attention, layers, init ===

def rms_norm(x):
    """RMSNorm with no learnable parameters."""
    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6)


def precompute_rope(seq_len, head_dim, base=10000):
    """Precompute rotary embedding cos/sin tables."""
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs).to(torch.bfloat16)
    sin = torch.sin(freqs).to(torch.bfloat16)
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary embeddings. x: (B, T, N, H), cos/sin: (1, T, 1, H/2)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def _expand_kv(k, v, n_head, n_kv_head):
    """Repeat KV heads (dim 1 of BNTH) to match Q head count for the math backend."""
    if n_kv_head == n_head:
        return k, v
    ratio = n_head // n_kv_head
    return (k.repeat_interleave(ratio, dim=1),
            v.repeat_interleave(ratio, dim=1))


def attn_ctx(config):
    """SDPA backend pin — applied at call sites, outside torch.compile regions."""
    if config.attn_impl == 'sdpa_cudnn':
        return sdpa_kernel([SDPBackend.CUDNN_ATTENTION])
    if config.attn_impl == 'sdpa_flash':
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION])
    return contextlib.nullcontext()


def attention(config, q, k, v):
    """Causal attention. q: (B,T,N,H), k/v: (B,T,K,H) → (B,T,N,H)."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # → (B,N,T,H) / (B,K,T,H)
    if config.attn_impl in ('sdpa_cudnn', 'sdpa_flash'):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             enable_gqa=True)
    elif config.attn_impl == 'math':
        k, v = _expand_kv(k, v, config.n_head, config.n_kv_head)
        seq_len = q.shape[2]
        scale = config.head_dim ** -0.5
        scores = torch.einsum('bnth,bnsh->bnts', q, k) * scale
        rows = torch.arange(seq_len, device=q.device)[:, None]
        cols = torch.arange(seq_len, device=q.device)[None, :]
        mask = cols <= rows
        scores = torch.where(mask[None, None, :, :], scores,
                             torch.finfo(scores.dtype).min)
        attn_weights = scores.softmax(dim=-1)
        out = torch.einsum('bnts,bnsh->bnth', attn_weights, v)
    else:
        raise ValueError(f'Unknown attn_impl: {config.attn_impl}')
    return out.transpose(1, 2)


class TransformerLayer(nn.Module):
    """One transformer layer — parameter shapes match the JAX notebook exactly."""

    def __init__(self, config, generator):
        super().__init__()
        self.config = config
        D, N, K, H = config.n_embd, config.n_head, config.n_kv_head, config.head_dim
        F_mlp = config.mlp_dim
        s = math.sqrt(3.0) / math.sqrt(D)

        def uniform(*shape):
            w = torch.empty(*shape, dtype=torch.bfloat16)
            w.uniform_(-s, s, generator=generator)
            return nn.Parameter(w)

        # Attention projections
        self.c_q = uniform(D, N, H)
        self.c_k = uniform(D, K, H)
        self.c_v = uniform(D, K, H)
        self.c_proj = nn.Parameter(torch.zeros(N, H, D, dtype=torch.bfloat16))

        # MLP — shape depends on mlp_type
        if config.mlp_type == 'glu':
            # SwiGLU: gate (D,F) + up (D,F) + down (F,D)
            self.w_gate = uniform(D, F_mlp)
            self.w_up = uniform(D, F_mlp)
            self.w_down = nn.Parameter(torch.zeros(F_mlp, D, dtype=torch.bfloat16))
        else:
            # Plain (ReLU²): up (D,F) + down (F,D)
            self.w_up = uniform(D, F_mlp)
            self.w_down = nn.Parameter(torch.zeros(F_mlp, D, dtype=torch.bfloat16))

    def forward(self, x, cos, sin):
        config = self.config
        h = rms_norm(x)

        q = torch.einsum('btd,dnh->btnh', h, self.c_q)
        k = torch.einsum('btd,dnh->btnh', h, self.c_k)
        v = torch.einsum('btd,dnh->btnh', h, self.c_v)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if config.qk_norm:
            q = rms_norm(q)
            k = rms_norm(k)

        attn_out = attention(config, q, k, v)
        attn_out = torch.einsum('btnh,nhd->btd', attn_out, self.c_proj)

        x = x + attn_out

        h2 = rms_norm(x)
        if config.mlp_type == 'glu':
            gate = F.silu(torch.einsum('btd,dh->bth', h2, self.w_gate))
            up = torch.einsum('btd,dh->bth', h2, self.w_up)
            mlp_out = torch.einsum('bth,hd->btd', gate * up, self.w_down)
        else:  # plain (ReLU²)
            mlp_out = torch.einsum('btd,dh->bth', h2, self.w_up)
            mlp_out = F.relu(mlp_out) ** 2
            mlp_out = torch.einsum('bth,hd->btd', mlp_out, self.w_down)

        x = x + mlp_out
        return x


class GPT(nn.Module):
    """Embed → layers → final norm. Returns hidden (B,T,D); loss is computed
    separately by chunked_lm_head_loss (same split as the JAX notebook)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        g = torch.Generator().manual_seed(config.param_seed)
        self.wte = nn.Parameter(
            torch.randn(config.vocab_size, config.n_embd,
                        generator=g, dtype=torch.bfloat16))
        self.lm_head = nn.Parameter(
            torch.randn(config.n_embd, config.vocab_size,
                        generator=g, dtype=torch.bfloat16) * 0.001)
        self.layers = nn.ModuleList(
            [TransformerLayer(config, g) for _ in range(config.n_layer)])
        cos, sin = precompute_rope(config.seq_len, config.head_dim)
        self.register_buffer('cos', cos[None, :, None, :], persistent=False)  # (1,T,1,H/2)
        self.register_buffer('sin', sin[None, :, None, :], persistent=False)

    def forward(self, tokens):
        T = tokens.shape[1]
        cos, sin = self.cos[:, :T], self.sin[:, :T]
        x = rms_norm(self.wte[tokens])
        for layer in self.layers:
            x = layer(x, cos, sin)
        return rms_norm(x)


def count_params(model):
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def count_non_embed_params(model):
    """Non-embedding params (unembed + layers). Excludes wte (lookup table)."""
    return count_params(model) - model.wte.numel()

# %%
# === Chunked LM head loss, train/eval/predict steps, optimizer ===

def chunked_lm_head_loss(hidden, lm_head, labels, config):
    """LM head + softcap + CE in vocab-projection chunks (mirrors the JAX
    custom_vjp version — autograd provides the backward here). Note: torch's
    CE kernel accumulates in fp32 internally vs optax's bf16 log_softmax —
    a documented (negligible) asymmetry."""
    B, T, D = hidden.shape
    N = config.num_lm_head_chunks
    S = B * T // N
    hidden_chunks = hidden.reshape(N, S, D)
    labels_chunks = labels.reshape(N, S)

    total = None
    for i in range(N):
        logits = torch.einsum('td,dv->tv', hidden_chunks[i], lm_head)
        if config.logit_dtype == 'fp32':
            logits = logits.float()
        logits = config.softcap * torch.tanh(logits / config.softcap)
        chunk = F.cross_entropy(logits, labels_chunks[i], reduction='sum')
        total = chunk if total is None else total + chunk
    return total / (B * T)


def make_optimizer(config, model, num_steps):
    """Fused AdamW + LambdaLR reproducing optax's warmup/constant/warmdown.

    Params are bf16 → Adam moments are bf16 (zeros_like), matching the pure
    bf16 optax recipe on TPU. Decoupled weight decay semantics also match
    (both scale wd by lr)."""
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                            betas=(config.beta1, config.beta2), eps=config.eps,
                            weight_decay=config.weight_decay, fused=True)

    warmup_steps = int(config.warmup_ratio * num_steps)
    warmdown_steps = int(config.warmdown_ratio * num_steps)
    constant_steps = num_steps - warmup_steps - warmdown_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if step < warmup_steps + constant_steps:
            return 1.0
        t = min(step - warmup_steps - constant_steps, warmdown_steps)
        return 1.0 + (config.final_lr_frac - 1.0) * (t / max(warmdown_steps, 1))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched


def maybe_compile(fn, config):
    if config.compile_mode == 'none':
        return fn
    return torch.compile(fn, mode=config.compile_mode, fullgraph=True)


def make_train_step(config, model, opt, sched):
    """Train step: fwd + loss (compiled region) → backward → fused AdamW.
    The SDPA backend pin wraps the compiled call from outside."""

    def fwd_loss(x, y):
        hidden = model(x)
        return chunked_lm_head_loss(hidden, model.lm_head, y, config)

    fwd_loss_c = maybe_compile(fwd_loss, config)

    def train_step(x, y):
        with attn_ctx(config):
            loss = fwd_loss_c(x, y)
            loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        return loss.detach()

    return train_step


@torch.no_grad()
def eval_step(config, model, x, y):
    """Eval: returns loss for a single batch."""
    with attn_ctx(config):
        hidden = model(x)
        return chunked_lm_head_loss(hidden, model.lm_head, y, config)


@torch.no_grad()
def predict_step(config, model, x):
    """Single step inference: returns logits."""
    with attn_ctx(config):
        hidden = model(x)
    logits = torch.einsum('btd,dv->btv', hidden, model.lm_head)
    if config.logit_dtype == 'fp32':
        logits = logits.float()
    return config.softcap * torch.tanh(logits / config.softcap)


def generate(config, model, enc, prompt, max_new_tokens=64,
             temperature=0.8, top_k=50):
    """Generate text from a prompt using top-k + temperature sampling."""
    bos_id = enc.encode_single_token('<|bos|>')
    ids = [bos_id] + enc.encode_ordinary(prompt)
    g = torch.Generator(device='cuda').manual_seed(42)

    for _ in range(max_new_tokens):
        context = ids[-config.seq_len:]
        pad_len = config.seq_len - len(context)
        x = torch.tensor([context + [0] * pad_len], dtype=torch.long, device='cuda')
        logits = predict_step(config, model, x)
        next_logits = logits[0, len(context) - 1, :].float()

        if temperature == 0:
            next_id = int(next_logits.argmax())
        else:
            next_logits = next_logits / temperature
            if top_k > 0:
                top_vals = torch.topk(next_logits, top_k).values
                next_logits = torch.where(next_logits >= top_vals[-1],
                                          next_logits, torch.tensor(-1e10, device='cuda'))
            probs = next_logits.softmax(dim=-1)
            next_id = int(torch.multinomial(probs, 1, generator=g))
        ids.append(next_id)

    return enc.decode(ids)

# %% [markdown]
# ## Section B — Component benchmarks
#
# Times each piece of the training step in isolation: the attention op, one
# transformer layer, the full model, the optimizer, and the assembled train
# step. Labels are string-identical to 11_gpu_jax so `gpu/compare.py` can
# join the two result sets. Includes an SDPA-vs-math numerical parity check.

# %%
# === Section B: correctness check (SDPA vs math attention) ===

_check_cfg_sdpa = Config(attn_impl='sdpa_cudnn')
_check_cfg_math = Config(attn_impl='math')
_B_check = 4    # small batch: math path materializes (B,N,T,T) scores

_gen = torch.Generator(device='cuda').manual_seed(0)
_q = torch.randn(_B_check, config.seq_len, config.n_head, config.head_dim,
                 dtype=torch.bfloat16, device='cuda', generator=_gen)
_k = torch.randn(_B_check, config.seq_len, config.n_kv_head, config.head_dim,
                 dtype=torch.bfloat16, device='cuda', generator=_gen)
_v = torch.randn(_B_check, config.seq_len, config.n_kv_head, config.head_dim,
                 dtype=torch.bfloat16, device='cuda', generator=_gen)

with torch.no_grad():
    with attn_ctx(_check_cfg_sdpa):
        _out_sdpa = attention(_check_cfg_sdpa, _q, _k, _v)
    _out_math = attention(_check_cfg_math, _q, _k, _v)
_max_diff = float((_out_sdpa.float() - _out_math.float()).abs().max())
print(f'SDPA vs math attention max abs diff: {_max_diff:.4f} (bf16 tolerance ~0.06)')
assert _max_diff < 0.0625, 'SDPA attention diverges from math reference'

# %%
# === Section B: component benchmarks ===

B, T = config.batch_size, config.seq_len
D, N, K, H = config.n_embd, config.n_head, config.n_kv_head, config.head_dim
F_mlp = config.mlp_dim

fwd_flops = (config.n_layer * layer_flops(B, T, D, N, K, H, F_mlp, config.mlp_type)
             + matmul_flops(B * T, config.vocab_size, D))
step_flops = 3 * fwd_flops  # fwd + 2x bwd

bench_model = GPT(config).to('cuda')
bench_layer = bench_model.layers[0]
gen = torch.Generator(device='cuda').manual_seed(0)
q_full = torch.randn(B, T, N, H, dtype=torch.bfloat16, device='cuda', generator=gen)
k_full = torch.randn(B, T, K, H, dtype=torch.bfloat16, device='cuda', generator=gen)
v_full = torch.randn(B, T, K, H, dtype=torch.bfloat16, device='cuda', generator=gen)
x_hidden = torch.randn(B, T, D, dtype=torch.bfloat16, device='cuda', generator=gen)
tokens = torch.randint(0, config.vocab_size, (B, T), device='cuda',
                       generator=gen, dtype=torch.long)
labels = torch.randint(0, config.vocab_size, (B, T), device='cuda',
                       generator=gen, dtype=torch.long)
cos_b, sin_b = bench_model.cos, bench_model.sin

SECTION_B_RESULTS = []
attn_fl = attention_flops(B, N, T, H)
layer_fl = layer_flops(B, T, D, N, K, H, F_mlp, config.mlp_type)

# -- attention op --
_attn_core = maybe_compile(lambda q, k, v: attention(config, q, k, v), config)


@torch.no_grad()
def attn_fwd(q, k, v):
    with attn_ctx(config):
        return _attn_core(q, k, v)


SECTION_B_RESULTS.append(benchmark(
    attn_fwd, q_full, k_full, v_full,
    flop_count=attn_fl, label='attention fwd'))


def attn_fwd_bwd(q, k, v):
    q, k, v = (t.detach().requires_grad_(True) for t in (q, k, v))
    with attn_ctx(config):
        out = _attn_core(q, k, v)
        out.float().sum().backward()
    return q.grad


SECTION_B_RESULTS.append(benchmark(
    attn_fwd_bwd, q_full, k_full, v_full,
    flop_count=3 * attn_fl, label='attention fwd+bwd'))

# -- single layer --
_layer_core = maybe_compile(lambda x: bench_layer(x, cos_b, sin_b), config)


@torch.no_grad()
def layer_fwd(x):
    with attn_ctx(config):
        return _layer_core(x)


SECTION_B_RESULTS.append(benchmark(
    layer_fwd, x_hidden,
    flop_count=layer_fl, label='layer fwd'))


def layer_fwd_bwd(x):
    x = x.detach().requires_grad_(True)
    with attn_ctx(config):
        out = _layer_core(x)
        out.float().sum().backward()
    return x.grad


SECTION_B_RESULTS.append(benchmark(
    layer_fwd_bwd, x_hidden,
    flop_count=3 * layer_fl, label='layer fwd+bwd'))
bench_model.zero_grad(set_to_none=True)

# -- full model --
_model_core = maybe_compile(lambda t: bench_model(t), config)


@torch.no_grad()
def model_fwd(t):
    with attn_ctx(config):
        return _model_core(t)


SECTION_B_RESULTS.append(benchmark(
    model_fwd, tokens,
    flop_count=config.n_layer * layer_fl, label='model fwd'))


def _fwd_loss_bench(x, y):
    hidden = bench_model(x)
    return chunked_lm_head_loss(hidden, bench_model.lm_head, y, config)


_fwd_loss_bench_c = maybe_compile(_fwd_loss_bench, config)


def model_fwd_bwd(x, y):
    bench_model.zero_grad(set_to_none=True)
    with attn_ctx(config):
        loss = _fwd_loss_bench_c(x, y)
        loss.backward()
    return loss


SECTION_B_RESULTS.append(benchmark(
    model_fwd_bwd, tokens, labels,
    flop_count=step_flops, label='model fwd+bwd (loss+grads)'))

# -- optimizer step alone (fake grads) --
NUM_QUICK_STEPS = 300
bench_opt, _bench_sched = make_optimizer(config, bench_model, NUM_QUICK_STEPS)
fake_grads = [torch.full_like(p, 1e-4) for p in bench_model.parameters()]


def opt_only():
    for p, g in zip(bench_model.parameters(), fake_grads):
        p.grad = g
    bench_opt.step()


SECTION_B_RESULTS.append(benchmark(opt_only, label='optimizer step'))
bench_model.zero_grad(set_to_none=True)

# -- full train step --
bench_train_step = make_train_step(config, bench_model, bench_opt, _bench_sched)
SECTION_B_RESULTS.append(benchmark(
    bench_train_step, tokens, labels,
    flop_count=step_flops, label='train step'))

print_summary(SECTION_B_RESULTS)

# Free benchmark buffers before training
del bench_model, bench_opt, fake_grads, q_full, k_full, v_full, x_hidden
del _attn_core, _layer_core, _model_core, _fwd_loss_bench_c, bench_train_step
torch.cuda.empty_cache()

# %% [markdown]
# ## Section C — Quick Training (torch.profiler)
#
# Trains for 300 steps — outputs MFU and throughput, captures a torch.profiler
# trace on steps 15-20 (chrome trace format, summarized in Section D). Check
# that the loss goes down and that the profiler shows dense kernel execution
# (compute-bound) rather than gaps (input-bound). The first step's time is
# reported separately — it includes torch.compile warmup.

# %%
# === Section C: quick training ===

EVAL_EVERY = 100
XPROF_START, XPROF_END = 15, 20
LOG_DIR = '/content/log_dir'
os.makedirs(LOG_DIR, exist_ok=True)

# Init model + optimizer
model = GPT(config).to('cuda')
total_p = count_params(model)
non_embed_p = count_non_embed_params(model)
print(f'Params: {total_p/1e6:.1f}M total, {non_embed_p/1e6:.1f}M non-embed')
print(f'Batch: {config.batch_size} x {config.seq_len} = '
      f'{config.batch_size * config.seq_len:,} tokens/step')

opt, sched = make_optimizer(config, model, NUM_QUICK_STEPS)
train_step = make_train_step(config, model, opt, sched)

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
prof = None

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
            vx, vy = _to_device(vx), _to_device(vy)
            vl = eval_step(config, model, vx, vy)
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

    # --- Profiler ---
    if step == XPROF_START:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA])
        prof.start()
        print("Profiler started...")
    if step == XPROF_END:
        prof.stop()
        prof.export_chrome_trace(os.path.join(LOG_DIR, 'trace_torch.json'))
        print(f"Profiler stopped. Trace saved to '{LOG_DIR}'.")

    # --- Train step ---
    x_batch, y_batch = next(train_loader)

    t0 = time.time()
    loss = train_step(x_batch, y_batch)
    loss_val = loss.item()   # single sync point per step, like block_until_ready
    dt = time.time() - t0

    if step == 0:
        compile_s = dt  # first call includes torch.compile warmup

    if step > XPROF_END:
        if mfu_t0 is None:
            mfu_t0 = time.time()
        mfu_tokens += config.batch_size * config.seq_len
        step_times.append(dt)
    report_tokens += config.batch_size * config.seq_len

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
    text = generate(config, model, enc, prompt, max_new_tokens=64)
    print(f'Prompt: {prompt}\nOutput: {text}\n')

# %% [markdown]
# ## Section D — Profiler summary
#
# Prints the top-25 GPU kernels by total device time from the torch.profiler
# capture, and tars the chrome trace for download via `colab download`.

# %%
# === Section D: profiler summary + trace tarball ===

TOP_KERNELS = []
if prof is not None:
    try:
        table = prof.key_averages().table(sort_by='self_device_time_total',
                                          row_limit=25)
    except Exception:
        table = prof.key_averages().table(sort_by='self_cuda_time_total',
                                          row_limit=25)
    print(table)

    events = prof.key_averages()
    total_dev_us = sum(
        getattr(e, 'self_device_time_total', getattr(e, 'self_cuda_time_total', 0))
        for e in events)
    ranked = sorted(
        events,
        key=lambda e: getattr(e, 'self_device_time_total',
                              getattr(e, 'self_cuda_time_total', 0)),
        reverse=True)
    for e in ranked[:25]:
        dev_us = getattr(e, 'self_device_time_total',
                         getattr(e, 'self_cuda_time_total', 0))
        if dev_us <= 0:
            continue
        TOP_KERNELS.append({
            'name': e.key,
            'total_ms': round(dev_us / 1000, 3),
            'pct_of_captured': round(100 * dev_us / total_dev_us, 2) if total_dev_us else 0,
        })

TRACE_TARBALL = '/content/trace_torch.tar.gz'
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
    'framework': 'torch',
    'notebook': '12_gpu_torch',
    'revision': REVISION,
    'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'env': {
        'gpu_name': GPU_INFO,
        'vram_gb': 96,
        'framework_version': torch.__version__,
        'python': sys.version.split()[0],
        'attn_impl': config.attn_impl,
        'compile_mode': config.compile_mode,
        'xla_flags': None,
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
