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
# # Pallas Kernel Dev — Code Execution Server
#
# Open this notebook in Colab with a TPU runtime. Run all cells to start
# an HTTP server that accepts code snippets from `colab_client.py`.
# The server pre-loads model params and benchmark utilities so you can
# iterate on Pallas kernels without restarting the runtime.

# %%
# !pip install -q --ignore-installed flask pyngrok
# !pip install -qU "jax[tpu]"  # align libtpu with JAX (fixes Mosaic IR version mismatch)

# %%
import functools as ft, io, json, os, secrets, sys, time, traceback, threading
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

PEAK_TFLOPS = 918
HBM_BW_GBS = 1600

print(f"JAX version : {jax.__version__}")
print(f"Devices     : {jax.devices()}")

# %%
# === dot_dict: JAX-compatible mutable dictionary ===

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
# === Benchmark harness ===

ALL_RESULTS = []

def benchmark(fn, *args, warmup=3, repeats=10, flop_count=None,
              hbm_bytes=None, label=""):
    """Run fn repeatedly and report wall time, TFLOP/s, MFU%, HBM bandwidth%."""
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    wall_s = sum(times) / len(times)
    wall_ms = wall_s * 1000

    tflops = flop_count / (wall_s * 1e12) if flop_count else 0.0
    mfu_pct = (tflops / PEAK_TFLOPS * 100
               if flop_count and PEAK_TFLOPS else 0.0)

    hbm_bw_gbs = hbm_bytes / (wall_s * 1e9) if hbm_bytes else 0.0
    hbm_bw_pct = hbm_bw_gbs / HBM_BW_GBS * 100 if hbm_bytes else 0.0

    result = dict(label=label, wall_ms=wall_ms, tflops=tflops,
                  mfu_pct=mfu_pct, hbm_bw_gbs=hbm_bw_gbs,
                  hbm_bw_pct=hbm_bw_pct)
    ALL_RESULTS.append(result)

    mfu_str = f"{mfu_pct:5.1f}%" if flop_count else "  n/a"
    tflop_str = f"{tflops:6.1f}" if flop_count else "   n/a"
    bw_str = f"{hbm_bw_pct:5.1f}%" if hbm_bytes else "  n/a"
    print(f"  {label:<40s}  {wall_ms:8.2f} ms  {tflop_str} TFLOP/s  MFU {mfu_str}  "
          f"HBM BW {bw_str}")
    return result


def print_summary(results):
    """Print a formatted comparison table from benchmark results."""
    print(f"\n  {'Label':<40s}  {'Wall ms':>8s}  {'TFLOP/s':>8s}  {'MFU %':>6s}  "
          f"{'HBM BW%':>7s}")
    print("  " + "-" * 88)
    for r in results:
        mxu = f"{r['mfu_pct']:5.1f}%" if r['tflops'] > 0 else "  n/a"
        tf = f"{r['tflops']:7.1f}" if r['tflops'] > 0 else "    n/a"
        bw = f"{r['hbm_bw_pct']:5.1f}%" if r['hbm_bw_pct'] > 0 else "  n/a"
        print(f"  {r['label']:<40s}  {r['wall_ms']:8.2f}  {tf}  {mxu}  "
              f"{bw:>7s}")
    print()

# %%
# === FLOP counting helpers ===

def matmul_flops(M, N, K, batch=1):
    """FLOPs for [M,K] @ [K,N].  2*M*N*K per batch element."""
    return 2 * batch * M * N * K

def attention_flops(B, N, T, H):
    """FLOPs for QK^T + AV (full T*T, not causal-halved)."""
    return 2 * (2 * B * N * T * T * H)

def layer_flops(B, T, D, N, K, H, F):
    """MXU-relevant FLOPs for one transformer layer."""
    tok = B * T
    q  = 2 * tok * D * N * H
    k  = 2 * tok * D * K * H
    v  = 2 * tok * D * K * H
    att = attention_flops(B, N, T, H)
    proj = 2 * tok * N * H * D
    gate = 2 * tok * D * F
    up   = 2 * tok * D * F
    down = 2 * tok * F * D
    return q + k + v + att + proj + gate + up + down

# %%
# === Config & model init ===

@jax.tree_util.register_static
@dataclass(kw_only=True, frozen=True)
class Config:
    attn_impl: str = 'splash'
    mlp_type: str = 'glu'
    qk_norm: bool = True

    n_embd: int = 1024
    n_layer: int = 8
    seq_len: int = 2048
    vocab_size: int = 32768
    n_head: int = 4
    n_kv_head: int = 1
    head_dim: int = 256
    mlp_dim: int = 3072
    softcap: float = 15.0
    logit_dtype: str = 'bf16'
    splash_block_size: int = 1024
    num_lm_head_chunks: int = 8
    batch_size: int = 64
    microbatch_size: int = 4

    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    warmup_ratio: float = 0.02
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0

    eval_steps: int = 10
    param_seed: int = 42

    @property
    def num_microbatches(self):
        return self.batch_size // self.microbatch_size


def init_layer_params(config, seed=42):
    """Initialize params for one transformer layer."""
    key = jax.random.key(seed)
    keys = jax.random.split(key, 7)
    s = (3.0 ** 0.5) * (config.n_embd ** -0.5)
    layer = dot_dict()

    layer.c_q = jax.random.uniform(keys[0], (config.n_embd, config.n_head, config.head_dim),
                                    dtype=jnp.bfloat16, minval=-s, maxval=s)
    layer.c_k = jax.random.uniform(keys[1], (config.n_embd, config.n_kv_head, config.head_dim),
                                    dtype=jnp.bfloat16, minval=-s, maxval=s)
    layer.c_v = jax.random.uniform(keys[2], (config.n_embd, config.n_kv_head, config.head_dim),
                                    dtype=jnp.bfloat16, minval=-s, maxval=s)
    layer.c_proj = jnp.zeros((config.n_head, config.head_dim, config.n_embd), dtype=jnp.bfloat16)

    if config.mlp_type == 'glu':
        layer.w_gate = jax.random.uniform(keys[3], (config.n_embd, config.mlp_dim),
                                           dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.w_up = jax.random.uniform(keys[4], (config.n_embd, config.mlp_dim),
                                         dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.w_down = jnp.zeros((config.mlp_dim, config.n_embd), dtype=jnp.bfloat16)
    else:
        layer.w_up = jax.random.uniform(keys[3], (config.n_embd, config.mlp_dim),
                                         dtype=jnp.bfloat16, minval=-s, maxval=s)
        layer.w_down = jnp.zeros((config.mlp_dim, config.n_embd), dtype=jnp.bfloat16)
    return layer


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


cfg = Config()
print(f'Config: D={cfg.n_embd}, L={cfg.n_layer}, T={cfg.seq_len}, '
      f'V={cfg.vocab_size}, N={cfg.n_head}, K={cfg.n_kv_head}, '
      f'H={cfg.head_dim}, F={cfg.mlp_dim}')

print("Initializing model params...")
params = init_full_model(cfg)
total_params = sum(p.size for p in jax.tree.leaves(params) if isinstance(p, jax.Array))
print(f"Total params: {total_params / 1e6:.1f}M")

x_hidden = jax.random.normal(jax.random.key(0),
                              (cfg.microbatch_size, cfg.seq_len, cfg.n_embd),
                              dtype=jnp.bfloat16)
print(f"x_hidden shape: {x_hidden.shape}")

# %%
# === Code execution server ===

from flask import Flask, request as flask_request, jsonify

EXEC_NAMESPACE = {
    # Standard
    'ft': ft, 'io': io, 'json': json, 'os': os, 'sys': sys,
    'time': time, 'np': np, 'threading': threading,
    # JAX
    'jax': jax, 'jnp': jnp, 'pl': pl, 'pltpu': pltpu,
    # Model
    'cfg': cfg, 'Config': Config, 'params': params, 'x_hidden': x_hidden,
    'dot_dict': dot_dict,
    'init_layer_params': init_layer_params, 'init_all_layers': init_all_layers,
    'init_full_model': init_full_model,
    # Utilities
    'rms_norm': rms_norm, 'precompute_rope': precompute_rope,
    'apply_rope': apply_rope,
    'benchmark': benchmark, 'print_summary': print_summary,
    'ALL_RESULTS': ALL_RESULTS,
    'matmul_flops': matmul_flops, 'attention_flops': attention_flops,
    'layer_flops': layer_flops,
    # Constants
    'PEAK_TFLOPS': PEAK_TFLOPS, 'HBM_BW_GBS': HBM_BW_GBS,
}

AUTH_TOKEN = secrets.token_urlsafe(32)

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    if flask_request.headers.get('X-Auth-Token') != AUTH_TOKEN:
        return jsonify(error='unauthorized'), 401
    return jsonify(
        status='ok',
        devices=str(jax.devices()),
        jax_version=jax.__version__,
        config=dict(D=cfg.n_embd, L=cfg.n_layer, T=cfg.seq_len,
                    N=cfg.n_head, K=cfg.n_kv_head, H=cfg.head_dim,
                    F=cfg.mlp_dim),
    )

@app.route('/exec', methods=['POST'])
def exec_code():
    if flask_request.headers.get('X-Auth-Token') != AUTH_TOKEN:
        return jsonify(error='unauthorized'), 401

    code = flask_request.json.get('code', '')
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    EXEC_NAMESPACE.pop('__result__', None)
    ALL_RESULTS.clear()  # prevent memory accumulation across requests

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_buf, stderr_buf
    try:
        exec(code, EXEC_NAMESPACE)
        error = None
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    result_val = EXEC_NAMESPACE.get('__result__')
    return jsonify(
        stdout=stdout_buf.getvalue(),
        stderr=stderr_buf.getvalue(),
        error=error,
        result=result_val,
    )

from google.colab import userdata
ngrok_token = userdata.get('NGROK_AUTH_TOKEN')

from pyngrok import ngrok
ngrok.set_auth_token(ngrok_token)
tunnel = ngrok.connect(5000)
public_url = tunnel.public_url

connection_string = f"{public_url}|{AUTH_TOKEN}"
print(f"\n{'='*60}")
print(f"Connection string (save to pallas/.colab_connection):")
print(f"{connection_string}")
print(f"{'='*60}\n")

server_thread = threading.Thread(
    target=lambda: app.run(port=5000, debug=False, use_reloader=False),
    daemon=True,
)
server_thread.start()
print("Server running in background. You can continue using this notebook.")
