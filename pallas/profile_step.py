from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Profile each component of a single layer forward pass
# to identify where wall time goes

M = cfg.microbatch_size * cfg.seq_len  # 4 * 2048 = 8192
D = cfg.n_embd                         # 1024
F = cfg.mlp_dim                        # 3072
N_heads = cfg.n_head                    # 4
K_heads = cfg.n_kv_head                # 1
H = cfg.head_dim                       # 256
T = cfg.seq_len                        # 2048
B_micro = cfg.microbatch_size          # 4

layer = params.layers[0]
cos, sin = precompute_rope(T, H)
cos = cos[None, None, :, :]
sin = sin[None, None, :, :]

x = jax.random.normal(jax.random.key(0), (B_micro, T, D), dtype=jnp.bfloat16)

print(f"Profiling single layer components: B={B_micro}, T={T}, D={D}, N={N_heads}, K={K_heads}, H={H}, F={F}")
print(f"{'='*100}")
print()

# 1. RMSNorm
@jax.jit
def do_rms_norm(x):
    return rms_norm(x)

flops_norm = 0  # elementwise, not MXU
hbm_norm = B_micro * T * D * 2 * 2  # read + write bf16
print("--- RMSNorm ---")
benchmark(do_rms_norm, x, hbm_bytes=hbm_norm, label="rms_norm")

# 2. Q projection: (B,T,D) @ (D,N,H) -> (B,N,T,H) via einsum
h = rms_norm(x)

@jax.jit
def do_q_proj(h, w):
    return jnp.einsum('btd,dnh->bnth', h, w)

flops_q = 2 * B_micro * T * D * N_heads * H
hbm_q = (B_micro * T * D + D * N_heads * H + B_micro * N_heads * T * H) * 2
print("\n--- Q projection ---")
benchmark(do_q_proj, h, layer.c_q, flop_count=flops_q, hbm_bytes=hbm_q, label="Q proj (btd,dnh->bnth)")

# 3. K projection
@jax.jit
def do_k_proj(h, w):
    return jnp.einsum('btd,dnh->bnth', h, w)

flops_k = 2 * B_micro * T * D * K_heads * H
hbm_k = (B_micro * T * D + D * K_heads * H + B_micro * K_heads * T * H) * 2
print("\n--- K projection ---")
benchmark(do_k_proj, h, layer.c_k, flop_count=flops_k, hbm_bytes=hbm_k, label="K proj (btd,dkh->bkth)")

# 4. V projection
@jax.jit
def do_v_proj(h, w):
    return jnp.einsum('btd,dnh->bnth', h, w)

flops_v = 2 * B_micro * T * D * K_heads * H
print("\n--- V projection ---")
benchmark(do_v_proj, h, layer.c_v, flop_count=flops_v, hbm_bytes=hbm_k, label="V proj (btd,dkh->bkth)")

# 5. Output projection: (B,N,T,H) @ (N,H,D) -> (B,T,D)
attn_out = jax.random.normal(jax.random.key(1), (B_micro, N_heads, T, H), dtype=jnp.bfloat16)

@jax.jit
def do_out_proj(attn_out, w):
    return jnp.einsum('bnth,nhd->btd', attn_out, w)

flops_out = 2 * B_micro * T * N_heads * H * D
hbm_out = (B_micro * N_heads * T * H + N_heads * H * D + B_micro * T * D) * 2
print("\n--- Output projection ---")
benchmark(do_out_proj, attn_out, layer.c_proj, flop_count=flops_out, hbm_bytes=hbm_out, label="Out proj (bnth,nhd->btd)")

# 6. MLP gate: (B,T,D) @ (D,F) -> (B,T,F)
@jax.jit
def do_mlp_gate(h, w_gate):
    return jax.nn.silu(jnp.einsum('btd,df->btf', h, w_gate))

flops_gate = 2 * B_micro * T * D * F
hbm_gate = (B_micro * T * D + D * F + B_micro * T * F) * 2
print("\n--- MLP gate (silu) ---")
benchmark(do_mlp_gate, h, layer.w_gate, flop_count=flops_gate, hbm_bytes=hbm_gate, label="MLP gate+silu (btd,df->btf)")

# 7. MLP up: (B,T,D) @ (D,F) -> (B,T,F)
@jax.jit
def do_mlp_up(h, w_up):
    return jnp.einsum('btd,df->btf', h, w_up)

flops_up = 2 * B_micro * T * D * F
print("\n--- MLP up ---")
benchmark(do_mlp_up, h, layer.w_up, flop_count=flops_up, hbm_bytes=hbm_gate, label="MLP up (btd,df->btf)")

# 8. MLP down: (B,T,F) @ (F,D) -> (B,T,D)
gate_up = jax.random.normal(jax.random.key(2), (B_micro, T, F), dtype=jnp.bfloat16)

@jax.jit
def do_mlp_down(x, w_down):
    return jnp.einsum('btf,fd->btd', x, w_down)

flops_down = 2 * B_micro * T * F * D
hbm_down = (B_micro * T * F + F * D + B_micro * T * D) * 2
print("\n--- MLP down ---")
benchmark(do_mlp_down, gate_up, layer.w_down, flop_count=flops_down, hbm_bytes=hbm_down, label="MLP down (btf,fd->btd)")

# 9. Full single layer forward
@jax.jit
def do_single_layer(x, layer, cos, sin):
    h = rms_norm(x)
    q = jnp.einsum('btd,dnh->bnth', h, layer.c_q)
    k = jnp.einsum('btd,dnh->bnth', h, layer.c_k)
    v = jnp.einsum('btd,dnh->bnth', h, layer.c_v)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    q = rms_norm(q)
    k = rms_norm(k)
    # Use einsum attention (simpler than splash for profiling)
    scale = H ** -0.5
    k_exp = jnp.repeat(k, N_heads, axis=1)
    v_exp = jnp.repeat(v, N_heads, axis=1)
    scores = jnp.einsum('bnth,bnsh->bnts', q, k_exp) * scale
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    scores = jnp.where(mask[None, None], scores, jnp.finfo(scores.dtype).min)
    attn_w = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.einsum('bnts,bnsh->bnth', attn_w, v_exp)
    attn_out = jnp.einsum('bnth,nhd->btd', attn_out, layer.c_proj)
    x = x + attn_out
    h2 = rms_norm(x)
    gate = jax.nn.silu(jnp.einsum('btd,df->btf', h2, layer.w_gate))
    up = jnp.einsum('btd,df->btf', h2, layer.w_up)
    mlp_out = jnp.einsum('btf,fd->btd', gate * up, layer.w_down)
    x = x + mlp_out
    return x

flops_layer = layer_flops(B_micro, T, D, N_heads, K_heads, H, F)
hbm_layer = (B_micro * T * D) * 2 * 4  # rough estimate
print("\n--- Full single layer (einsum attn) ---")
benchmark(do_single_layer, x, layer, cos, sin, flop_count=flops_layer, label="single layer fwd")

# 10. Full single layer forward + backward
@jax.jit
def do_single_layer_grad(x, layer, cos, sin):
    def fwd(x):
        return jnp.sum(do_single_layer(x, layer, cos, sin))
    return jax.grad(fwd)(x)

print("\n--- Full single layer fwd+bwd ---")
benchmark(do_single_layer_grad, x, layer, cos, sin, flop_count=flops_layer * 3, label="single layer fwd+bwd")

print_summary(ALL_RESULTS)
