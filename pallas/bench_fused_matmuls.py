# Benchmark: fused QKV and gate+up matmuls vs separate
# Hypothesis: larger matmuls fill the 256x256 MXU better

M = cfg.microbatch_size * cfg.seq_len  # 8192
D = cfg.n_embd                         # 1024
F = cfg.mlp_dim                        # 3072
N_h = cfg.n_head                       # 4
K_h = cfg.n_kv_head                    # 1
H = cfg.head_dim                       # 256

h = jax.random.normal(jax.random.key(0), (cfg.microbatch_size, cfg.seq_len, D), dtype=jnp.bfloat16)
layer = params.layers[0]

# ============================================================
# Test 1: QKV projections
# ============================================================
print("=" * 80)
print("QKV PROJECTIONS")
print("=" * 80)

# Current: 3 separate einsums
@jax.jit
def qkv_separate(h, c_q, c_k, c_v):
    q = jnp.einsum('btd,dnh->bnth', h, c_q)
    k = jnp.einsum('btd,dnh->bnth', h, c_k)
    v = jnp.einsum('btd,dnh->bnth', h, c_v)
    return q, k, v

flops_qkv = 2 * M * D * (N_h * H + K_h * H + K_h * H)
hbm_qkv = (M * D + D * (N_h + 2 * K_h) * H + M * (N_h + 2 * K_h) * H) * 2

print("\n--- Separate Q, K, V einsums (current) ---")
benchmark(qkv_separate, h, layer.c_q, layer.c_k, layer.c_v,
          flop_count=flops_qkv, hbm_bytes=hbm_qkv, label="QKV separate (3 einsums)")

# Fused: pack into one (D, (N+2K)*H) weight, one big matmul
W_qkv = jnp.concatenate([
    layer.c_q.reshape(D, -1),   # (1024, 1024)
    layer.c_k.reshape(D, -1),   # (1024, 256)
    layer.c_v.reshape(D, -1),   # (1024, 256)
], axis=1)  # (1024, 1536)

@jax.jit
def qkv_fused(h, W_qkv):
    # One big 2D matmul: (B*T, D) @ (D, 1536)
    B, T, D_ = h.shape
    out = h.reshape(B * T, D_) @ W_qkv  # (B*T, 1536)
    out = out.reshape(B, T, -1)
    # Split and reshape to multi-head
    q_flat, k_flat, v_flat = jnp.split(out, [N_h * H, N_h * H + K_h * H], axis=-1)
    q = q_flat.reshape(B, T, N_h, H).transpose(0, 2, 1, 3)  # (B,N,T,H)
    k = k_flat.reshape(B, T, K_h, H).transpose(0, 2, 1, 3)  # (B,K,T,H)
    v = v_flat.reshape(B, T, K_h, H).transpose(0, 2, 1, 3)  # (B,K,T,H)
    return q, k, v

print("\n--- Fused QKV (one matmul, D -> (N+2K)*H = 1536) ---")
benchmark(qkv_fused, h, W_qkv,
          flop_count=flops_qkv, hbm_bytes=hbm_qkv, label="QKV fused (1 matmul)")

# ============================================================
# Test 2: MLP gate + up projections
# ============================================================
print("\n" + "=" * 80)
print("MLP GATE + UP PROJECTIONS")
print("=" * 80)

h2 = jax.random.normal(jax.random.key(1), (cfg.microbatch_size, cfg.seq_len, D), dtype=jnp.bfloat16)

# Current: 2 separate einsums + silu
@jax.jit
def mlp_gate_up_separate(h2, w_gate, w_up):
    gate = jax.nn.silu(jnp.einsum('btd,df->btf', h2, w_gate))
    up = jnp.einsum('btd,df->btf', h2, w_up)
    return gate * up

flops_gate_up = 2 * M * D * F * 2  # two matmuls
hbm_gate_up = (M * D + D * F * 2 + M * F * 2) * 2

print("\n--- Separate gate + up (current, 2 einsums + silu) ---")
benchmark(mlp_gate_up_separate, h2, layer.w_gate, layer.w_up,
          flop_count=flops_gate_up, hbm_bytes=hbm_gate_up,
          label="gate+up separate (2 einsums)")

# Fused: one (D, 2F) matmul then split
W_gate_up = jnp.concatenate([layer.w_gate, layer.w_up], axis=1)  # (1024, 6144)

@jax.jit
def mlp_gate_up_fused(h2, W_gate_up):
    B, T, D_ = h2.shape
    out = h2.reshape(B * T, D_) @ W_gate_up  # (B*T, 2F)
    out = out.reshape(B, T, 2, F)
    gate = jax.nn.silu(out[:, :, 0, :])
    up = out[:, :, 1, :]
    return gate * up

print("\n--- Fused gate+up (one matmul D -> 2F = 6144, then split+silu) ---")
benchmark(mlp_gate_up_fused, h2, W_gate_up,
          flop_count=flops_gate_up, hbm_bytes=hbm_gate_up,
          label="gate+up fused (1 matmul)")

print_summary(ALL_RESULTS[-4:])
