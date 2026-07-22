"""Compare two GPU benchmark metrics.json files (JAX vs PyTorch).

Usage:
    uv run python gpu/compare.py gpu/results/metrics_jax.json gpu/results/metrics_torch.json

Prints markdown tables: headline training numbers, component benchmarks joined
on label, calibration, val loss trajectories, and top kernels. Paste into
gpu/REPORT.md.
"""
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(value, spec=''):
    if value is None:
        return 'n/a'
    return format(value, spec) if spec else str(value)


def ratio(a, b):
    """b relative to a, as a speed factor (wall times: lower is better)."""
    if not a or not b:
        return 'n/a'
    return f'{a / b:.2f}x'


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    m1, m2 = load(sys.argv[1]), load(sys.argv[2])
    n1, n2 = m1['framework'], m2['framework']

    print(f'# {n1} vs {n2} — G4 GPU benchmark comparison\n')
    print(f"- {n1}: `{m1['notebook']}` rev {m1['revision']}, "
          f"{m1['env']['framework_version']}, attn={m1['env']['attn_impl']}, "
          f"compile={m1['env'].get('compile_mode')}, {m1['timestamp']}")
    print(f"- {n2}: `{m2['notebook']}` rev {m2['revision']}, "
          f"{m2['env']['framework_version']}, attn={m2['env']['attn_impl']}, "
          f"compile={m2['env'].get('compile_mode')}, {m2['timestamp']}")
    print(f"- GPU: {m1['env']['gpu_name']}\n")

    # --- Headline training numbers ---
    t1, t2 = m1.get('training', {}), m2.get('training', {})
    print('## Training (300-step quick run)\n')
    print(f'| Metric | {n1} | {n2} | {n2}/{n1} |')
    print('|--------|------|------|-------|')
    rows = [
        ('tok/s', 'tok_per_sec', ',', 'higher'),
        ('median step ms', 'step_ms_median', '.1f', 'lower'),
        ('MFU % (vs quoted peak)', 'mfu_quoted_pct', '.1f', 'higher'),
        ('MFU % (vs measured peak)', 'mfu_measured_pct', '.1f', 'higher'),
        ('compile/warmup s', 'compile_s', '.1f', 'lower'),
        ('final train loss (EMA)', 'final_train_loss_ema', '.4f', ''),
    ]
    for name, key, spec, _ in rows:
        v1, v2 = t1.get(key), t2.get(key)
        if v1 and v2:
            r = f'{v2 / v1:.2f}x'
        else:
            r = 'n/a'
        print(f'| {name} | {fmt(v1, spec)} | {fmt(v2, spec)} | {r} |')

    # --- Calibration ---
    c1, c2 = m1.get('calibration', {}), m2.get('calibration', {})
    print('\n## Matmul peak calibration\n')
    print(f'| | {n1} | {n2} |')
    print('|---|------|------|')
    print(f"| quoted peak TFLOP/s | {c1.get('quoted_peak_tflops')} "
          f"| {c2.get('quoted_peak_tflops')} |")
    print(f"| measured peak TFLOP/s | {fmt(c1.get('measured_peak_tflops'), '.0f')} "
          f"| {fmt(c2.get('measured_peak_tflops'), '.0f')} |")

    # --- Components joined on label ---
    comp1 = {c['label']: c for c in m1.get('components', [])}
    comp2 = {c['label']: c for c in m2.get('components', [])}
    labels = [l for l in comp1 if l in comp2]
    only1 = [l for l in comp1 if l not in comp2]
    only2 = [l for l in comp2 if l not in comp1]
    print('\n## Component benchmarks (wall ms — lower is better)\n')
    print(f'| Component | {n1} ms | {n2} ms | {n1}-vs-{n2} | {n1} MFU%m | {n2} MFU%m |')
    print('|-----------|-------|-------|------|------|------|')
    for label in labels:
        a, b = comp1[label], comp2[label]
        faster = f"{n1} {ratio(b['wall_ms'], a['wall_ms'])}" \
            if a['wall_ms'] <= b['wall_ms'] else f"{n2} {ratio(a['wall_ms'], b['wall_ms'])}"
        print(f"| {label} | {a['wall_ms']:.2f} | {b['wall_ms']:.2f} | {faster} "
              f"| {a.get('mfu_measured_pct', 0):.1f} | {b.get('mfu_measured_pct', 0):.1f} |")
    if only1 or only2:
        print(f'\nUnmatched labels — {n1}: {only1}, {n2}: {only2}')

    # --- Val losses ---
    v1 = {v['step']: v['loss'] for v in t1.get('val_losses', [])}
    v2 = {v['step']: v['loss'] for v in t2.get('val_losses', [])}
    steps = sorted(set(v1) & set(v2))
    if steps:
        print('\n## Val loss trajectory\n')
        print(f'| Step | {n1} | {n2} | diff |')
        print('|------|------|------|------|')
        for s in steps:
            print(f'| {s} | {v1[s]:.4f} | {v2[s]:.4f} | {v2[s] - v1[s]:+.4f} |')

    # --- Top kernels ---
    for m, n in [(m1, n1), (m2, n2)]:
        kernels = m.get('profile', {}).get('top_kernels', [])[:10]
        if kernels:
            print(f'\n## Top kernels — {n}\n')
            print('| Kernel | Total ms | % of captured |')
            print('|--------|----------|---------------|')
            for k in kernels:
                name = k['name'][:70].replace('|', '\\|')
                print(f"| `{name}` | {k['total_ms']:.2f} | {k['pct_of_captured']:.1f}% |")


if __name__ == '__main__':
    main()
