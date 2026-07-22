#!/usr/bin/env bash
# Colab CLI driver for the GPU benchmark notebooks (11_gpu_jax, 12_gpu_torch).
#
# Prereqs: `uv tool install google-colab-cli`, then authenticate ONCE by
# running any command interactively (e.g. `colab sessions`) and completing the
# browser OAuth flow. Note: `colab auth` is something else (VM-side GCP creds).
#
# One long-lived G4 session is reused across runs so data and compile caches
# stay warm; `stop` it when idle to save compute units. Kernel state persists
# across exec calls — `snippet` runs ad-hoc code against the warm kernel.
#
# Usage:
#   bash gpu/run_bench.sh setup            # provision G4 session (+ upload secrets if present)
#   bash gpu/run_bench.sh jax              # run 11_gpu_jax.py on the session
#   bash gpu/run_bench.sh torch            # run 12_gpu_torch.py on the session
#   bash gpu/run_bench.sh snippet FILE     # run an ad-hoc .py against the warm kernel
#   bash gpu/run_bench.sh fetch jax|torch  # download metrics.json + trace tarball
#   bash gpu/run_bench.sh status|log|stop
#
# Env overrides: COLAB_SESSION (default gpu-bench), COLAB_GPU (default G4).
set -euo pipefail
cd "$(dirname "$0")/.."

SESSION="${COLAB_SESSION:-gpu-bench}"
GPU="${COLAB_GPU:-G4}"
CMD="${1:-help}"

case "$CMD" in
  setup)
    colab new --gpu "$GPU" -s "$SESSION"
    if [ -f gpu/.secrets.json ]; then
      colab upload -s "$SESSION" gpu/.secrets.json /content/secrets.json
      echo "Uploaded gpu/.secrets.json -> /content/secrets.json"
    else
      echo "No gpu/.secrets.json — notebooks will run anonymously (HF repo is public)"
    fi
    colab status -s "$SESSION"
    ;;
  jax)
    colab exec --timeout 3600 -s "$SESSION" -f 11_gpu_jax.py
    ;;
  torch)
    colab exec --timeout 3600 -s "$SESSION" -f 12_gpu_torch.py
    ;;
  snippet)
    colab exec --timeout 3600 -s "$SESSION" -f "${2:?usage: run_bench.sh snippet FILE.py}"
    ;;
  fetch)
    FW="${2:?usage: run_bench.sh fetch jax|torch}"
    mkdir -p gpu/results
    colab download -s "$SESSION" /content/metrics.json "gpu/results/metrics_${FW}.json"
    colab download -s "$SESSION" "/content/trace_${FW}.tar.gz" \
      "gpu/results/trace_${FW}.tar.gz" || echo "(no trace tarball fetched)"
    echo "Fetched gpu/results/metrics_${FW}.json"
    ;;
  status)
    colab status -s "$SESSION"
    ;;
  log)
    colab log -s "$SESSION" ${2:+-n "$2"}
    ;;
  stop)
    colab stop -s "$SESSION"
    ;;
  *)
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -20
    ;;
esac
