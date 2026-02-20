#!/bin/bash
# Regenerate all .ipynb files from their .py sources using jupytext
set -e
cd "$(dirname "$0")"

# Notebooks that are run locally (not exported to Colab)
EXCLUDE="06_apple_silicon_perf.py 07_train_mlx.py"

for py_file in [0-9]*.py; do
    if echo "$EXCLUDE" | grep -qw "$py_file"; then
        echo "Skipping $py_file (local-only)"
        continue
    fi
    echo "Converting $py_file -> ${py_file%.py}.ipynb"
    jupytext --to ipynb --update "$py_file"
done

echo "Done."
