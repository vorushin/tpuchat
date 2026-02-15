#!/bin/bash
# Regenerate all .ipynb files from their .py sources using jupytext
set -e
cd "$(dirname "$0")"

for py_file in [0-9]*.py; do
    echo "Converting $py_file -> ${py_file%.py}.ipynb"
    jupytext --to ipynb "$py_file"
done

echo "Done."
