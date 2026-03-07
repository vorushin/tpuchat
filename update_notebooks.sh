#!/bin/bash
# Regenerate all .ipynb files from their .py sources using jupytext
# If a .py file has a REVISION = N line and its content changed (tracked via
# .notebook_hashes), the revision is auto-incremented before conversion.
set -e
cd "$(dirname "$0")"

EXCLUDE="06_apple_silicon_perf.py 07_train_mlx.py"
HASH_FILE=".notebook_hashes"

# Compute hash of .py content, excluding revision-related lines
content_hash() {
    sed '/^REVISION = /d' "$1" | sed '/^# # .*rev [0-9]/d' | md5 -q
}

for py_file in [0-9]*.py; do
    if echo "$EXCLUDE" | grep -qw "$py_file"; then
        echo "Skipping $py_file (local-only)"
        continue
    fi

    ipynb_file="${py_file%.py}.ipynb"

    # Auto-increment REVISION if content changed
    if grep -q '^REVISION = ' "$py_file"; then
        cur_hash=$(content_hash "$py_file")
        stored_hash=$(grep "^${py_file} " "$HASH_FILE" 2>/dev/null | awk '{print $2}' || true)

        if [ "$cur_hash" != "$stored_hash" ]; then
            old_rev=$(grep -oE '^REVISION = [0-9]+' "$py_file" | grep -oE '[0-9]+')
            new_rev=$((old_rev + 1))
            sed -i '' "s/^REVISION = ${old_rev}$/REVISION = ${new_rev}/" "$py_file"
            sed -i '' -E "s/\\(rev ${old_rev}\\)/(rev ${new_rev})/" "$py_file"
            echo "Converting $py_file -> $ipynb_file (rev $old_rev -> $new_rev)"
        else
            echo "Converting $py_file -> $ipynb_file (unchanged)"
        fi

        # Update stored hash (recompute after possible revision bump)
        new_hash=$(content_hash "$py_file")
        grep -v "^${py_file} " "$HASH_FILE" 2>/dev/null > "${HASH_FILE}.tmp" || true
        echo "${py_file} ${new_hash}" >> "${HASH_FILE}.tmp"
        mv "${HASH_FILE}.tmp" "$HASH_FILE"
    else
        echo "Converting $py_file -> $ipynb_file"
    fi

    jupytext --to ipynb --update "$py_file"
done

# Pallas notebooks
if [ -f pallas/colab_server.py ]; then
    echo "Converting pallas/colab_server.py -> pallas/colab_server.ipynb"
    jupytext --to ipynb --update pallas/colab_server.py
fi

echo "Done."
