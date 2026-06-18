#!/usr/bin/env bash
# Extract the pre-bundled ALFworld TW-mode data (json_2.1.1 + logic) into the
# location alfworld expects. No network needed.
#
# Usage:
#   bash scripts/extract_alfworld_data.sh                # → $HOME/.cache/alfworld/
#   ALFWORLD_DATA=/scratch/alfworld bash ... [DST]       # custom destination
#
# After extraction:
#   - $DST/json_2.1.1/{train,valid_seen,valid_unseen}
#   - $DST/logic/{alfred.pddl,alfred.twl2}
# Then `export ALFWORLD_DATA=$DST` so alfworld picks it up.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$REPO_ROOT/data/alfworld"
DST="${ALFWORLD_DATA:-${1:-$HOME/.cache/alfworld}}"

echo "[extract] src: $SRC_DIR"
echo "[extract] dst: $DST"

[ -d "$SRC_DIR" ] || { echo "ERROR: $SRC_DIR not found — did you 'git pull'?"; exit 1; }
parts=("$SRC_DIR"/alfworld_tw.tar.gz.part-*)
if [ ! -e "${parts[0]}" ]; then
    echo "ERROR: no alfworld_tw.tar.gz.part-* files under $SRC_DIR"
    exit 1
fi
echo "[extract] found ${#parts[@]} chunks"

# Verify per-chunk SHA256 if SHA256SUMS available
if [ -f "$SRC_DIR/SHA256SUMS" ]; then
    echo "[extract] verifying chunk SHA256 ..."
    (cd "$SRC_DIR" && sha256sum -c SHA256SUMS) || {
        echo "ERROR: SHA256 check failed — re-pull repo"
        exit 1
    }
fi

mkdir -p "$DST"
echo "[extract] reassembling + extracting (this takes ~30s for 129 MB)..."
cat "${parts[@]}" | tar xzf - -C "$DST"

# Sanity check
for sub in json_2.1.1/train json_2.1.1/valid_seen json_2.1.1/valid_unseen logic/alfred.pddl logic/alfred.twl2; do
    [ -e "$DST/$sub" ] || { echo "ERROR: missing $DST/$sub after extract"; exit 1; }
done

echo ""
echo "[extract] ✓ done. Set: export ALFWORLD_DATA=$DST"
echo "[extract] (skip if you used the default $HOME/.cache/alfworld)"
echo ""
echo "Sanity:"
echo "  train episodes:        $(ls -1 $DST/json_2.1.1/train         | wc -l)"
echo "  valid_seen episodes:   $(ls -1 $DST/json_2.1.1/valid_seen    | wc -l)"
echo "  valid_unseen episodes: $(ls -1 $DST/json_2.1.1/valid_unseen  | wc -l)"
