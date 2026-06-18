# Bundled ALFworld TW-mode data

Pre-downloaded `~/.cache/alfworld/` slice that the **AlfredTWEnv** (text mode) needs.
Bundled here so air-gapped servers don't need `alfworld-download` (which requires
network access to GitHub releases).

## Contents (after extract)

```
$ALFWORLD_DATA/
├── json_2.1.1/
│   ├── train/        ← 8810 episodes (~1.9 GB)
│   ├── valid_seen/   ← seen unseen (~76 MB)
│   └── valid_unseen/ ← held-out (~78 MB)
└── logic/
    ├── alfred.pddl
    └── alfred.twl2
```

Total **~2.1 GB extracted**, **129 MB compressed across 2 chunks**.

## Extract

```bash
bash scripts/extract_alfworld_data.sh
# → defaults to $HOME/.cache/alfworld (which alfworld auto-picks up)

# or to a custom location:
ALFWORLD_DATA=/scratch/<user>/alfworld bash scripts/extract_alfworld_data.sh
export ALFWORLD_DATA=/scratch/<user>/alfworld   # then point alfworld at it
```

## What's NOT included (and why)

| Excluded | Size | Reason |
|---|---|---|
| `detectors/mrcnn.pth` | 170 MB | Only for **Thor visual** mode; this repo uses `AlfredTWEnv` (text-only) |
| `json_2.1.1/valid_train/` | 63 MB | Not referenced by verl-agent's TW config |

If you switch to Thor visual mode later, run `alfworld-download` on an
internet-connected box and bring the detectors over manually.

## Verifying integrity

```bash
cd data/alfworld
sha256sum -c SHA256SUMS              # per-chunk integrity
# also FULL_SHA256 contains the sha256 of the reassembled tar.gz
```

## Files in this dir

```
alfworld_tw.tar.gz.part-aa    95 MB
alfworld_tw.tar.gz.part-ab    34 MB
SHA256SUMS                    per-chunk SHA256
FULL_SHA256                   reassembled tar.gz SHA256 (sanity)
README.md                     this file
```

## Re-bundling (if alfworld releases a new version)

```bash
# on a machine WITH internet:
pip install alfworld
alfworld-download -f                  # writes ~/.cache/alfworld/

# strip detectors + valid_train, compress, split:
cd ~/.cache/alfworld
tar c json_2.1.1/train json_2.1.1/valid_seen json_2.1.1/valid_unseen logic | \
    pigz -p 8 > /tmp/alfworld_tw.tar.gz
cd /tmp
split -b 95M alfworld_tw.tar.gz alfworld_tw.tar.gz.part-

# move into repo + SHA256
mv alfworld_tw.tar.gz.part-* <repo>/data/alfworld/
cd <repo>/data/alfworld
sha256sum alfworld_tw.tar.gz.part-* > SHA256SUMS
cat alfworld_tw.tar.gz.part-* | sha256sum | awk '{print $1}' > FULL_SHA256
git add . && git commit -m "data: refresh alfworld TW bundle"
```
