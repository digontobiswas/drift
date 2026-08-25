"""Merge per-shard annotation JSONs written by a `tools/prepare_nuscenes.py` Slurm array.

Each array task writes `drift_<split>.shardNNN.json`; this concatenates them into
the single `drift_<split>.json` that `--ann-file` expects, de-duplicating by
`sample_token` and sorting for reproducibility.

    python tools/merge_shards.py --out-root $DRIFT_DATA_ROOT --split train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", type=str, required=True)
    p.add_argument("--split", type=str, default="train", choices=["train", "val"])
    p.add_argument("--keep-shards", action="store_true", help="Do not delete the per-shard files after merging.")
    args = p.parse_args(argv)

    root = Path(args.out_root)
    shards = sorted(root.glob(f"drift_{args.split}.shard*.json"))
    if not shards:
        raise SystemExit(f"No shard files matching drift_{args.split}.shard*.json in {root}")

    seen = set()
    merged = []
    for s in shards:
        with open(s) as f:
            for entry in json.load(f):
                tok = entry.get("sample_token")
                if tok in seen:
                    continue
                seen.add(tok)
                merged.append(entry)

    merged.sort(key=lambda e: e.get("sample_token", ""))
    out = root / f"drift_{args.split}.json"
    with open(out, "w") as f:
        json.dump(merged, f)
    print(f"[merge] {len(shards)} shards -> {len(merged)} unique samples -> {out}")

    if not args.keep_shards:
        for s in shards:
            s.unlink()
        print(f"[merge] removed {len(shards)} shard files")


if __name__ == "__main__":
    main()
