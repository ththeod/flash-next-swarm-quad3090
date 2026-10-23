#!/usr/bin/env python3
"""Build the DFlash 2 drafter directory this fork serves with.

    VENV/bin/python scripts/graft-drafter-embed.py DRAFTER_DIR CKPT_DIR OUT_DIR

DRAFTER_DIR: incoai/Qwen3.8-Flash-Next-DFlash2 (tested at revision a22a1744).
CKPT_DIR:    halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin.
OUT_DIR:     config.json and model.safetensors hard-linked (or copied) from DRAFTER_DIR, plus
             embed_tokens.safetensors = the checkpoint's input embedding (model.language_model.embed_tokens.weight,
             BF16, 248320 x 2560, ~1.3 GB).

Why: with pipeline parallelism the drafter runs on the last stage, which holds no target embedding to share, so the
drafter loads its own copy. The LM head is still shared from the target at load time. No weights are modified.
"""
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

KEY = "model.language_model.embed_tokens.weight"


def link_or_copy(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    drafter, ckpt, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    for name in ("config.json", "model.safetensors"):
        link_or_copy(os.path.join(drafter, name), os.path.join(out, name))
    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
    shard = index["weight_map"][KEY]
    with safe_open(os.path.join(ckpt, shard), framework="pt") as f:
        w = f.get_tensor(KEY)
    if tuple(w.shape) != (248320, 2560):
        sys.exit(f"unexpected embedding shape {tuple(w.shape)}")
    save_file({"embed_tokens.weight": w.contiguous()}, os.path.join(out, "embed_tokens.safetensors"),
              metadata={"format": "pt", "source": f"{os.path.basename(os.path.normpath(ckpt))} {KEY}"})
    print(f"wrote {out}: config.json, model.safetensors (from {drafter}), embed_tokens.safetensors {tuple(w.shape)} {w.dtype}")


if __name__ == "__main__":
    main()
