#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a local prompt txt file from FLUX-Reason-6M using streaming.

Notes:
- We avoid decoding images to keep it fast and stable.
- Default caption field: caption_composition / caption_composition_cn.
"""

import argparse
import os
import random
import re
from datasets import load_dataset

REASONING_REGEX = re.compile(
    r"\b("
    r"left|right|above|below|under|over|behind|in front of|inside|outside|between|"
    r"two|three|four|five|six|seven|eight|nine|ten|"
    r"one|1|2|3|4|5|6|7|8|9|10|"
    r"nearest|farthest|next to|adjacent|"
    r"top|bottom|upper|lower|"
    r"count|number|exactly"
    r")\b",
    flags=re.IGNORECASE,
)

CAPTION_MAP = {
    # main type
    "composition": ("caption_composition", "caption_composition_cn"),
    "entity": ("caption_entity", "caption_entity_cn"),
    "text": ("caption_text", "caption_text_cn"),
    "imaginative": ("caption_imaginative", "caption_imaginative_cn"),
    "style": ("caption_style", "caption_style_cn"),
    "abstract": ("caption_abstract", "caption_abstract_cn"),
    "original": ("caption_original", "caption_original_cn"),
    "detail": ("caption_detail", "caption_detail_cn"),
}


def looks_reasoning(prompt: str) -> bool:
    """Return True if prompt contains reasoning-related keywords."""
    return bool(prompt) and (REASONING_REGEX.search(prompt) is not None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="LucasFang/FLUX-Reason-6M")
    parser.add_argument("--split", default="train")
    parser.add_argument("--lang", choices=["en", "cn"], default="en",
                        help="Use English (en) or Chinese (cn) captions")
    parser.add_argument("--caption_type", choices=list(CAPTION_MAP.keys()), default="composition",
                        help="Which caption field family to use")
    parser.add_argument("--out", default="data/flux_prompts.txt")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reasoning_only", action="store_true")
    parser.add_argument("--print_k", type=int, default=20)
    parser.add_argument("--max_scan", type=int, default=200000)
    parser.add_argument("--progress_every", type=int, default=5000)
    args = parser.parse_args()

    random.seed(args.seed)

    cap_en, cap_cn = CAPTION_MAP[args.caption_type]
    caption_key = cap_en if args.lang == "en" else cap_cn

    # Stream dataset; remove image column to avoid image decoding overhead
    ds = load_dataset(args.repo, split=args.split, streaming=True)
    ds = ds.remove_columns(["image"])  # critical: speed + avoid hanging on image decoding

    collected = []
    scanned = 0

    for ex in ds:
        scanned += 1
        if args.progress_every > 0 and scanned % args.progress_every == 0:
            print(f"[INFO] scanned={scanned}, collected={len(collected)}", flush=True)

        if scanned > args.max_scan:
            break

        prompt = ex.get(caption_key, None)
        if not isinstance(prompt, str):
            continue

        prompt = prompt.strip()
        if len(prompt) < 5:
            continue

        if args.reasoning_only and not looks_reasoning(prompt):
            continue

        collected.append(prompt)

        if len(collected) >= args.n:
            break

    if not collected:
        raise RuntimeError(
            f"No prompts collected. scanned={scanned}. "
            f"Try increasing --max_scan or relaxing filters."
        )

    # Deduplicate while preserving order
    collected = list(dict.fromkeys(collected))

    random.shuffle(collected)
    collected = collected[: args.n]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in collected:
            f.write(p.replace("\n", " ").strip() + "\n")

    print(f"[OK] caption_key={caption_key}")
    print(f"[OK] scanned={scanned}, collected={len(collected)}")
    print(f"[OK] prompts written to: {args.out}")
    print("---- sample prompts ----")
    for i, p in enumerate(collected[: args.print_k]):
        print(f"{i:04d}: {p}")


if __name__ == "__main__":
    main()