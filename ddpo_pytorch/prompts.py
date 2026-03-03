import os
import random
import functools
from typing import Tuple, Dict

@functools.cache
def _read_lines(txt_path: str):
    """Read non-empty lines from a UTF-8 txt file and cache them."""
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Prompt file not found: {txt_path}")
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    lines = [x for x in lines if x]
    if not lines:
        raise ValueError(f"Prompt file is empty: {txt_path}")
    return lines

def flux_reasoning_20k(path: str = "data/flux_reasoning_prompts_20k.txt") -> Tuple[str, Dict]:
    """Return one random prompt from the FLUX-Reason-6M extracted txt."""
    lines = _read_lines(path)
    prompt = random.choice(lines)
    return prompt, {"source": "flux_txt", "path": path}