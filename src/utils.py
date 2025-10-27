import os
from typing import Tuple, Optional

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def input_size_from_arch(arch: str) -> Tuple[int, int]:
    arch = (arch or "").lower()
    if "inception" in arch:
        return (299, 299)
    # Many TF backbones default to 224
    return (224, 224)

def parse_bool(x: Optional[str]) -> bool:
    if x is None: return False
    return str(x).lower() in {"1","true","yes","y"}