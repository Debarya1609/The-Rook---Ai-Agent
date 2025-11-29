# rook_orchestrator/utils/key_loader.py
import os
from typing import List

def load_keys_from_env(env_var: str = "MULTI_GEMINI_KEYS") -> List[str]:
    raw = os.getenv(env_var, "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise RuntimeError(f"No API keys found in env var {env_var}. Set MULTI_GEMINI_KEYS in your .env.")
    return keys

def mask_key(key: str) -> str:
    """Return a masked representation for logs, e.g. ...a1b2c3"""
    if not key:
        return "EMPTY"
    if len(key) <= 8:
        return f"...{key}"
    return f"...{key[-6:]}"
