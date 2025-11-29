# rook_orchestrator/utils/llm_client.py
"""
Gemini (Google GenAI) client wrapper for The Rook with multi-key rotation.
- Exposes same public function: call_llm(prompt: str, model: Optional[str], max_output_tokens: int, temperature: float)
- Reads MULTI_GEMINI_KEYS env var (comma-separated) for rotation dynamically at call time.
- Tries SDK (from google.genai) when available; otherwise can fallback to HTTP if desired.
- On quota/429 errors rotates to next key and retries with exponential backoff.
- If all keys fail or SDK not present, returns your deterministic stub (meta.source: "stub-fallback").
"""

import os
import json
import time
import random
import threading
import logging
from typing import Any, Dict, Optional, List, Tuple

# dotenv to load .env locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Logging
logger = logging.getLogger("rook_orchestrator.llm_client")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.DEBUG if os.getenv("DEBUG_MODE", "False").lower() in ("1", "true") else logging.INFO)

# Try import of google-genai (the older/alternate import you used)
try:
    from google import genai  # type: ignore
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False
    logger.info("google.genai not available — will fallback to stub if no keys/HTTP.")

# ENV defaults (read directly where needed)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY_SINGLE = os.environ.get("GEMINI_API_KEY")  # backwards-compat
# NOTE: MULTI_GEMINI_KEYS is read dynamically inside _load_key_list
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "4"))
LLM_BACKOFF_BASE = float(os.environ.get("LLM_BACKOFF_BASE", "0.8"))
USE_SDK = GENAI_AVAILABLE

def _stub_response(prompt: str) -> Dict[str, Any]:
    # preserve your previous deterministic fallback behaviour
    if "high_cpa" in prompt or "CPA" in prompt or "cost increase" in prompt:
        plan = [{"action_type":"adjust_budget","campaign_id":"leadgen_nov","adjustment":-0.2,"reason":"Reduce spend to control CPA","confidence":0.7},
                {"action_type":"create_task","task":"Investigate creatives for leadgen_nov","assignee":"marketing_lead","reason":"Possible creative fatigue","confidence":0.5}]
    elif "overload" in prompt or "overloaded" in prompt or "dev_overload" in prompt:
        plan = [{"action_type":"reassign_task","task_id":"t123","from":"dev_ajay","to":"dev_sana","reason":"Balance load","confidence":0.8},
                {"action_type":"draft_email","to":"client@example.com","subject":"Timeline update","body":"We recommend a 3-day extension to ensure quality.","confidence":0.6}]
    else:
        plan = [{"action_type":"create_task","task":"Review campaign performance","assignee":"marketing_lead","reason":"Periodic check","confidence":0.4}]
    return {"text": json.dumps({"plan": plan}), "meta": {"source":"stub"}}

# -----------------------
# Key rotation utilities
# -----------------------
class APIKeyRotation:
    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("APIKeyRotation requires at least one key.")
        self._keys = list(keys)
        self._lock = threading.Lock()
        self._index = random.randrange(len(self._keys))

    def current(self) -> str:
        with self._lock:
            return self._keys[self._index]

    def rotate(self) -> str:
        with self._lock:
            self._index = (self._index + 1) % len(self._keys)
            return self._keys[self._index]

    def mark_dead(self, key: str):
        # Simple: rotate away (we could implement cooldown pools later)
        self.rotate()

    def all_keys(self) -> List[str]:
        return list(self._keys)

def _load_key_list() -> List[str]:
    """
    Read MULTI_GEMINI_KEYS (CSV) from the current environment dynamically.
    Falls back to GEMINI_API_KEY if MULTI_GEMINI_KEYS not set.
    """
    multi_raw = os.getenv("MULTI_GEMINI_KEYS", "") or ""
    keys = [k.strip() for k in multi_raw.split(",") if k.strip()]
    if not keys and GEMINI_API_KEY_SINGLE:
        keys = [GEMINI_API_KEY_SINGLE.strip()]
    return keys

def _mask_key(key: str) -> str:
    if not key:
        return "EMPTY"
    if len(key) <= 8:
        return f"...{key}"
    return f"...{key[-6:]}"

# -----------------------
# Internal call helpers
# -----------------------
def _is_quota_error(exc: Exception, raw_status: Optional[int] = None) -> bool:
    if raw_status == 429:
        return True
    msg = str(exc).lower()
    if "quota" in msg or "rate limit" in msg or "429" in msg or "quotaexceeded" in msg:
        return True
    return False

def _call_sdk_once(prompt: str, key: str, model: str, max_output_tokens: Optional[int], temperature: float) -> Tuple[Dict[str, Any], Any]:
    """
    Call the SDK once and return (parsed_text_dict, raw_response_object)
    Adapted for the client.models.generate_content(...) you used previously.
    """
    if key:
        os.environ["GEMINI_API_KEY"] = key
    client = genai.Client()  # may use ADC or GEMINI_API_KEY
    config = {"temperature": float(temperature), "max_output_tokens": None if max_output_tokens is None else int(max_output_tokens)}
    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    text = getattr(resp, "text", None)
    if text is None:
        if isinstance(resp, dict):
            text = resp.get("text") or resp.get("outputs") or json.dumps(resp)
        else:
            text = str(resp)
    return {"text": text}, resp

def _call_http_once(prompt: str, key: str, model: str, max_output_tokens: Optional[int], temperature: float) -> Tuple[Dict[str, Any], Any]:
    import requests
    url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model}:generate"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"input": prompt}
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}: {resp.text}", resp.status_code)
    data = resp.json()
    text = ""
    if isinstance(data, dict):
        text = data.get("output", "") or data.get("text", "") or json.dumps(data)
    else:
        text = str(data)
    return {"text": text}, data

# -----------------------
# Public function (keeps your interface)
# -----------------------
def call_llm(prompt: str,
             model: Optional[str] = None,
             max_output_tokens: int = 1024,
             temperature: float = 0.2) -> Dict[str, Any]:
    """
    Call Gemini (with multi-key rotation). Returns dict:
      { "text": str, "raw": raw_obj_or_dict, "meta": {...} }

    - Keeps same signature as previous implementation so RookOrchestrator and run_demo.py don't change.
    - Uses MULTI_GEMINI_KEYS env var (CSV) if present; otherwise uses GEMINI_API_KEY.
    - Falls back to deterministic stub if SDK not available or all keys fail.
    """

    model = model or GEMINI_MODEL
    keys = _load_key_list()

    # If no keys and no SDK, return stub immediately
    if not keys and not USE_SDK:
        logger.info("No API keys and SDK not available — returning stub response.")
        stub = _stub_response(prompt)
        stub["meta"]["source"] = "stub"
        return stub

    rotation = None
    if keys:
        rotation = APIKeyRotation(keys)

    last_exc = None
    for attempt in range(1, max(1, LLM_MAX_RETRIES) + 1):
        key = rotation.current() if rotation else (GEMINI_API_KEY_SINGLE or "")
        logger.debug(f"[LLM attempt {attempt}] using key { _mask_key(key) } model={model}")

        try:
            if USE_SDK:
                raw_text_obj, raw_resp = _call_sdk_once(prompt, key, model, max_output_tokens, temperature)
            else:
                raw_text_obj, raw_resp = _call_http_once(prompt, key, model, max_output_tokens, temperature)

            meta = {"model": model, "source": "gemini", "api_key_masked": _mask_key(key)}
            result = {
                "text": raw_text_obj.get("text"),
                "raw": raw_resp,
                "meta": meta
            }
            return result

        except Exception as e:
            last_exc = e
            raw_status = None
            if isinstance(e, tuple) and len(e) > 1 and isinstance(e[1], int):
                raw_status = e[1]

            if _is_quota_error(e, raw_status):
                logger.warning(f"Quota/rate-limit detected for key {_mask_key(key)}: {e}. Rotating key and retrying.")
                if rotation:
                    rotation.mark_dead(key)
                backoff = LLM_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.debug(f"Backoff {backoff:.2f}s before retry.")
                time.sleep(backoff)
                continue
            else:
                logger.exception(f"LLM call error (non-quota) with key {_mask_key(key)}: {e}. Rotating and retrying.")
                if rotation:
                    rotation.rotate()
                time.sleep(LLM_BACKOFF_BASE * (2 ** (attempt - 1)))
                continue

    logger.error("All LLM attempts exhausted. Falling back to stub-fallback response.")
    stub = _stub_response(prompt)
    stub_meta = stub.get("meta", {})
    stub_meta["source"] = "stub-fallback"
    stub_meta["error"] = str(last_exc)
    stub["meta"] = stub_meta
    return stub
