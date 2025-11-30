# rook_orchestrator/utils/llm_client.py
"""
Gemini (Google GenAI) client wrapper for The Rook.

Exports:
- call_llm(prompt: str, model: Optional[str], max_output_tokens: int, temperature: float) -> Dict
- call_llm_structured(scenario_text: str, system_instruction: Optional[str], ...)

Features:
- multi-key rotation (reads MULTI_GEMINI_KEYS env var)
- SDK (google.genai) usage when available
- HTTP fallback (requests) when SDK missing
- quota detection + rotation + backoff
- deterministic stub fallback
- robust structured JSON extraction + repair flow
"""

import os
import json
import time
import random
import threading
import logging
import re
from typing import Any, Dict, Optional, List, Tuple

# dotenv (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# configure logger
logger = logging.getLogger("rook_orchestrator.llm_client")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.DEBUG if os.getenv("DEBUG_MODE", "False").lower() in ("1", "true") else logging.INFO)

# Try import of google-genai (older style)
try:
    from google import genai  # type: ignore
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False
    logger.info("google.genai SDK not available.")

# ENV defaults and reading helpers
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY_SINGLE = os.getenv("GEMINI_API_KEY")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))
LLM_BACKOFF_BASE = float(os.getenv("LLM_BACKOFF_BASE", "0.8"))
USE_SDK = GENAI_AVAILABLE

MULTI_KEYS_RAW_ENV = "MULTI_GEMINI_KEYS"

def _load_key_list() -> List[str]:
    raw = os.getenv(MULTI_KEYS_RAW_ENV, "") or ""
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys and GEMINI_API_KEY_SINGLE:
        keys = [GEMINI_API_KEY_SINGLE.strip()]
    return keys

def _mask_key(key: str) -> str:
    if not key:
        return "EMPTY"
    return f"...{key[-6:]}" if len(key) > 8 else f"...{key}"

# -----------------------
# deterministic stub fallback (keeps previous behavior)
# -----------------------
def _stub_response(prompt: str) -> Dict[str, Any]:
    if "high_cpa" in prompt or "CPA" in prompt or "cost increase" in prompt:
        plan = [
            {"action_type":"adjust_budget","campaign_id":"leadgen_nov","adjustment":-0.2,"reason":"Reduce spend to control CPA","confidence":0.7},
            {"action_type":"create_task","task":"Investigate creatives for leadgen_nov","assignee":"marketing_lead","reason":"Possible creative fatigue","confidence":0.5}
        ]
    elif "overload" in prompt or "overloaded" in prompt or "dev_overload" in prompt:
        plan = [
            {"action_type":"reassign_task","task_id":"t123","from":"dev_ajay","to":"dev_sana","reason":"Balance load","confidence":0.8},
            {"action_type":"draft_email","to":"client@example.com","subject":"Timeline update","body":"We recommend a 3-day extension to ensure quality.","confidence":0.6}
        ]
    else:
        plan = [{"action_type":"create_task","task":"Review campaign performance","assignee":"marketing_lead","reason":"Periodic check","confidence":0.4}]
    return {"text": json.dumps({"plan": plan}), "meta": {"source":"stub"}}

# -----------------------
# rotation helper
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
        # simple rotate away
        self.rotate()

# -----------------------
# internal SDK / HTTP callers
# -----------------------
def _is_quota_error(exc: Exception, raw_status: Optional[int] = None) -> bool:
    if raw_status == 429:
        return True
    msg = str(exc).lower()
    if "quota" in msg or "rate limit" in msg or "429" in msg or "quotaexceeded" in msg:
        return True
    return False

def _call_sdk_once(prompt: str, key: str, model: str, max_output_tokens: Optional[int], temperature: float) -> Tuple[Dict[str, Any], Any]:
    if key:
        os.environ["GEMINI_API_KEY"] = key
    client = genai.Client()
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
# Backwards-compatible public call: call_llm
# -----------------------
def call_llm(prompt: str,
             model: Optional[str] = None,
             max_output_tokens: int = 1024,
             temperature: float = 0.2) -> Dict[str, Any]:
    """
    Backward-compatible call. Returns dict: { "text": str, "raw": raw_resp, "meta": {...} }
    Uses multi-key rotation, SDK or HTTP as available, and stub fallback.
    """
    model = model or GEMINI_MODEL
    keys = _load_key_list()

    # no keys and no SDK -> stub
    if not keys and not USE_SDK:
        logger.info("No keys and SDK missing -> returning stub")
        stub = _stub_response(prompt)
        stub["meta"]["source"] = "stub"
        return stub

    rotation = APIKeyRotation(keys) if keys else None
    last_exc = None

    for attempt in range(1, max(1, LLM_MAX_RETRIES) + 1):
        key = rotation.current() if rotation else (GEMINI_API_KEY_SINGLE or "")
        logger.debug(f"[LLM attempt {attempt}] using key {_mask_key(key)} model={model}")

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
                logger.warning(f"Quota/rate-limit for key {_mask_key(key)}: {e}. Rotating.")
                if rotation:
                    rotation.mark_dead(key)
                backoff = LLM_BACKOFF_BASE * (2 ** (attempt - 1))
                time.sleep(backoff)
                continue
            else:
                logger.exception(f"Non-quota error with key {_mask_key(key)}: {e}. Rotating and retrying.")
                if rotation:
                    rotation.rotate()
                time.sleep(LLM_BACKOFF_BASE * (2 ** (attempt - 1)))
                continue

    # exhausted -> stub-fallback
    logger.error("All attempts exhausted; returning stub-fallback.")
    stub = _stub_response(prompt)
    stub_meta = stub.get("meta", {})
    stub_meta["source"] = "stub-fallback"
    stub_meta["error"] = str(last_exc)
    stub["meta"] = stub_meta
    return stub

# -----------------------
# Structured helpers (parsing + structured call)
# -----------------------
def get_text_from_sdk_resp(raw_resp: Any) -> str:
    try:
        if hasattr(raw_resp, "candidates"):
            collected = []
            for c in getattr(raw_resp, "candidates", []):
                cnt = getattr(c, "content", None)
                if cnt and hasattr(cnt, "parts"):
                    for p in getattr(cnt, "parts", []):
                        t = getattr(p, "text", None)
                        if t:
                            collected.append(t)
                else:
                    collected.append(str(c))
            if collected:
                return "\n".join(collected)
    except Exception:
        pass
    try:
        txt = getattr(raw_resp, "text", None)
        if txt:
            return txt
    except Exception:
        pass
    return str(raw_resp)

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text or not isinstance(text, str):
        return None
    text_clean = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text_clean = re.sub(r"\s*```", "", text_clean, flags=re.IGNORECASE)
    obj_match = re.search(r"\{(?:.|\s)*\}", text_clean)
    arr_match = re.search(r"\[(?:.|\s)*\]", text_clean)
    candidate = None
    if obj_match and arr_match:
        candidate = obj_match if obj_match.start() < arr_match.start() else arr_match
    else:
        candidate = obj_match or arr_match
    if candidate:
        snippet = candidate.group(0)
        snippet = re.sub(r",\s*}", "}", snippet)
        snippet = re.sub(r",\s*]", "]", snippet)
        try:
            return json.loads(snippet)
        except Exception:
            return None
    start = text_clean.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text_clean)):
        if text_clean[i] == "{":
            depth += 1
        elif text_clean[i] == "}":
            depth -= 1
            if depth == 0:
                snippet = text_clean[start:i+1]
                snippet = re.sub(r",\s*}", "}", snippet)
                try:
                    return json.loads(snippet)
                except Exception:
                    return None
    return None

def call_llm_structured(
    scenario_text: str,
    system_instruction: Optional[str] = None,
    model: Optional[str] = None,
    max_output_tokens: int = 400,
    temperature: float = 0.0,
    repair_max_tokens: int = 200
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Structured wrapper:
    - uses a stronger system instruction with example
    - tries primary call, then extract, then repair via extraction and regeneration
    Returns (parsed_dict, raw_response_dict)
    """
    model = model or GEMINI_MODEL

    # stronger system instruction with example
    sys_inst = system_instruction or (
        "You are an assistant used by an automated agent. Follow these rules EXACTLY:\n"
        "1) Return ONLY ONE valid compact JSON object and NOTHING ELSE (no markdown, no explanation, no backticks).\n"
        "2) The JSON MUST follow this schema exactly:\n"
        "{\"actions\":[{\"action_type\":\"<action>\",\"details\":{...},\"reason\":\"<short>\",\"confidence\":0.0}],\"summary\":\"<short>\"}\n"
        "3) Limit actions to at most 4. Keep all strings short (<=120 chars). Use numeric confidence 0.0-1.0.\n"
        "4) If you cannot propose actions, return {\"actions\":[],\"summary\":\"no_action\"}.\n"
        "5) Do NOT include any other fields or surrounding text.\n"
        "EXAMPLE_OUTPUT: {\"actions\":[{\"action_type\":\"refresh_creatives\",\"details\":{\"campaign_id\":\"SpringSale\",\"notes\":\"test 3 variants\"},\"reason\":\"low CTR\",\"confidence\":0.8}],\"summary\":\"Refresh creatives and test.\"}\n"
    )

    prompt_string = (
        f"{sys_inst}\n\n"
        "INSTRUCTIONS (must be followed exactly):\n"
        "1) RETURN ONLY ONE VALID JSON object and NOTHING ELSE.\n"
        "2) Top-level keys MUST be: \"actions\" (array) and \"summary\" (string).\n"
        "3) Limit actions to at most 4 objects.\n"
        "4) Each action must be: { \"action_type\": \"<one-of-list>\", \"details\": {...}, \"reason\": \"<short>\", \"confidence\": 0.0-1.0 }.\n"
        "5) Keep all strings short (<= 120 chars). If uncertain, use confidence 0.4-0.5.\n"
        "6) If no actions, return {\"actions\": [], \"summary\": \"no_action\"}.\n\n"
        "SCENARIO:\n"
        "===BOARD_START===\n"
        f"{scenario_text}\n"
        "===BOARD_END===\n\n"
        "Return only the compact JSON object as the output."
    )

    # Primary call
    raw = call_llm(prompt_string, model=model, max_output_tokens=max_output_tokens, temperature=temperature)

    raw_resp = raw.get("raw", raw.get("text", ""))
    textual = get_text_from_sdk_resp(raw_resp) if raw_resp is not None else raw.get("text", "")

    # First-pass parse
    parsed = extract_json_from_text(textual)

    # Repair attempt 1: extract JSON from textual repr
    if parsed is None:
        repair_prompt = (
            "Extract and return ONLY the valid JSON object found in the text between TEXT_START and TEXT_END.\n"
            "If no JSON is present, return {\"actions\": [], \"summary\": \"no_action\"}.\n\n"
            "TEXT_START\n"
            + textual +
            "\nTEXT_END\n"
        )
        repair_raw = call_llm(repair_prompt, model=model, max_output_tokens=repair_max_tokens, temperature=0.0)
        repair_text = get_text_from_sdk_resp(repair_raw.get("raw", repair_raw.get("text", "")))
        parsed = extract_json_from_text(repair_text)
        raw["_repair_attempt"] = {
            "attempt": "extract_from_text",
            "repaired_text_snippet": repair_text[:1000],
            "repair_meta": repair_raw.get("meta", {}),
        }

    # Repair attempt 2: regenerate clean JSON from scenario
    if parsed is None:
        regen_prompt = (
            "You did not provide valid JSON earlier. Based on the SCENARIO below, "
            "RE-GENERATE ONLY the compact JSON object that matches the schema exactly.\n\n"
            "REQUIRED_SCHEMA: {\"actions\":[{\"action_type\":\"<action>\",\"details\":{...},\"reason\":\"<short>\",\"confidence\":0.0}],\"summary\":\"<short>\"}\n"
            "Return only the JSON and nothing else. Use up to %d tokens.\n\n"
            "SCENARIO:\n%s\n\nEXAMPLE_OUTPUT: {\"actions\":[{\"action_type\":\"refresh_creatives\",\"details\":{\"campaign_id\":\"SpringSale\",\"notes\":\"test 3 variants\"},\"reason\":\"low CTR\",\"confidence\":0.8}],\"summary\":\"Refresh creatives and test.\"}\n"
        ) % (repair_max_tokens, scenario_text)

        regen_raw = call_llm(regen_prompt, model=model, max_output_tokens=repair_max_tokens, temperature=0.0)
        regen_text = get_text_from_sdk_resp(regen_raw.get("raw", regen_raw.get("text", "")))
        parsed = extract_json_from_text(regen_text)
        raw["_repair_attempt_2"] = {
            "attempt": "regenerate_from_scenario",
            "repaired_text_snippet": regen_text[:1000],
            "repair_meta": regen_raw.get("meta", {}),
        }

    if parsed is None:
        parsed = {"actions": [], "summary": "parse_failed"}
        raw["_parse_status"] = "failed"

    return parsed, raw
