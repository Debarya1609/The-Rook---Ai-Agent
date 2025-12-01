# rook_orchestrator/tools/email_api.py
"""
Email API tool for The Rook.

Generates parallel email drafts using the LLM client, merges them,
and returns a final normalized email JSON object.

This version contains robust handling for raw SDK responses (e.g. GenerateContentResponse)
by coercing them to plain text before any regex / JSON extraction.
"""

import json
import os
import uuid
import datetime
import logging
import re
from typing import List, Optional, Tuple, Dict, Any

from ..utils.llm_client import call_llm_structured  # assumes this returns (parsed, raw)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# Helper: safe JSON save
# ----------------------------------------------------------
def _save_json(obj: dict, prefix: str, folder: str = "logs/emails") -> str:
    os.makedirs(folder, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fn = f"{prefix}_{ts}.json"
    path = os.path.join(folder, fn)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


# ----------------------------------------------------------
# Helper: coerce SDK/raw responses to plain text
# ----------------------------------------------------------
def _coerce_to_text(raw_obj: Any) -> str:
    """
    Convert different kinds of 'raw' responses to a readable string.
    Handles:
      - None
      - plain str
      - dicts with 'text' or 'raw' keys
      - SDK objects like GenerateContentResponse that expose .text or can be str()'ed
    """
    if raw_obj is None:
        return ""

    # If it's already a string
    if isinstance(raw_obj, str):
        return raw_obj

    # If it's bytes
    if isinstance(raw_obj, (bytes, bytearray)):
        try:
            return raw_obj.decode("utf-8", errors="replace")
        except Exception:
            return str(raw_obj)

    # If it's a dict, try to find common fields
    if isinstance(raw_obj, dict):
        # common places where text might live
        for key in ("text", "output", "raw", "response", "result"):
            if key in raw_obj and raw_obj[key] is not None:
                return _coerce_to_text(raw_obj[key])
        # fallback: dump limited JSON
        try:
            return json.dumps(raw_obj, default=str)
        except Exception:
            return str(raw_obj)

    # If it has 'text' attribute (e.g., SDK response)
    text_attr = getattr(raw_obj, "text", None)
    if text_attr:
        try:
            return str(text_attr)
        except Exception:
            pass

    # Some SDK objects may contain candidates/content parts
    # Try to stringify gracefully
    try:
        return str(raw_obj)
    except Exception:
        return ""


# ----------------------------------------------------------
# Helper: extract JSON snippet from raw LLM text
# ----------------------------------------------------------
def _extract_json_snippet(text: Any) -> Optional[dict]:
    """
    Try to locate a JSON object inside `text`.
    Uses only Python-compatible regex (no (?R)) and simple substring heuristics.
    Returns parsed dict if found, else None.
    """
    if text is None:
        return None

    s = _coerce_to_text(text)
    if not s:
        return None

    # 1) Look for fenced ```json { ... } ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 2) Look for fenced ``` { ... } ```
    m = re.search(r"```\s*(\{.*?\})\s*```", s, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 3) Try to parse the entire string as JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # 4) Naive window: from first "{" to last "}"
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = s[first:last + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Give up
    return None


# ----------------------------------------------------------
# Helper: normalize parsed email
# ----------------------------------------------------------
def _normalize_parsed(p: Dict[str, Any], raw: Any, subject_hint: str) -> dict:
    """
    p: parsed dict returned by call_llm_structured (may be {})
    raw: raw SDK response (various types)
    subject_hint: fallback subject
    """
    result = {
        "to": None,
        "subject": None,
        "body": None,
        "meta": {}
    }

    # Fill from parsed structure first (if any)
    if isinstance(p, dict):
        result["to"] = p.get("to") or p.get("recipient") or p.get("email_to")
        result["subject"] = p.get("subject")
        # body may be in 'body' or 'text' or 'content'
        result["body"] = p.get("body") or p.get("text") or p.get("content")

        # copy meta if present
        if "meta" in p and isinstance(p["meta"], dict):
            result["meta"].update(p["meta"])

    # Coerce raw to text for attempts at extraction
    raw_text = _coerce_to_text(raw)

    # If any field missing, try to extract JSON snippet from raw text
    if not result["to"] or not result["subject"] or not result["body"]:
        parsed_raw = _extract_json_snippet(raw_text)
        if parsed_raw:
            result["to"] = result["to"] or parsed_raw.get("to") or parsed_raw.get("recipient")
            result["subject"] = result["subject"] or parsed_raw.get("subject")
            result["body"] = result["body"] or parsed_raw.get("body") or parsed_raw.get("text")
            result["meta"]["from_raw_json"] = True

    # Final fallbacks
    if not result["to"]:
        result["to"] = "client@example.com"

    if not result["subject"]:
        # pick first non-empty line from raw_text or subject hint
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        result["subject"] = subject_hint or (lines[0][:80] if lines else "Update")

    if not result["body"]:
        # use the remainder of raw_text (trim length)
        lines = [ln for ln in raw_text.splitlines() if ln.strip()]
        if len(lines) > 1:
            result["body"] = "\n".join(lines[1:])[:2000]
        else:
            result["body"] = raw_text[:2000]

    return result


# ----------------------------------------------------------
# Email API CLASS
# ----------------------------------------------------------
class EmailAPI:

    def __init__(self):
        pass

    # Worker prompt
    def _worker_prompt(self, subject: str, notes: str) -> Tuple[str, str]:
        system = (
            "You are an email-writing assistant. ALWAYS output EXACT JSON ONLY.\n"
            "Return JSON: {\"to\":\"...\",\"subject\":\"...\",\"body\":\"...\"}\n"
            "Use professional marketing tone. KEEP IT SHORT."
        )

        user_prompt = (
            f"Subject hint: {subject}\n"
            f"Notes: {notes}\n"
            f"Write a short professional email. JSON only."
        )

        return system, user_prompt

    # Merge prompt
    def _merge_prompt(self, drafts: List[dict], subject_hint: str) -> Tuple[str, str]:
        sys = (
            "You are an email merger. You will receive multiple JSON emails.\n"
            "Your job: produce ONE final JSON email: {\"to\":\"...\",\"subject\":\"...\",\"body\":\"...\"}.\n"
            "No extra text. Always valid JSON."
        )

        text = "DRAFTS:\n"
        for i, d in enumerate(drafts):
            text += f"[DRAFT {i+1}]\n{json.dumps(d, indent=2)}\n\n"

        text += f"Subject hint: {subject_hint}\n\nFINAL_JSON:"

        return sys, text

    # Generate email interactive
    def generate_email_interactive(
            self,
            subject_hint: str,
            notes: str,
            workers: int = 3,
            worker_tokens: int = 250,
            merge_tokens: int = 400,
    ) -> dict:

        logger.info(f"Generating {workers} parallel drafts...")

        # 1) Generate worker drafts
        raw_worker_calls: List[Any] = []
        parsed_workers: List[Dict[str, Any]] = []

        for i in range(workers):
            sys_inst, prompt = self._worker_prompt(subject_hint, notes)
            parsed, raw = call_llm_structured(
                scenario_text=prompt,
                system_instruction=sys_inst,
                max_output_tokens=worker_tokens,
                temperature=0.3,
                repair_max_tokens=150
            )
            parsed_workers.append(parsed or {})
            raw_worker_calls.append(raw or {})

        # Normalize workers
        clean_workers: List[dict] = []
        for p, r in zip(parsed_workers, raw_worker_calls):
            try:
                clean_workers.append(_normalize_parsed(p, r, subject_hint))
            except Exception as e:
                logger.warning("Normalization failed for worker, falling back to raw text: %s", e)
                clean_workers.append({
                    "to": "client@example.com",
                    "subject": subject_hint,
                    "body": _coerce_to_text(r)[:1200],
                    "meta": {"normalization_error": str(e)}
                })

        # 2) MERGE
        sys_merge, prompt_merge = self._merge_prompt(clean_workers, subject_hint)
        parsed_merge, raw_merge = call_llm_structured(
            scenario_text=prompt_merge,
            system_instruction=sys_merge,
            max_output_tokens=merge_tokens,
            temperature=0.0,
            repair_max_tokens=150
        )

        final = _normalize_parsed(parsed_merge or {}, raw_merge or {}, subject_hint)

        # 3) REPAIR if corrupted
        bad_words = ["sdk_http_response", "Candidate(", "MAX_TOKENS", "GenerateContentResponse"]
        combined_check = (final.get("body", "") + final.get("subject", ""))
        if any(b in combined_check for b in bad_words):
            logger.info("Final looks corrupted → running repair LLM...")

            raw_text = json.dumps({"workers": [_coerce_to_text(r) for r in raw_worker_calls]}, indent=2)

            sys_repair = (
                "You will be given messy text. Extract and output EXACTLY one JSON:\n"
                "{\"to\":\"...\",\"subject\":\"...\",\"body\":\"...\"}\n"
                "Short, professional marketing email."
            )

            parsed_fix, raw_fix = call_llm_structured(
                scenario_text=raw_text,
                system_instruction=sys_repair,
                max_output_tokens=350,
                temperature=0.0,
                repair_max_tokens=150
            )

            repaired = _normalize_parsed(parsed_fix or {}, raw_fix or {}, subject_hint)
            if repaired.get("body"):
                final = repaired
                final["meta"]["repair_used"] = True

        # Save log
        log_record = {
            "final": final,
            "workers": clean_workers,
            "raw_workers": [_coerce_to_text(r) for r in raw_worker_calls],
            "raw_merge_meta": (raw_merge.meta if hasattr(raw_merge, "meta") else {}),
            "created_at": datetime.datetime.utcnow().isoformat()
        }

        path = _save_json(log_record, "final_email")
        final["meta"]["log_path"] = path

        return final
