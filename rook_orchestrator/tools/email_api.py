# rook_orchestrator/tools/email_api.py

import json
import os
import uuid
import datetime
import logging
import re
from typing import List, Optional, Tuple, Dict, Any

from ..utils.llm_client import call_llm_structured, get_text_from_sdk_resp

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
# Helper: extract JSON snippet from raw LLM text
# ----------------------------------------------------------
def _extract_json_snippet(text: str) -> Optional[dict]:
    if not text:
        return None

    # Look for ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            pass

    # Look for ``` ... ```
    m = re.search(r"```\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            pass

    # Naive find of first JSON object
    start = text.find("{")
    if start != -1:
        for end in range(start + 50, min(start + 5000, len(text))):
            if text[end] == "}":
                candidate = text[start:end + 1]
                try:
                    return json.loads(candidate)
                except:
                    pass

    return None


# ----------------------------------------------------------
# Helper: normalize parsed email
# ----------------------------------------------------------
def _normalize_parsed(p: dict, raw: dict, subject_hint: str) -> dict:
    result = {
        "to": p.get("to") or "client@example.com",
        "subject": p.get("subject") or subject_hint or "Update",
        "body": p.get("body") or "",
        "meta": p.get("meta") or {},
    }

    # If body and subject missing → try raw JSON extraction
    raw_text = ""
    try:
        if isinstance(raw, dict):
            raw_text = raw.get("raw") or raw.get("text") or ""
        else:
            raw_text = str(raw)
    except:
        raw_text = str(raw)

    if not result["body"] or not result["subject"]:
        parsed_raw = _extract_json_snippet(raw_text)
        if parsed_raw:
            result["to"] = parsed_raw.get("to", result["to"])
            result["subject"] = parsed_raw.get("subject", result["subject"])
            result["body"] = parsed_raw.get("body", result["body"])
            result["meta"]["from_raw_json"] = True
            return result

        # Fallback: extract first lines
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        if not result["subject"] and lines:
            result["subject"] = lines[0][:80]
        if not result["body"] and len(lines) > 1:
            result["body"] = "\n".join(lines[1:])[:1200]
        elif not result["body"]:
            result["body"] = raw_text[:1200]

    return result


# ----------------------------------------------------------
# Email API CLASS
# ----------------------------------------------------------
class EmailAPI:

    def __init__(self):
        pass

    # ------------------------------------------------------
    # Generate parallel drafts
    # ------------------------------------------------------
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

    # ------------------------------------------------------
    # Merge drafts
    # ------------------------------------------------------
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

    # ------------------------------------------------------
    # Email generation (parallel)
    # ------------------------------------------------------
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
        raw_worker_calls = []
        parsed_workers = []

        for i in range(workers):
            sys, prompt = self._worker_prompt(subject_hint, notes)
            parsed, raw = call_llm_structured(
                scenario_text=prompt,
                system_instruction=sys,
                max_output_tokens=worker_tokens,
                temperature=0.3,
                repair_max_tokens=150
            )
            parsed_workers.append(parsed or {})
            raw_worker_calls.append(raw or {})

        # Normalize workers
        clean_workers = []
        for p, r in zip(parsed_workers, raw_worker_calls):
            clean_workers.append(_normalize_parsed(p, r, subject_hint))

        # 2) MERGE
        sys_merge, prompt_merge = self._merge_prompt(clean_workers, subject_hint)
        parsed_merge, raw_merge = call_llm_structured(
            scenario_text=prompt_merge,
            system_instruction=sys_merge,
            max_output_tokens=merge_tokens,
            temperature=0.0,
            repair_max_tokens=150
        )

        # Normalize
        final = _normalize_parsed(parsed_merge or {}, raw_merge or {}, subject_hint)

        # 3) REPAIR if broken
        bad_words = ["sdk_http_response", "Candidate(", "MAX_TOKENS", "GenerateContentResponse"]
        if any(b in (final.get("body", "") + final.get("subject", "")) for b in bad_words):
            logger.info("Final looks corrupted → running repair LLM...")

            raw_text = json.dumps({"workers": raw_worker_calls}, indent=2)

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
            "raw_workers": raw_worker_calls,
            "raw_merge_meta": raw_merge.get("meta") if isinstance(raw_merge, dict) else {},
            "created_at": datetime.datetime.utcnow().isoformat()
        }

        path = _save_json(log_record, "final_email")
        final["meta"]["log_path"] = path

        return final
