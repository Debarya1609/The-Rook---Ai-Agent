# rook_orchestrator/utils/llm_client.py
"""
Gemini (Google GenAI) client wrapper for The Rook.

Uses google-genai Client.models.generate_content(..., config={...}) when available.
Falls back to deterministic stub on any error and returns error text in meta.
"""

import os, json
from typing import Any, Dict, Optional

# dotenv to load .env locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Try import of google-genai
try:
    from google import genai  # type: ignore
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

def _stub_response(prompt: str) -> Dict[str, Any]:
    # deterministic fallback (same as previous stub)
    if "high_cpa" in prompt or "CPA" in prompt or "cost increase" in prompt:
        plan = [{"action_type":"adjust_budget","campaign_id":"leadgen_nov","adjustment":-0.2,"reason":"Reduce spend to control CPA","confidence":0.7},
                {"action_type":"create_task","task":"Investigate creatives for leadgen_nov","assignee":"marketing_lead","reason":"Possible creative fatigue","confidence":0.5}]
    elif "overload" in prompt or "overloaded" in prompt or "dev_overload" in prompt:
        plan = [{"action_type":"reassign_task","task_id":"t123","from":"dev_ajay","to":"dev_sana","reason":"Balance load","confidence":0.8},
                {"action_type":"draft_email","to":"client@example.com","subject":"Timeline update","body":"We recommend a 3-day extension to ensure quality.","confidence":0.6}]
    else:
        plan = [{"action_type":"create_task","task":"Review campaign performance","assignee":"marketing_lead","reason":"Periodic check","confidence":0.4}]
    return {"text": json.dumps({"plan": plan}), "meta": {"source":"stub"}}

def call_llm(prompt: str,
             model: Optional[str] = None,
             max_output_tokens: int = 1024,
             temperature: float = 0.2) -> Dict[str, Any]:
    """
    Call Gemini via google-genai. If SDK/key not available or error occurs,
    return deterministic stub response with error metadata.
    Returns: dict with 'text' (string) and optional 'raw' and 'meta'.
    """
    model = model or GEMINI_MODEL

    # If genai not available, fallback immediately
    if not GENAI_AVAILABLE:
        return _stub_response(prompt)

    try:
        # Optionally set API key via env (genai Client can use ADC or api_key)
        if GEMINI_API_KEY:
            os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

        client = genai.Client()  # uses ADC or GEMINI_API_KEY

        # Correct call: put sampling/length params inside 'config'
        config = {
            "temperature": float(temperature),
            "max_output_tokens": None if max_output_tokens is None else int(max_output_tokens)
        }

        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )

        # resp typically exposes .text
        text = getattr(resp, "text", None)
        if text is None:
            if isinstance(resp, dict):
                text = resp.get("text") or resp.get("outputs") or json.dumps(resp)
            else:
                text = str(resp)

        return {"text": text, "raw": resp, "meta": {"model": model, "source": "gemini"}}

    except Exception as e:
        # return stub but include error info so you can inspect what went wrong
        stub = _stub_response(prompt)
        stub["meta"]["error"] = str(e)
        stub["meta"]["source"] = "stub-fallback"
        return stub
