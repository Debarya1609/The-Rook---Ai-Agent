# rook_orchestrator/agents/strategy_agent.py
"""
Strategy agent for The Rook — robust JSON extraction + normalization.

Returns: {"plan": [normalized_actions], "llm_raw": llm_resp}
"""

import os
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from ..utils.llm_client import call_llm

PROMPT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "strategy_prompt.txt")
)

# Acceptable canonical action types
CANON_ACTIONS = {
    "adjust_budget",
    "create_task",
    "reassign_task",
    "draft_email",
    "pause_campaign",
    "increase_budget",
    "run_analysis",
    "send_alert",
    "schedule_post"
}

# -------------------------
# Prompt loader
# -------------------------
def _load_prompt() -> str:
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return (
            "SYSTEM: You are The Rook agent. RETURN EXACTLY ONE JSON OBJECT with keys: actions (array) and summary (string). "
            "Each action should be an object with keys: action_type, details (object), reason (string), confidence (float). "
            "USER: Board:{board_state_here} Insights:{insights_here}"
        )

# -------------------------
# Extract JSON from model text
# -------------------------
def _strip_code_fence(text: str) -> str:
    if not text:
        return text
    s = text.strip()
    # fenced ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fenced ~~~
    m = re.search(r"~~~(?:json)?\s*([\s\S]*?)\s*~~~", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # inline `...`
    m = re.search(r"`([^`]+)`", s)
    if m:
        return m.group(1).strip()
    return s

def _extract_first_json(text: str) -> Optional[Any]:
    """
    Try to parse the first JSON object/array in given text.
    Returns Python object or None.
    """
    if not text or not isinstance(text, str):
        return None
    cleaned = _strip_code_fence(text)

    # First try full parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Then try to find first balanced { ... } or [ ... ]
    # find index of first { or [
    start = None
    open_ch = None
    for i, ch in enumerate(cleaned):
        if ch in ("{", "["):
            start = i
            open_ch = ch
            break
    if start is None:
        return None

    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    for j in range(start, len(cleaned)):
        if cleaned[j] == open_ch:
            depth += 1
        elif cleaned[j] == close_ch:
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : j + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    break

    # last-resort greedy regex
    m = re.search(r"(\{[\s\S]*\})", cleaned)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None

# -------------------------
# Normalize action shapes
# -------------------------
def _normalize_action(raw: Any, campaign_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert raw action (string, list, or dict) to canonical Rook action dict.
    Canonical shape:
    {
      "id": "<uuid>",
      "action_type": "...",
      "details": {...},
      "reason": "...",
      "confidence": 0.6,
      ...
    }
    """
    def make_base():
        return {
            "id": str(uuid.uuid4()),
            "action_type": "create_task",
            "details": {},
            "reason": None,
            "confidence": 0.6,
            "expected_impact": None,
            "preconditions": [],
            "rollback_plan": None,
            "estimated_time_hours": 0.0,
        }

    # Strings -> create_task
    if isinstance(raw, str):
        s = raw.strip()
        base = make_base()
        base["details"] = {"task": s}
        base["reason"] = s
        return base

    # List -> combine into single create_task with items
    if isinstance(raw, list):
        parts = []
        for it in raw:
            if isinstance(it, str):
                parts.append(it.strip())
            elif isinstance(it, dict):
                parts.append(it.get("action") or it.get("title") or json.dumps(it))
            else:
                parts.append(str(it))
        combined = " | ".join([p for p in parts if p])[:800]
        base = make_base()
        base["details"] = {"task": combined, "items": parts}
        base["reason"] = "Converted from list of actions"
        return base

    # Dict -> try to map fields
    if isinstance(raw, dict):
        base = make_base()
        # determine action_type
        raw_type = raw.get("action_type") or raw.get("type") or raw.get("action")
        if isinstance(raw_type, str):
            rt = raw_type.lower()
            # simple mapping
            mapping = {
                "move_card": "create_task",
                "create_card": "create_task",
                "add_comment": "create_task",
                "assign_member": "reassign_task",
                "adjust_budget": "adjust_budget",
                "reassign_task": "reassign_task",
                "draft_email": "draft_email",
                "pause_campaign": "pause_campaign",
                "run_analysis": "run_analysis",
                "investigation": "create_task",
                "analysis": "create_task",
                "audit": "create_task",
                "communication": "draft_email",
            }
            base["action_type"] = mapping.get(rt, rt if rt in CANON_ACTIONS else "create_task")
        else:
            base["action_type"] = "create_task"

        # details: start with explicit details if dict
        details = {}
        if isinstance(raw.get("details"), dict):
            details.update(raw.get("details"))

        # bring common top-level fields into details if present
        for key in ("task", "task_id", "campaign_id", "adjustment", "assignee", "to", "subject", "body", "due_date", "member_id"):
            if raw.get(key) is not None:
                details[key] = raw.get(key)

        # flatten nested details.details
        if isinstance(details.get("details"), dict):
            inner = details.pop("details")
            for kk, vv in inner.items():
                if kk not in details:
                    details[kk] = vv
                else:
                    details[f"_inner_{kk}"] = vv

        base["details"] = details
        base["reason"] = raw.get("reason") or raw.get("summary") or raw.get("title") or raw.get("action")
        try:
            base["confidence"] = float(raw.get("confidence")) if raw.get("confidence") is not None else base["confidence"]
        except Exception:
            base["confidence"] = base["confidence"]

        # convenience expose common keys
        for k in ("campaign_id", "adjustment", "assignee", "to", "subject", "body", "due_date", "task_id"):
            if k in details:
                base[k] = details[k]

        # map member_id -> assignee
        if "member_id" in details and "assignee" not in base:
            base["assignee"] = details.get("member_id")

        # if no details and raw has 'action' text, use it
        if not base["details"] and raw.get("action"):
            base["details"] = {"task": raw.get("action")}
            if not base["reason"]:
                base["reason"] = raw.get("action")

        # if campaign_hint available and campaign_id missing -> attach hint
        if campaign_hint and "campaign_id" not in base and "campaign_id" not in base["details"]:
            base["details"].setdefault("campaign_id", campaign_hint)
            base.setdefault("campaign_id", campaign_hint)

        return base

    # Fallback: stringified
    base = make_base()
    base["details"] = {"task": str(raw)[:800]}
    base["reason"] = "Fallback conversion"
    return base

# -------------------------
# Synthesize from board-like parsed JSON
# -------------------------
def _synthesize_from_board_like(parsed: Dict[str, Any], insights: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions = []
    # look for campaigns -> risks
    for c in parsed.get("campaigns", []) or []:
        if not isinstance(c, dict):
            continue
        risks = c.get("risks") or []
        if isinstance(risks, dict):
            risks = [risks]
        for r in risks:
            if isinstance(r, dict) and r.get("issue") == "high_cpa":
                actions.append({
                    "action_type": "adjust_budget",
                    "details": {"campaign_id": c.get("campaign_id"), "adjustment": -0.2},
                    "reason": r.get("note") or "High CPA detected",
                    "confidence": 0.6
                })
    # global insights fallback
    for r in (insights.get("risks") or []):
        if isinstance(r, dict) and r.get("issue") == "high_cpa":
            actions.append({
                "action_type": "adjust_budget",
                "details": {"campaign_id": r.get("campaign_id"), "adjustment": -0.2},
                "reason": r.get("note"),
                "confidence": 0.6
            })
    return actions

# -------------------------
# Main: plan_actions
# -------------------------
def plan_actions(board_state: Dict[str, Any], insights: Dict[str, Any], use_llm: bool = False) -> Dict[str, Any]:
    """
    Build prompt, call LLM (via utils.call_llm), extract/normalize actions, and return:
    { "plan": [normalized_actions], "llm_raw": llm_resp }
    """
    prompt_template = _load_prompt()
    board_json = json.dumps(board_state, separators=(",", ":"), ensure_ascii=False)
    insights_json = json.dumps(insights or {}, separators=(",", ":"), ensure_ascii=False)
    prompt = prompt_template.replace("{board_state_here}", board_json).replace("{insights_here}", insights_json)

    # call LLM
    try:
        llm_resp = call_llm(prompt, model=os.environ.get("GEMINI_MODEL"), max_output_tokens=1500, temperature=0.0)
    except Exception as e:
        # return deterministic fallback plan
        fallback = []
        for r in (insights or {}).get("risks", []):
            if isinstance(r, dict) and r.get("issue") == "high_cpa":
                fallback.append({
                    "action_type": "adjust_budget",
                    "details": {"campaign_id": r.get("campaign_id"), "adjustment": -0.2},
                    "reason": r.get("note"),
                    "confidence": 0.6
                })
        if not fallback:
            fallback = [{"action_type":"create_task","details":{"task":"Review campaign performance"},"reason":"Periodic check","confidence":0.4}]
        return {"plan": [ _normalize_action(a) for a in fallback], "llm_raw": {"text": str(e), "meta": {"source":"error","error": str(e)}}}

    # Ensure llm_resp is a dict with 'text'
    if isinstance(llm_resp, dict):
        llm_text = llm_resp.get("text") or ""
    else:
        llm_text = str(llm_resp)

    parsed = _extract_first_json(llm_text)

    normalized_plan: List[Dict[str, Any]] = []

    # Case A: parsed looks like plan (actions)
    if parsed and isinstance(parsed, dict) and ("actions" in parsed or "plan" in parsed):
        actions_raw = parsed.get("actions") or parsed.get("plan") or []
        # If actions list of strings -> normalize each
        if isinstance(actions_raw, list):
            for item in actions_raw:
                normalized_plan.append(_normalize_action(item, campaign_hint=_infer_single_campaign_id(board_state)))
        else:
            # If it's a dict or other, map accordingly
            normalized_plan.append(_normalize_action(actions_raw, campaign_hint=_infer_single_campaign_id(board_state)))
        return {"plan": normalized_plan, "llm_raw": llm_resp}

    # Case B: parsed board-like -> synthesize
    if parsed and isinstance(parsed, dict):
        synthesized = _synthesize_from_board_like(parsed, insights or {})
        if synthesized:
            normalized_plan = [_normalize_action(a, campaign_hint=_infer_single_campaign_id(board_state)) for a in synthesized]
            return {"plan": normalized_plan, "llm_raw": llm_resp}

    # Case C: no parsed JSON or parsed unknown -> attempt to parse plain text heuristics:
    # Look for lines prefixed with -, * or numbered steps and map to tasks
    heuristics = []
    for line in (llm_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # skip code fences lines
        if line.startswith("```") or line.startswith("~~~"):
            continue
        # common bullet patterns
        if re.match(r"^[\-\*\d\.\)]\s+", line):
            heuristics.append(re.sub(r"^[\-\*\d\.\)]\s+", "", line))
        elif len(line) > 30 and ("audit" in line.lower() or "investigate" in line.lower() or "budget" in line.lower() or "client" in line.lower()):
            heuristics.append(line)
    if heuristics:
        for h in heuristics:
            normalized_plan.append(_normalize_action(h, campaign_hint=_infer_single_campaign_id(board_state)))
        return {"plan": normalized_plan, "llm_raw": llm_resp}

    # Final fallback based on insights
    final_plan = []
    for r in (insights or {}).get("risks", []):
        if isinstance(r, dict) and r.get("issue") == "high_cpa":
            final_plan.append({"action_type": "adjust_budget", "details": {"campaign_id": r.get("campaign_id"), "adjustment": -0.2}, "reason": r.get("note"), "confidence": 0.6})
    if not final_plan:
        final_plan = [{"action_type":"create_task","details":{"task":"Review campaign performance"},"reason":"Periodic check","confidence":0.4}]
    normalized_plan = [_normalize_action(a, campaign_hint=_infer_single_campaign_id(board_state)) for a in final_plan]
    return {"plan": normalized_plan, "llm_raw": llm_resp}


# -------------------------
# small helper to infer single campaign id from board
# -------------------------
def _infer_single_campaign_id(board_state: Dict[str, Any]) -> Optional[str]:
    try:
        campaigns = board_state.get("campaigns") or board_state.get("analytics", {}).get("campaigns") or board_state.get("board", {}).get("analytics", {}).get("campaigns")
        if isinstance(campaigns, list) and len(campaigns) == 1:
            return campaigns[0].get("campaign_id")
    except Exception:
        pass
    # fallback: check board_state top-level keys
    if isinstance(board_state, dict):
        for k, v in board_state.items():
            if isinstance(v, dict) and "campaign_id" in v:
                return v.get("campaign_id")
    return None
