# rook_orchestrator/agents/strategy_agent.py
"""
Robust strategy agent for The Rook.

Responsibilities:
- Load prompt template
- Call LLM with retries (deterministic)
- Extract JSON even from fenced or partial responses
- Validate parsed JSON against a minimal schema
- Normalize different action schemas into canonical Rook actions
- Synthesize safe fallback actions when model returns board-like JSON
- Return dict: {"plan": [normalized_actions], "llm_raw": llm_response}
"""

import os
import json
import re
import time
from typing import Any, Dict, List, Optional

# local llm wrapper
from ..utils.llm_client import call_llm

PROMPT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "strategy_prompt.txt")
)

### ---------------------------
### Prompt loader
### ---------------------------
def _load_prompt() -> str:
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        # fallback prompt (minimal)
        return (
            "SYSTEM: You are The Rook. Output EXACTLY one JSON object with keys: actions (array) and summary (string)."
            " USER: Board:{board_state_here} Insights:{insights_here}"
        )

### ---------------------------
### Utils: strip fences and extract JSON
### ---------------------------
def _strip_code_fence(text: str) -> str:
    if not text:
        return text
    s = text.strip()
    # remove triple-backtick fences (```json ... ```), or triple-tilde
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    fence_match = re.search(r"~~~(?:json)?\s*([\s\S]*?)\s*~~~", s, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    # remove leading "```" or trailing "```" if present
    if s.startswith("```") and s.endswith("```"):
        return s.strip("`").strip()
    # inline code
    inline = re.search(r"`([^`]+)`", s)
    if inline:
        return inline.group(1).strip()
    return s

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try several strategies to extract the first JSON object from text.
    Returns dict or None.
    """
    if not text:
        return None
    cleaned = _strip_code_fence(text)

    # Try direct load
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Find the first balanced top-level {...} block
    start = cleaned.find("{")
    if start == -1:
        return None

    stack = 0
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0:
                candidate = cleaned[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    # continue searching forward in case first braced object is invalid
                    break

    # Last resort: try to find any {...} via regex greedy and parse
    m = re.search(r"(\{[\s\S]*\})", cleaned)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None

### ---------------------------
### Minimal JSON schema validation (no external dependency)
### ---------------------------
def _is_valid_plan_schema(parsed: Dict[str, Any]) -> bool:
    """
    Validates that parsed is a dict with 'actions' as a list and 'summary' as a string.
    Each action should be a dict with 'action_type' (str) and 'details' (dict or omitted).
    This is intentionally conservative.
    """
    if not isinstance(parsed, dict):
        return False
    actions = parsed.get("actions")
    if actions is None or not isinstance(actions, list):
        return False
    for a in actions:
        if not isinstance(a, dict):
            return False
        # accept if action_type or type exists; we'll normalize later
        if not (isinstance(a.get("action_type"), str) or isinstance(a.get("type"), str) or isinstance(a.get("action"), str)):
            # allow a less strict structure if 'details' present
            if "details" not in a:
                return False
        # details if present must be a dict
        if "details" in a and not isinstance(a["details"], dict):
            return False
    # summary should be string (or absent allowed)
    s = parsed.get("summary")
    if s is not None and not isinstance(s, str):
        return False
    return True

### ---------------------------
### Map model-specific action schema --> canonical Rook action
### ---------------------------
def _map_action(raw_action: Any) -> Dict[str, Any]:
    """
    Normalize a raw action (model output) into canonical Rook action:
    {
      "action_type": "...",
      "details": {...},
      "reason": "...",
      "confidence": 0.5,
      ...
    }

    Improvements:
    - If raw_action is a string, convert to create_task with that string as the task and details.task
    - If raw_action is a list of strings, join or create multiple create_task entries (we convert to a single combined task)
    - Flatten nested 'details' keys to avoid details.details
    """
    # If simple string -> create a sensible task
    if isinstance(raw_action, str):
        summary = raw_action.strip()
        return {
            "action_type": "create_task",
            "task": summary[:512],
            "details": {"task": summary},
            "reason": "Converted from string action",
            "confidence": 0.6,
            "expected_impact": None,
            "preconditions": [],
            "rollback_plan": None,
            "estimated_time_hours": 0.0,
        }

    # If the model returned a list (e.g., list of strings), combine into one task with bullets
    if isinstance(raw_action, list):
        # join items into a short task description
        parts = []
        for item in raw_action:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                # prefer 'action' / 'title' if present
                parts.append(str(item.get("action") or item.get("title") or json.dumps(item)))
            else:
                parts.append(str(item))
        combined = " | ".join([p for p in parts if p])[:800]
        return {
            "action_type": "create_task",
            "task": combined,
            "details": {"task": combined, "items": parts},
            "reason": "Converted from list of actions",
            "confidence": 0.6,
            "expected_impact": None,
            "preconditions": [],
            "rollback_plan": None,
            "estimated_time_hours": 0.0,
        }

    # If not a dict by now, fallback
    if not isinstance(raw_action, dict):
        return {
            "action_type": "create_task",
            "task": str(raw_action)[:512],
            "details": {"task": str(raw_action)},
            "reason": "Converted fallback action",
            "confidence": 0.5
        }

    # From here, raw_action is a dict. Safely extract and flatten details.
    # Pull out possible action-type synonyms
    raw_type = raw_action.get("action_type") or raw_action.get("type") or raw_action.get("action")
    raw_type = str(raw_type) if raw_type is not None else None

    type_map = {
        "move_card": "create_task",
        "assign_member": "reassign_task",
        "add_comment": "create_task",
        "create_card": "create_task",
        "set_due_date": "create_task",
        "adjust_budget": "adjust_budget",
        "reassign_task": "reassign_task",
        "create_task": "create_task",
        "draft_email": "draft_email",
        "schedule_post": "schedule_post",
        "pause_campaign": "pause_campaign",
        "run_analysis": "run_analysis",
        # keep lowercase mapping for common values
        "investigation": "create_task",
        "analysis": "create_task",
        "audit": "create_task",
        "communication": "draft_email",
    }

    if raw_type:
        action_type = type_map.get(raw_type, None)
        if action_type is None:
            if raw_type in set(type_map.values()):
                action_type = raw_type
            else:
                action_type = "create_task"
    else:
        action_type = "create_task"

    # Start building normalized object
    norm = {
        "action_type": action_type,
        "details": {},
        "reason": raw_action.get("reason") or raw_action.get("summary") or raw_action.get("title") or raw_action.get("action") or None,
        "expected_impact": raw_action.get("expected_impact"),
        "preconditions": raw_action.get("preconditions", []),
        "rollback_plan": raw_action.get("rollback_plan"),
        "confidence": float(raw_action.get("confidence")) if raw_action.get("confidence") is not None else 0.5,
        "estimated_time_hours": raw_action.get("estimated_time_hours", 0.0),
    }

    # Start with explicit details, ensure dict
    details = {}
    if isinstance(raw_action.get("details"), dict):
        # copy to avoid mutation
        details.update(raw_action.get("details"))

    # Merge other top-level keys into details (excluding reserved ones)
    reserved = {"action_type", "type", "action", "reason", "expected_impact", "preconditions", "rollback_plan", "confidence", "estimated_time_hours", "summary", "task"}
    for k, v in raw_action.items():
        if k in reserved:
            continue
        # avoid nesting details.details: if k == "details" it's already handled
        if k == "details":
            continue
        details[k] = v

    # Flatten nested "details" inside details (if model nested)
    if isinstance(details.get("details"), dict):
        inner = details.pop("details")
        # keys from inner should not overwrite existing top-level details unless empty
        for kk, vv in inner.items():
            if kk not in details:
                details[kk] = vv
            else:
                # if collision, create a namespaced key to preserve info
                details[f"_inner_{kk}"] = vv

    # Respect explicit 'task' or 'task_id' top-level or inside details
    # (bring them to top-level norm for convenience)
    if "task" in raw_action and raw_action.get("task"):
        norm["task"] = raw_action.get("task")
    if "task_id" in raw_action and raw_action.get("task_id"):
        norm["task_id"] = raw_action.get("task_id")

    # merge any details.task into norm.task for executor convenience
    if "task" not in norm and "task" in details:
        norm["task"] = details.get("task")

    # If details ended up empty, but raw_action has an 'action' text field, use it
    if not details and raw_action.get("action"):
        details["task"] = raw_action.get("action")
        # bump confidence slightly when converted from prose
        if not raw_action.get("confidence"):
            norm["confidence"] = max(norm["confidence"], 0.6)
        if not norm["reason"]:
            norm["reason"] = raw_action.get("action")[:200]

    # Finalize details into norm
    norm["details"] = details

    # Expose common keys for convenience (campaign_id, adjustment, assignee, to, subject, body, due_date)
    for top_k in ("campaign_id", "adjustment", "assignee", "to", "subject", "body", "due_date", "member_id"):
        if top_k in details and details.get(top_k) is not None:
            norm[top_k] = details[top_k]

    # Map member_id -> assignee if present
    if "member_id" in details and "assignee" not in norm:
        norm["assignee"] = details["member_id"]

    return norm

### ---------------------------
### LLM call with retries
### ---------------------------
def _call_llm_with_retries(prompt: str, max_output_tokens: int = 2000, temperature: float = 0.0, attempts: int = 3, backoff: float = 0.8):
    """
    Try calling call_llm up to `attempts` times on transient errors.
    Returns the first successful response (dict) or the last response.
    We consider a successful response one where meta.source == 'gemini' (live) or meta.source == 'stub' but no error.
    """
    last_exc = None
    for i in range(attempts):
        try:
            resp = call_llm(prompt, max_output_tokens=max_output_tokens, temperature=temperature)
            meta = resp.get("meta", {}) or {}
            # if gemini responded, return it (even if text needs extraction)
            if meta.get("source") == "gemini":
                return resp
            # if stub but without an error field, accept it
            if meta.get("source") == "stub" and not meta.get("error"):
                return resp
            # if stub-fallback with error, keep retrying
            last_exc = resp
        except Exception as e:
            last_exc = e
        time.sleep(backoff * (2 ** i))
    # return last response or raise if exception
    if isinstance(last_exc, dict):
        return last_exc
    raise Exception(f"LLM call failed after {attempts} attempts: {last_exc}")

### ---------------------------
### Synthesize actions from board-like JSON when no actions provided
### ---------------------------
def _synthesize_from_board(parsed: Dict[str, Any], insights: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    If model returns a board-like JSON (with campaigns or similar), synthesize safe actions,
    prioritizing 'high_cpa' risks to an adjust_budget action.
    """
    actions = []
    # Campaign-level risks
    for c in parsed.get("campaigns", []) or []:
        if not isinstance(c, dict):
            continue
        risks = c.get("risks") or []
        # risks may be list or single dict
        if isinstance(risks, dict):
            risks = [risks]
        for r in risks:
            if not isinstance(r, dict):
                continue
            if r.get("issue") == "high_cpa":
                actions.append({
                    "action_type": "adjust_budget",
                    "details": {"campaign_id": c.get("campaign_id"), "adjustment": -0.2},
                    "reason": r.get("note") or "High CPA detected",
                    "confidence": 0.6
                })
    # Top-level insights fallback
    if not actions:
        for r in (insights.get("risks") or []):
            if isinstance(r, dict) and r.get("issue") == "high_cpa":
                actions.append({
                    "action_type": "adjust_budget",
                    "details": {"campaign_id": r.get("campaign_id"), "adjustment": -0.2},
                    "reason": r.get("note"),
                    "confidence": 0.6
                })
    return actions

### ---------------------------
### Main entry: plan_actions
### ---------------------------
def plan_actions(board_state: Dict[str, Any], insights: Dict[str, Any], use_llm: bool = False) -> Dict[str, Any]:
    """
    Build prompt, call LLM (or stub), extract/validate/normalize actions and return:
    { "plan": [...], "llm_raw": llm_resp }
    """
    prompt_template = _load_prompt()
    # inject compact JSON to the prompt (avoid pretty spacing to save tokens)
    board_json = json.dumps(board_state, separators=(",", ":"), ensure_ascii=False)
    insights_json = json.dumps(insights, separators=(",", ":"), ensure_ascii=False)
    prompt = prompt_template.replace("{board_state_here}", board_json).replace("{insights_here}", insights_json)

    # If use_llm == False, call_llm will return stub per llm_client design
    try:
        llm_resp = _call_llm_with_retries(prompt, max_output_tokens=2000, temperature=0.0, attempts=3)
    except Exception as e:
        # Hard fallback: deterministic small plan
        fallback_plan = []
        for r in (insights.get("risks") or []):
            if isinstance(r, dict) and r.get("issue") == "high_cpa":
                fallback_plan.append({"action_type": "adjust_budget", "campaign_id": r.get("campaign_id"), "adjustment": -0.2, "reason": r.get("note"), "confidence": 0.6})
        if not fallback_plan:
            fallback_plan = [{"action_type":"create_task","task":"Review campaign performance","assignee":"marketing_lead","reason":"Periodic check","confidence":0.4}]
        return {"plan": fallback_plan, "llm_raw": {"text": str(e), "meta": {"source": "error", "error": str(e)}}}

    text = (llm_resp.get("text") or "") if isinstance(llm_resp, dict) else str(llm_resp)
    parsed = _extract_json(text)

    # If parsed valid and matches schema, normalize actions
    if parsed and _is_valid_plan_schema(parsed):
        raw_actions = parsed.get("actions", []) or []
        normalized = []
        for ra in raw_actions:
            normalized.append(_map_action(ra))
        return {"plan": normalized, "llm_raw": llm_resp}

    # If parsed exists but no actions, but it looks like a board -> synthesize
    if parsed and isinstance(parsed, dict):
        synthesized = _synthesize_from_board(parsed, insights)
        if synthesized:
            normalized = [_map_action(a) for a in synthesized]
            return {"plan": normalized, "llm_raw": llm_resp}

    # If model returned something else (string, invalid), try to look for actions keys manually
    # attempt heuristic: if parsed has 'actions' or 'plan' as non-list but convertible
    if parsed:
        maybe_actions = parsed.get("actions") or parsed.get("plan") or []
        if isinstance(maybe_actions, list) and maybe_actions:
            normalized = [_map_action(a) for a in maybe_actions]
            return {"plan": normalized, "llm_raw": llm_resp}

    # Final fallback: use insights-based deterministic plan
    final_plan = []
    for r in (insights.get("risks") or []):
        if isinstance(r, dict) and r.get("issue") == "high_cpa":
            final_plan.append({"action_type": "adjust_budget", "campaign_id": r.get("campaign_id"), "adjustment": -0.2, "reason": r.get("note"), "confidence": 0.6})
    if not final_plan:
        final_plan = [{"action_type":"create_task","task":"Review campaign performance","assignee":"marketing_lead","reason":"Periodic check","confidence":0.4}]

    return {"plan": final_plan, "llm_raw": llm_resp}
