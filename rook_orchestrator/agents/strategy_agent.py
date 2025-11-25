# rook_orchestrator/agents/strategy_agent.py
from typing import Dict, Any, Optional, List
import json, os, re
from ..utils.llm_client import call_llm

PROMPT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "strategy_prompt.txt"))

def _load_prompt():
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "Board:\n{board_state_here}\nInsights:\n{insights_here}\nRespond with JSON."

def _strip_code_fence(text: str) -> str:
    """
    Remove surrounding markdown code fences (```json ... ```), or single backticks.
    """
    if not text:
        return text
    # remove leading/trailing whitespace
    s = text.strip()
    # common fences: ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    # some models use triple tildes
    fence_match = re.search(r"~~~(?:json)?\s*([\s\S]*?)\s*~~~", s, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    # single-line code `...`
    inline = re.search(r"`([^`]+)`", s)
    if inline:
        return inline.group(1).strip()
    return s

def _extract_json(text: str) -> Optional[dict]:
    """
    Try multiple strategies to extract JSON from model text.
    - strip code fences
    - find the first JSON object {...}
    - try to json.loads directly
    """
    if not text:
        return None
    cleaned = _strip_code_fence(text)
    # Try direct json load
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # If cleaned contains JSON somewhere, extract with regex for first object
    m = re.search(r"(\{[\s\S]*\})", cleaned)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            # last resort: try to balance braces naively
            start = cleaned.find("{")
            if start != -1:
                stack = 0
                for i in range(start, len(cleaned)):
                    if cleaned[i] == "{":
                        stack += 1
                    elif cleaned[i] == "}":
                        stack -= 1
                        if stack == 0:
                            candidate = cleaned[start:i+1]
                            try:
                                return json.loads(candidate)
                            except Exception:
                                break
    return None

def _map_action(raw_action: dict) -> dict:
    """
    Map model-specific action schemas into the internal action format used by Rook.
    Returns a dict with keys: action_type, reason, confidence, rollback_plan, estimated_time_hours, and merged details.
    If action_type is not one of our known enterprise actions, fall back to create_task with details summarised.
    """
    if not isinstance(raw_action, dict):
        return {
            "action_type": "create_task",
            "task": str(raw_action),
            "reason": "Converted from unknown action format by normalizer",
            "confidence": 0.5
        }

    # Known external action -> internal mapping
    type_map = {
        "move_card": "create_task",     # interpret board moves as a follow-up task
        "assign_member": "reassign_task",
        "add_comment": "create_task",
        "create_card": "create_task",
        "set_due_date": "create_task",
        # keep room to extend mappings: trello-style -> rook actions
    }

    # Accept multiple possible keys for action type used by model
    raw_type = raw_action.get("action_type") or raw_action.get("type") or raw_action.get("action")
    raw_type = raw_type if raw_type is None else str(raw_type)

    # Map to our canonical action_type
    action_type = None
    if raw_type:
        action_type = type_map.get(raw_type, None)
        if action_type is None:
            # If model used same names we support, pass-through
            if raw_type in {"adjust_budget", "create_task", "reassign_task", "draft_email", "schedule_post", "pause_campaign", "run_analysis"}:
                action_type = raw_type
            else:
                # default fallback
                action_type = "create_task"
    else:
        action_type = "create_task"

    # Build normalized action
    normalized = {
        "action_type": action_type,
        "reason": raw_action.get("reason") or raw_action.get("summary") or raw_action.get("title") or None,
        "expected_impact": raw_action.get("expected_impact"),
        "preconditions": raw_action.get("preconditions", []),
        "rollback_plan": raw_action.get("rollback_plan"),
        "confidence": raw_action.get("confidence", 0.5),
        "estimated_time_hours": raw_action.get("estimated_time_hours", 0.0)
    }

    # Merge details sensibly depending on raw keys
    # If the model used Trello-like fields, keep them in details
    details = {}
    # copy common card/task fields
    for k in ("card_id", "task_id", "card_title", "task", "description", "comment", "member_id", "list_name", "from_list", "to_list", "from", "to", "campaign_id", "adjustment", "subject", "body", "due_date"):
        if k in raw_action:
            details[k] = raw_action[k]

    # Also include any 'details' dict returned by model
    if isinstance(raw_action.get("details"), dict):
        details.update(raw_action.get("details"))

    # Flatten common mappings:
    # - If there's a card_id or task_id, expose it as task_id for task APIs
    if "card_id" in details and "task_id" not in details:
        details["task_id"] = details["card_id"]

    # Attach details fields to normalized action (flatten top-level where helpful)
    # For Rook, we prefer top-level fields like 'task', 'campaign_id', 'adjustment', 'to'/'assignee'
    # copy them if present
    for top_k in ("task", "task_id", "campaign_id", "adjustment", "subject", "body", "due_date", "to", "assignee"):
        if top_k in details and details.get(top_k) is not None:
            normalized[top_k] = details[top_k]

    # keep full details for debugging/execution
    normalized["details"] = details

    # If model suggested member_id or to (new assignee) map to 'to'/'assignee'
    if "member_id" in details and "to" not in normalized:
        normalized["to"] = details["member_id"]

    # If the model intended a reassignment but we defaulted to create_task, make sure both task and assignee are present
    if normalized["action_type"] == "reassign_task" and "task_id" not in normalized:
        # try to use any id-like field
        if "task" in normalized and isinstance(normalized["task"], str) and normalized["task"].lower().startswith("task-"):
            normalized["task_id"] = normalized["task"]

    return normalized

def plan_actions(board_state: Dict[str,Any], insights: Dict[str,Any], use_llm: bool = False) -> Dict[str,Any]:
    """
    Call LLM (or stub) and return normalized plan: {'plan': [actions], 'llm_raw': ...}
    Defensive: handles noisy/partial 'insights' structures and model outputs.
    """
    prompt_template = _load_prompt()
    prompt = prompt_template.replace("{board_state_here}", json.dumps(board_state)).replace("{insights_here}", json.dumps(insights))

    # ask for larger output and deterministic render
    llm_resp = call_llm(prompt, max_output_tokens=2000, temperature=0.0)

    text = llm_resp.get("text", "") or ""
    parsed = _extract_json(text)

    if parsed and isinstance(parsed, dict):
        raw_actions = parsed.get("actions") or parsed.get("plan") or []
        # If model returned no actions but returned 'campaigns' structure, synthesize actions
        if not raw_actions:
            if "campaigns" in parsed or "summary" in parsed:
                raw_actions = []
                for c in parsed.get("campaigns", []) or []:
                    # campaigns may include risks nested; try to pull them safely
                    for risk in (c.get("risks") or []) if isinstance(c, dict) else []:
                        if isinstance(risk, dict) and risk.get("issue") == "high_cpa":
                            raw_actions.append({
                                "action_type":"adjust_budget",
                                "details":{"campaign_id": c.get("campaign_id") if isinstance(c, dict) else None, "adjustment": -0.2},
                                "reason": risk.get("note"),
                                "confidence": 0.6
                            })
                # fallback to top-level insights if still empty
                if not raw_actions:
                    for r in (insights.get("risks") or []):
                        # tolerate string or dict entries
                        if isinstance(r, dict):
                            if r.get("issue") == "high_cpa":
                                raw_actions.append({
                                    "action_type":"adjust_budget",
                                    "details":{"campaign_id": r.get("campaign_id"), "adjustment": -0.2},
                                    "reason": r.get("note"),
                                    "confidence": 0.6
                                })
                        elif isinstance(r, str):
                            # if it's simply the string "high_cpa" or contains it, act conservatively
                            if "high_cpa" in r:
                                # try to pick campaign_id from insights if available
                                cid = None
                                # try inspect insights.campaign_insights for candidate campaign_id
                                for c_ins in (insights.get("campaign_insights") or []):
                                    if isinstance(c_ins, dict) and c_ins.get("campaign_id"):
                                        cid = c_ins.get("campaign_id")
                                        break
                                raw_actions.append({
                                    "action_type":"adjust_budget",
                                    "details":{"campaign_id": cid, "adjustment": -0.2},
                                    "reason": "Detected high_cpa (string risk)",
                                    "confidence": 0.5
                                })
                        else:
                            # ignore unknown risk shapes
                            continue

        # normalize actions
        normalized_actions = []
        for ra in raw_actions:
            normalized_actions.append(_map_action(ra))
        return {"plan": normalized_actions, "llm_raw": llm_resp}

    # If parsed is empty or malformed, fall back to safe rule-based plan using insights
    plan = []
    for r in (insights.get("risks") or []):
        if isinstance(r, dict):
            if r.get("issue") == "high_cpa":
                plan.append({"action_type":"adjust_budget","campaign_id": r.get("campaign_id"), "adjustment": -0.2, "reason": r.get("note"), "confidence": 0.6})
        elif isinstance(r, str):
            if "high_cpa" in r:
                # best effort: try to get campaign id from campaign_insights
                cid = None
                for c_ins in (insights.get("campaign_insights") or []):
                    if isinstance(c_ins, dict) and c_ins.get("campaign_id"):
                        cid = c_ins.get("campaign_id")
                        break
                plan.append({"action_type":"adjust_budget","campaign_id": cid, "adjustment": -0.2, "reason": "Detected high_cpa (string risk)", "confidence": 0.5})
        else:
            # ignore other shapes
            continue

    if not plan:
        plan.append({"action_type":"create_task","task":"Review campaign performance","assignee":"marketing_lead","reason":"Periodic check","confidence":0.4})

    return {"plan": plan, "llm_raw": llm_resp}
