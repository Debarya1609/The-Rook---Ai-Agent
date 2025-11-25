# rook_orchestrator/agents/action_agent.py
from typing import List, Dict, Any
from ..tools.task_api import TaskAPI
from ..tools.analytics_api import AnalyticsAPI
from ..tools.email_api import EmailAPI

task_api = TaskAPI()
email_api = EmailAPI()

def _build_task_payload_from_details(p: Dict[str,Any]) -> Dict[str,Any]:
    """
    Build a sensible task payload from action dict or its 'details' sub-dict.
    """
    details = p.get("details", {}) or {}
    # pick sensible fields in order of preference
    task_title = p.get("task") or details.get("card_title") or details.get("task") or details.get("card_title") or details.get("description") or details.get("title")
    assignee = p.get("assignee") or details.get("assignee") or p.get("to") or details.get("member_id") or details.get("to")
    due = p.get("due") or details.get("due_date") or details.get("due")
    payload = {
        "task": task_title or "Auto-created task",
        "assignee": assignee,
        "due": due,
        "meta": details
    }
    # if original included explicit task_id, carry it forward
    if p.get("task_id"):
        payload["task_id"] = p.get("task_id")
    return payload

def execute_plan(plan: List[Dict[str,Any]], analytics_api: AnalyticsAPI) -> Dict[str,Any]:
    results = []
    for p in plan:
        t = p.get("action_type")
        if t == "create_task":
            payload = _build_task_payload_from_details(p)
            res = task_api.create_task(payload)
            results.append({"action": p, "result": res})
        elif t == "adjust_budget":
            # expect campaign_id + adjustment
            cid = p.get("campaign_id") or (p.get("details") or {}).get("campaign_id")
            adj = p.get("adjustment") or (p.get("details") or {}).get("adjustment")
            if cid is None or adj is None:
                results.append({"action": p, "result": {"ok": False, "reason": "missing_campaign_or_adjustment"}})
            else:
                res = analytics_api.adjust_budget(cid, adj)
                results.append({"action": p, "result": res})
        elif t == "reassign_task":
            # prefer explicit task_id; else try details or find-by-assignee
            task_id = p.get("task_id") or (p.get("details") or {}).get("task_id")
            new_assignee = p.get("to") or (p.get("details") or {}).get("to") or p.get("assignee")
            if not task_id:
                # try find by assignee-from field
                from_field = p.get("from") or (p.get("details") or {}).get("from") or (p.get("details") or {}).get("member_id")
                if from_field:
                    found = task_api.find_task_by_assignee(str(from_field))
                    if found:
                        task_id = found
            if task_id:
                res = task_api.reassign(task_id, new_assignee)
                results.append({"action": p, "result": res})
            else:
                # fallback: create a new task assigned to the 'to' person
                payload = _build_task_payload_from_details(p)
                created = task_api.create_task(payload)
                results.append({"action": p, "result": {"ok": True, "created_task": created}})
        elif t in ("draft_email", "send_email"):
            to = p.get("to") or (p.get("details") or {}).get("to") or "ops@company"
            subject = p.get("subject") or (p.get("details") or {}).get("subject") or "Update"
            body = p.get("body") or (p.get("details") or {}).get("body") or ""
            res = email_api.send(to, subject, body)
            results.append({"action": p, "result": res})
        else:
            results.append({"action": p, "result": {"ok": False, "reason": "unknown_action"}})
    return {"results": results}
