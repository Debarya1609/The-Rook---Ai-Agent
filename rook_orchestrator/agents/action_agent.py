# rook_orchestrator/agents/action_agent.py
from typing import List, Dict, Any
from ..tools.task_api import TaskAPI
from ..tools.analytics_api import AnalyticsAPI
from ..tools.email_api import EmailAPI

task_api = TaskAPI()
email_api = EmailAPI()

def execute_plan(plan: List[Dict[str,Any]], analytics_api: AnalyticsAPI) -> Dict[str,Any]:
    results = []
    for p in plan:
        t = p.get("action_type")
        if t == "create_task":
            res = task_api.create_task(p)
            results.append({"action": p, "result": res})
        elif t == "adjust_budget":
            res = analytics_api.adjust_budget(p["campaign_id"], p["adjustment"])
            results.append({"action": p, "result": res})
        elif t in ("draft_email", "send_email"):
            res = email_api.send(p.get("to", "ops@company"), p.get("subject", "Update"), p.get("body", ""))
            results.append({"action": p, "result": res})
        else:
            results.append({"action": p, "result": {"ok": False, "reason": "unknown_action"}})
    return {"results": results}
