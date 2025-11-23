# rook_orchestrator/agents/strategy_agent.py
from typing import Dict, Any
import json
from ..utils.llm_client import call_llm

def plan_actions(board_state: Dict[str,Any], insights: Dict[str,Any]) -> Dict[str,Any]:
    # Build prompt that contains board + insights; stubbed LLM will look for keywords.
    prompt = f"Board:\n{json.dumps(board_state)}\nInsights:\n{json.dumps(insights)}\n"
    llm_resp = call_llm(prompt)
    plan = []
    try:
        plan_json = json.loads(llm_resp.get("text", "{}"))
        plan = plan_json.get("plan", [])
    except Exception:
        # fallback rule-based plan from insights
        for r in insights.get("risks", []):
            if r["issue"] == "high_cpa":
                plan.append({
                    "action_type":"adjust_budget",
                    "campaign_id": r["campaign_id"],
                    "adjustment": -0.2,
                    "reason": r["note"],
                    "confidence": 0.6
                })
    return {"plan": plan, "llm_raw": llm_resp}
