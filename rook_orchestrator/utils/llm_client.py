# rook_orchestrator/utils/llm_client.py
import json

def call_llm(prompt: str, max_tokens: int = 512, temperature: float = 0.2):
    """
    Deterministic stub for initial development/demo.
    Replace this with a real LLM client later.
    Returns a dict: {"text": "<json string>"}
    """
    # Very simple keyword-based stub to return JSON plan
    if "high_cpa" in prompt or "CPA" in prompt or "cost increase" in prompt:
        plan = [
            {"action_type":"adjust_budget","campaign_id":"leadgen_nov","adjustment":-0.2,"reason":"Reduce spend to control CPA","confidence":0.7},
            {"action_type":"create_task","task":"Investigate creatives for leadgen_nov","assignee":"marketing_lead","due":None,"reason":"Possible creative fatigue","confidence":0.5}
        ]
    elif "overload" in prompt or "overloaded" in prompt or "dev_overload" in prompt:
        plan = [
            {"action_type":"reassign_task","task_id":"t123","from":"dev_ajay","to":"dev_sana","reason":"Balance load","confidence":0.8},
            {"action_type":"draft_email","to":"client@example.com","subject":"Timeline update","body":"We recommend a 3-day extension to ensure quality.","confidence":0.6}
        ]
    else:
        plan = [
            {"action_type":"create_task","task":"Review campaign performance","assignee":"marketing_lead","due":None,"reason":"Periodic check","confidence":0.4}
        ]
    return {"text": json.dumps({"plan": plan})}
