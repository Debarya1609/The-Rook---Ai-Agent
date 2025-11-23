# rook_orchestrator/agents/analytics_agent.py
from typing import Dict, Any

def analyze_metrics(analytics: Dict[str,Any]) -> Dict[str,Any]:
    insights = {"campaign_insights": [], "risks": []}
    for c in analytics.get("campaigns", []):
        target = c.get("target_cpa", float("inf"))
        cpa = c.get("cpa", 0)
        if target and cpa > target:
            insights["risks"].append({
                "campaign_id": c.get("campaign_id"),
                "issue": "high_cpa",
                "urgency": 8,
                "note": f"CPA {cpa} > target {target}"
            })
        if c.get("trend") == "down":
            insights["campaign_insights"].append({
                "campaign_id": c.get("campaign_id"),
                "recommendation": "investigate_creatives",
                "confidence": 0.6
            })
    return insights
