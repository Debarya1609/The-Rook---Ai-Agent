# rook_orchestrator/tools/analytics_api.py
from typing import Dict, Any

class AnalyticsAPI:
    def __init__(self, analytics_data: Dict[str,Any] = None):
        self.data = analytics_data or {}

    def fetch(self) -> Dict[str,Any]:
        return self.data

    def adjust_budget(self, campaign_id: str, adjustment: float) -> Dict[str,Any]:
        for c in self.data.get("campaigns", []):
            if c.get("campaign_id") == campaign_id:
                old = c.get("daily_spend", 0)
                new = round(old * (1 + adjustment), 2)
                c["daily_spend"] = new
                return {"ok": True, "campaign_id": campaign_id, "old_spend": old, "new_spend": new}
        return {"ok": False, "reason": "campaign_not_found"}
