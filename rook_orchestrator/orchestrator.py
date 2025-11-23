# rook_orchestrator/orchestrator.py
from rook_orchestrator.agents.observer import observe
from rook_orchestrator.agents.analytics_agent import analyze_metrics
from rook_orchestrator.agents.strategy_agent import plan_actions
from rook_orchestrator.agents.action_agent import execute_plan
from rook_orchestrator.tools.analytics_api import AnalyticsAPI
import json, datetime, sys, os

class RookOrchestrator:
    def __init__(self, analytics_data=None):
        self.memory = {"decisions": []}
        self.analytics_api = AnalyticsAPI(analytics_data or {})

    def run_cycle(self, manual_inputs: dict):
        analytics = self.analytics_api.fetch()
        board = observe(manual_inputs, analytics)
        insights = analyze_metrics(board["analytics"])
        plan_resp = plan_actions(board, insights)
        plan = plan_resp.get("plan", [])
        results = execute_plan(plan, self.analytics_api)
        log = {
            "date": str(datetime.date.today()),
            "board": board,
            "insights": insights,
            "plan": plan,
            "results": results
        }
        self.memory["decisions"].append(log)
        return log

if __name__ == "__main__":
    demo_path = os.path.join(os.path.dirname(__file__), "..", "demo_inputs", "campaign_spike.json")
    demo_path = os.path.normpath(demo_path)
    if not os.path.exists(demo_path):
        print("Demo input not found:", demo_path)
        sys.exit(1)
    demo = json.load(open(demo_path))
    r = RookOrchestrator(demo.get("analytics"))
    out = r.run_cycle(demo.get("inputs", {}))
    print(json.dumps(out, indent=2))
