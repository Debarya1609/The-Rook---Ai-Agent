# rook_orchestrator/orchestrator.py
from rook_orchestrator.agents.observer import observe
from rook_orchestrator.agents.analytics_agent import analyze_metrics
from rook_orchestrator.agents.strategy_agent import plan_actions
from rook_orchestrator.agents.action_agent import execute_plan
from rook_orchestrator.tools.analytics_api import AnalyticsAPI
import json, datetime, sys, os
from typing import List, Dict, Any
from datetime import timezone

def _sanitize_llm_raw(llm_raw):
    if not llm_raw:
        return None
    if isinstance(llm_raw, dict):
        out = {}
        if "text" in llm_raw:
            out["text"] = llm_raw["text"]
        if "meta" in llm_raw:
            out["meta"] = llm_raw["meta"]
        if "raw" in llm_raw:
            try:
                json.dumps(llm_raw["raw"])
                out["raw"] = llm_raw["raw"]
            except Exception:
                try:
                    out["raw_str"] = str(llm_raw["raw"])
                except Exception:
                    out["raw_str"] = "<unserializable raw>"
        if "raw_str" in llm_raw:
            out["raw_str"] = llm_raw["raw_str"]
        return out
    else:
        try:
            return {"raw_str": str(llm_raw)}
        except Exception:
            return {"raw_str": "<unserializable llm_raw>"}

def _prompt_user_approve(action: Dict[str,Any]) -> bool:
    """
    Prompt user to approve a low-confidence action.
    Returns True if approved, False otherwise.
    - If environment variable AUTO_APPROVE=true, auto-approve.
    - If environment variable AUTO_REJECT=true, auto-reject.
    """
    auto_approve = os.environ.get("AUTO_APPROVE", "").lower() in ("1","true","yes")
    auto_reject = os.environ.get("AUTO_REJECT", "").lower() in ("1","true","yes")
    if auto_approve:
        return True
    if auto_reject:
        return False

    # Interactive approval request
    try:
        print("\nLow-confidence action requires approval:")
        print(json.dumps(action, indent=2))
        ans = input("Approve this action? [y/N]: ").strip().lower()
        return ans in ("y", "yes")
    except Exception:
        # if input not available (non-interactive), reject by default
        return False

def _filter_and_approve_actions(plan: List[Dict[str,Any]], threshold: float = 0.6) -> List[Dict[str,Any]]:
    """
    For actions with confidence < threshold, prompt user; if rejected, drop the action.
    Returns filtered plan (approved actions).
    """
    approved = []
    for a in plan:
        conf = a.get("confidence")
        if conf is None:
            conf = 0.5
        try:
            conf = float(conf)
        except Exception:
            conf = 0.5
        if conf <= threshold:
            ok = _prompt_user_approve(a)

            if ok:
                approved.append(a)
            else:
                # log that we skipped it by adding a marker
                print("Action skipped by user:", a.get("action_type"))
        else:
            approved.append(a)
    return approved

class RookOrchestrator:
    def __init__(self, analytics_data=None):
        self.memory = {"decisions": []}
        self.analytics_api = AnalyticsAPI(analytics_data or {})

    def _save_logs(self, log: Dict[str,Any]) -> None:
        """
        Save llm_raw and full decision to logs with UTC timestamp.
        """
        ts = datetime.datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            os.makedirs("logs/llm_samples", exist_ok=True)
            os.makedirs("logs/decisions", exist_ok=True)
            llm = log.get("llm_raw", {})
            with open(os.path.join("logs","llm_samples", f"llm_{ts}.json"), "w", encoding="utf-8") as f:
                json.dump(llm, f, indent=2)
            with open(os.path.join("logs","decisions", f"decision_{ts}.json"), "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2)
        except Exception as e:
            print("Warning: failed to write logs:", e)

    def run_cycle(self, manual_inputs: dict, use_llm: bool = False, save_logs: bool = False):
        analytics = self.analytics_api.fetch()
        board = observe(manual_inputs, analytics)
        insights = analyze_metrics(board["analytics"])
        plan_resp = plan_actions(board, insights, use_llm=use_llm)
        plan = plan_resp.get("plan", []) or []

        # Human approval gate for low-confidence actions
        plan = _filter_and_approve_actions(plan, threshold=0.6)

        results = execute_plan(plan, self.analytics_api)

        # sanitize llm_raw for safe serialization
        llm_raw_safe = _sanitize_llm_raw(plan_resp.get("llm_raw"))

        log = {
            "date": str(datetime.date.today()),
            "board": board,
            "insights": insights,
            "plan": plan,
            "results": results,
            "llm_raw": llm_raw_safe
        }
        self.memory["decisions"].append(log)

        # Optionally save logs to disk
        if save_logs:
            try:
                self._save_logs(log)
            except Exception as e:
                print("Warning: failed to save logs:", e)

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
