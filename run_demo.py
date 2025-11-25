import json
import sys
import argparse
import os
from datetime import datetime
from rook_orchestrator.orchestrator import RookOrchestrator
from datetime import timezone

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def run_demo(file_path, use_llm=False, save_logs=False):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    orchestrator = RookOrchestrator(data.get("analytics"))
    result = orchestrator.run_cycle(data.get("inputs", {}), use_llm=use_llm)

    # Print to stdout
    print(json.dumps(result, indent=2))

    # Optionally save logs (llm_raw + full decision)
    if save_logs:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        ensure_dir("logs/llm_samples")
        ensure_dir("logs/decisions")
        # llm_raw may contain non-serializable fields but orchestrator sanitizes it
        try:
            # Save llm_raw only
            with open(f"logs/llm_samples/llm_{timestamp}.json", "w", encoding="utf-8") as lf:
                json.dump(result.get("llm_raw", {}), lf, indent=2)
            # Save full decision trace
            with open(f"logs/decisions/decision_{timestamp}.json", "w", encoding="utf-8") as df:
                json.dump(result, df, indent=2)
            print(f"Saved logs to logs/llm_samples/llm_{timestamp}.json and logs/decisions/decision_{timestamp}.json")
        except Exception as e:
            print("Warning: failed to save logs:", e)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", help="scenario name (campaign_spike, dev_overload, content_calendar)")
    parser.add_argument("--use-llm", action="store_true", help="Call live Gemini LLM (if configured).")
    parser.add_argument("--save-logs", action="store_true", help="Save llm_raw and full decision to logs/")
    args = parser.parse_args()

    file_map = {
        "campaign_spike": "demo_inputs/campaign_spike.json",
        "dev_overload": "demo_inputs/dev_overload.json",
        "content_calendar": "demo_inputs/content_calendar.json"
    }

    scenario = args.scenario
    if scenario not in file_map:
        print("Unknown scenario:", scenario)
        sys.exit(1)

    run_demo(file_map[scenario], use_llm=args.use_llm, save_logs=args.save_logs)
