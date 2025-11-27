import json
import sys
import argparse
import os
import glob
from datetime import datetime, timezone
from rook_orchestrator.orchestrator import RookOrchestrator

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def discover_scenarios():
    """
    Return map: scenario_name -> path to demo_inputs/<scenario_name>.json
    """
    base = os.path.join(os.path.dirname(__file__), "demo_inputs")
    pattern = os.path.join(base, "*.json")
    files = glob.glob(pattern)
    mapping = {}
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        mapping[name] = p
    return mapping

def run_demo_from_path(file_path, use_llm=False, save_logs=False):
    """
    Run the orchestrator given a full path to a scenario json.
    """
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
        try:
            with open(f"logs/llm_samples/llm_{timestamp}.json", "w", encoding="utf-8") as lf:
                json.dump(result.get("llm_raw", {}), lf, indent=2)
            with open(f"logs/decisions/decision_{timestamp}.json", "w", encoding="utf-8") as df:
                json.dump(result, df, indent=2)
            print(f"Saved logs to logs/llm_samples/llm_{timestamp}.json and logs/decisions/decision_{timestamp}.json")
        except Exception as e:
            print("Warning: failed to save logs:", e)

    return result

def print_available_scenarios(mapping):
    if not mapping:
        print("No scenarios found in demo_inputs/. Add JSON files there, e.g. demo_inputs/low_budget.json")
        return
    print("Available scenarios:")
    for k in sorted(mapping.keys()):
        print("  -", k, "->", mapping[k])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a demo scenario. Pass scenario name (basename of demo_inputs/*.json) or a path to a JSON file."
    )
    parser.add_argument("scenario", help="scenario name (basename) or path to JSON file (e.g. demo_inputs/low_budget.json)")
    parser.add_argument("--use-llm", action="store_true", help="Call live Gemini LLM (if configured).")
    parser.add_argument("--save-logs", action="store_true", help="Save llm_raw and full decision to logs/")
    args = parser.parse_args()

    file_map = discover_scenarios()

    scenario_arg = args.scenario

    # 1) If user passed an explicit path to a file that exists -> use it
    if os.path.isfile(scenario_arg):
        scenario_path = scenario_arg
    else:
        # 2) if they passed a basename that matches discovered scenarios -> use mapping
        if scenario_arg in file_map:
            scenario_path = file_map[scenario_arg]
        else:
            # Not found — print helpful message and list available scenarios
            print(f"Unknown scenario: {scenario_arg}")
            print_available_scenarios(file_map)
            sys.exit(1)

    # run
    run_demo_from_path(scenario_path, use_llm=args.use_llm, save_logs=args.save_logs)
