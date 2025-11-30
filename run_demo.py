import json
import sys
import argparse
import os
import glob
import time
from datetime import datetime, timezone
from pathlib import Path
from rook_orchestrator.orchestrator import RookOrchestrator

# import the structured call wrapper and helper
from rook_orchestrator.utils.llm_client import call_llm_structured, get_text_from_sdk_resp
from rook_orchestrator.utils.key_loader import mask_key, load_keys_from_env

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

def build_compact_scenario(data: dict) -> str:
    """
    Build a short, compact scenario string from the demo JSON to conserve tokens.
    """
    parts = []
    # add notes if present
    notes = data.get("inputs", {}).get("notes") or data.get("notes") or ""
    if notes:
        parts.append(f"notes:{notes}")
    # add analytics campaigns succinctly
    campaigns = (data.get("analytics") or {}).get("campaigns", [])
    for c in campaigns:
        cid = c.get("campaign_id") or c.get("campaignId") or c.get("id") or "campaign"
        channel = c.get("channel") or c.get("platform") or "unknown"
        spend = c.get("daily_spend") or c.get("spend") or ""
        cpa = c.get("cpa") or ""
        target = c.get("target_cpa") or ""
        trend = c.get("trend") or ""
        parts.append(f"{cid}|{channel}|spend:{spend}|cpa:{cpa}|target:{target}|trend:{trend}")
    # add insights if present (short)
    insights = data.get("insights", {})
    if insights:
        try:
            if isinstance(insights, dict):
                for k, v in insights.items():
                    parts.append(f"insight_{k}:{str(v)[:120]}")
            else:
                parts.append(str(insights)[:300])
        except Exception:
            pass
    return "\n".join(parts)

def sanitize_llm_raw_in_result(result: dict) -> None:
    """
    Ensure result['llm_raw'] does not contain non-serializable SDK objects.
    - Extract human-readable text into 'raw_str' using get_text_from_sdk_resp.
    - Set 'raw' to None to avoid json.dumps errors.
    """
    try:
        if "llm_raw" in result:
            lr = result["llm_raw"]
            raw_obj = lr.get("raw")
            if raw_obj is not None:
                try:
                    lr["raw_str"] = get_text_from_sdk_resp(raw_obj)
                except Exception:
                    lr["raw_str"] = str(raw_obj)
                # remove heavy SDK object
                lr["raw"] = None
    except Exception:
        # fail-safe: don't raise from sanitization
        pass

def run_demo_from_path(file_path, use_llm=False, save_logs=False):
    """
    Run the orchestrator given a full path to a scenario json.
    If use_llm is True, call call_llm_structured() to get parsed plan (compact JSON).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    orchestrator = RookOrchestrator(data.get("analytics"))
    result = orchestrator.run_cycle(data.get("inputs", {}), use_llm=use_llm)

    # If LLM should be used, build compact scenario and call structured LLM wrapper
    if use_llm:
        try:
            scenario_text = build_compact_scenario(data)
            # read strategy prompt if exists
            strategy_prompt_path = Path("rook_orchestrator/strategy_prompt.txt")
            system_instruction = None
            # also try repo root copy
            if not strategy_prompt_path.exists():
                strategy_prompt_path = Path("strategy_prompt.txt")
            if strategy_prompt_path.exists():
                try:
                    system_instruction = strategy_prompt_path.read_text(encoding="utf-8")
                except Exception:
                    system_instruction = None

            # choose per-scenario token budget (adjust as needed)
            per_scenario_tokens = {
                "low_budget": 1400,
                "sudden_drop_in_ROAS": 600,
                "campaign_spike": 500,
                "bad_creatives": 450,
                "content_calendar": 300,
                "dev_overload": 250
            }
            scenario_name = Path(file_path).stem
            token_budget = per_scenario_tokens.get(scenario_name, int(os.getenv("LLM_MAX_TOKENS", 400)))

            parsed_plan, raw_response = call_llm_structured(
                scenario_text=scenario_text,
                system_instruction=system_instruction,
                max_output_tokens=token_budget,
                temperature=float(os.getenv("LLM_TEMP", 0.0)),
                repair_max_tokens=int(os.getenv("LLM_REPAIR_TOKENS", 200))
            )

            # attach to result for logging & decision simulation
            result["llm_raw"] = raw_response
            result["llm_parsed"] = parsed_plan

        except Exception as e:
            # keep original result but annotate failure
            result.setdefault("llm_raw", {})
            result["llm_raw"]["error"] = str(e)

    # Sanitize non-serializable objects before printing
    sanitize_llm_raw_in_result(result)

    # Print to stdout (previous behaviour)
    print(json.dumps(result, indent=2))

    # Optionally save logs (llm_raw + full decision)
    if save_logs:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        ensure_dir("logs/llm_samples")
        ensure_dir("logs/decisions")
        try:
            # Prepare sample object for saving
            sample = result.get("llm_raw", {}) or {}
            # include parsed if present
            if "llm_parsed" in result:
                sample["parsed"] = result["llm_parsed"]
            # Ensure sample.raw is serializable (defensive)
            raw_obj = sample.get("raw")
            if raw_obj is not None:
                try:
                    sample["raw_str"] = get_text_from_sdk_resp(raw_obj)
                except Exception:
                    sample["raw_str"] = str(raw_obj)
                sample["raw"] = None

            with open(f"logs/llm_samples/llm_{timestamp}.json", "w", encoding="utf-8") as lf:
                json.dump(sample, lf, indent=2)

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
