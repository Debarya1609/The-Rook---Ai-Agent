#!/usr/bin/env python3
# Rook ai.py
"""
Rook ai.py — Interactive demo / entrypoint for The Rook AI Agent

Usage:
  python "Rook ai.py"
"""
import sys
import os
import json
import pathlib
from pathlib import Path
from datetime import datetime, timezone
import textwrap

# Ensure repo root on sys.path so imports work when running the file directly
REPO_ROOT = pathlib.Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Imports from your repo
try:
    from run_demo import discover_scenarios, run_demo_from_path, build_compact_scenario, ensure_dir
except Exception as e:
    print("Import error importing run_demo. Please check file locations.")
    raise

# Tools (mocks)
try:
    from rook_orchestrator.tools.email_api import EmailAPI
except Exception as e:
    EmailAPI = None
    print("Warning: rook_orchestrator.tools.email_api not importable. Email features will be disabled.", e)

try:
    from rook_orchestrator.tools.task_api import TaskAPI
except Exception as e:
    TaskAPI = None
    print("Warning: rook_orchestrator.tools.task_api not importable. Task features will be disabled.", e)

# helper dirs
LOG_DIR = REPO_ROOT / "logs"
ensure_dir(LOG_DIR)
EMAIL_LOG_DIR = LOG_DIR / "emails"
ensure_dir(EMAIL_LOG_DIR)
TASK_LOG_DIR = LOG_DIR / "tasks"
ensure_dir(TASK_LOG_DIR)
DECISIONS_DIR = LOG_DIR / "decisions"
ensure_dir(DECISIONS_DIR)
LLM_SAMPLES_DIR = LOG_DIR / "llm_samples"
ensure_dir(LLM_SAMPLES_DIR)

# load token budgets file if present
TOKEN_BUDGETS_PATH = REPO_ROOT / "token_budgets.json"
token_budgets = {}
if TOKEN_BUDGETS_PATH.exists():
    try:
        with open(TOKEN_BUDGETS_PATH, "r", encoding="utf-8") as bf:
            raw = json.load(bf)
            for fname, info in raw.items():
                stem = Path(fname).stem
                token_budgets[stem] = info.get("recommended")
    except Exception as e:
        print("Warning: failed to read token_budgets.json:", e)

# default per-scenario map (fallback)
per_scenario_tokens = {
    "low_budget": 1400,
    "sudden_drop_in_ROAS": 1200,
    "campaign_spike": 800,
    "bad_creatives": 700,
    "content_calendar": 500,
    "dev_overload": 500
}

def get_token_budget_for_scenario(scenario_name: str) -> int:
    if scenario_name in token_budgets and token_budgets[scenario_name]:
        return int(token_budgets[scenario_name])
    return per_scenario_tokens.get(scenario_name, int(os.getenv("LLM_MAX_TOKENS", 400)))

def pretty_json(obj):
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)

def list_scenarios():
    mapping = discover_scenarios()
    if not mapping:
        print("No scenarios found in demo_inputs/. Add JSON files there.")
        return {}
    print("Available scenarios:")
    for i, k in enumerate(sorted(mapping.keys()), 1):
        print(f"  {i}) {k} -> {mapping[k]}")
    return mapping

def run_scenario_interactive():
    mapping = discover_scenarios()
    if not mapping:
        print("No scenarios available.")
        return
    list_scenarios()
    choice = input("Enter scenario name or number: ").strip()
    if not choice:
        return
    if choice.isdigit():
        idx = int(choice) - 1
        keys = sorted(mapping.keys())
        if idx < 0 or idx >= len(keys):
            print("Invalid choice.")
            return
        scenario_name = keys[idx]
        scenario_path = mapping[scenario_name]
    else:
        if choice in mapping:
            scenario_name = choice
            scenario_path = mapping[choice]
        else:
            print("Unknown scenario name.")
            return

    token_budget = get_token_budget_for_scenario(Path(scenario_path).stem)
    print(f"Running scenario '{scenario_name}' with token budget = {token_budget}")
    run_demo_from_path(scenario_path, use_llm=True, save_logs=True)
    print(f"Logs saved in {LLM_SAMPLES_DIR} and {DECISIONS_DIR}")

def generate_email_interactive():
    if EmailAPI is None:
        print("EmailAPI not available.")
        return

    # 1) choose scenario (no free-form topic)
    mapping = discover_scenarios()
    if not mapping:
        print("No scenarios found.")
        return
    print("Choose one scenario to generate an email for:")
    keys = sorted(mapping.keys())
    for i, k in enumerate(keys, 1):
        print(f"  {i}) {k}")
    choice = input("Enter scenario name or number (default 1): ").strip() or "1"
    if choice.isdigit():
        idx = int(choice) - 1
        if idx < 0 or idx >= len(keys):
            print("Invalid number, using first scenario.")
            idx = 0
        scenario_name = keys[idx]
    else:
        if choice in mapping:
            scenario_name = choice
        else:
            print("Unknown scenario, using first scenario.")
            scenario_name = keys[0]

    scenario_path = mapping[scenario_name]
    # Build context text from the scenario: use inputs.notes if present, else fallback to compact builder
    try:
        with open(scenario_path, "r", encoding="utf-8") as sf:
            scenario_json = json.load(sf)
        notes = scenario_json.get("inputs", {}).get("notes") or scenario_json.get("notes") or ""
        if notes:
            context_text = notes
        else:
            try:
                if callable(build_compact_scenario):
                    try:
                        context_text = build_compact_scenario(scenario_json)
                    except Exception:
                        context_text = build_compact_scenario(str(scenario_path))
                else:
                    context_text = json.dumps(scenario_json.get("inputs", {}))[:800]
            except Exception:
                context_text = json.dumps(scenario_json.get("inputs", {}))[:800]
    except Exception as e:
        print("Failed to read scenario file, using generic context:", e)
        context_text = "Client reduced budget mid-month and wants recommendations to maximize conversions."

    # 2) Choose recipient and subject hint (recipient will be embedded into notes for the email API)
    to = input("Recipient email (default client@example.com): ").strip() or "client@example.com"
    subject_hint = input(f"Subject hint (optional, default '{scenario_name} update'): ").strip() or f"{scenario_name} update"

    # embed recipient into notes so the LLM can pick it up (email_api expects notes)
    context_text_with_to = f"TO: {to}\n\n{context_text}"

    # 3) Get token budget for this scenario and derive worker/merge tokens
    budget = get_token_budget_for_scenario(Path(scenario_path).stem)
    worker_tokens = max(200, int(budget * 0.20))
    merge_tokens = max(300, int(budget * 0.30))
    n_workers = 3  # fixed, robust default for demo

    print(f"Using scenario '{scenario_name}' (budget={budget}). Running {n_workers} parallel drafts.")
    print(f"Worker tokens: {worker_tokens}, Merge tokens: {merge_tokens}")

    api = EmailAPI()
    # NOTE: updated call signature to match current email_api.py
    final = api.generate_email_interactive(
        subject_hint=subject_hint,
        notes=context_text_with_to,
        workers=n_workers,
        worker_tokens=worker_tokens,
        merge_tokens=merge_tokens
    )
    print("\nFINAL MERGED EMAIL:")
    print(pretty_json(final))
    print(f"Saved: {final.get('meta', {}).get('log_path') or final.get('saved_path') or 'logs/emails/'}")

def generate_tasks_interactive():
    if TaskAPI is None:
        print("TaskAPI not available.")
        return
    print("Enter a short prompt for task generation (one line):")
    prompt_text = input().strip() or "Investigate campaign performance for leadgen_oct; CPA high."
    try:
        token_budget = int(input("Max output tokens for task generation (default 400): ").strip() or "400")
    except Exception:
        token_budget = 400

    from rook_orchestrator.utils.llm_client import call_llm_structured
    sys_inst = (
        "Return EXACTLY one JSON with keys 'actions' (array) and 'summary' (string). "
        "Each action must have action_type, details, reason, confidence. Limit to 4 actions. No extra text."
    )
    print("Generating tasks via LLM (this will make one live LLM call)...")
    parsed, raw = call_llm_structured(
        scenario_text=prompt_text,
        system_instruction=sys_inst,
        max_output_tokens=token_budget,
        temperature=float(os.getenv("LLM_TEMP", 0.0)),
        repair_max_tokens=int(os.getenv("LLM_REPAIR_TOKENS", 200))
    )
    actions = parsed.get("actions", []) if isinstance(parsed, dict) else []
    print("LLM returned actions:")
    print(pretty_json(parsed))
    task_client = TaskAPI()
    created = []
    for a in actions:
        details = a.get("details", {}) if isinstance(a, dict) else {}
        try:
            res = task_client.create_task(details)
        except Exception as e:
            res = {"ok": False, "error": str(e)}
        created.append({"action": a, "created": res})
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_fn = TASK_LOG_DIR / f"sim_tasks_{ts}.json"
    with open(out_fn, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt_text, "parsed": parsed, "created": created}, f, indent=2, ensure_ascii=False)
    print("Simulated tasks saved to:", out_fn)
    print(pretty_json(created))

def show_logs_location():
    print("Logs directory:", LOG_DIR)
    print("LLM samples:", LLM_SAMPLES_DIR)
    print("Decisions:", DECISIONS_DIR)
    print("Emails:", EMAIL_LOG_DIR)
    print("Tasks:", TASK_LOG_DIR)
    print("Token budgets (if present):", TOKEN_BUDGETS_PATH if TOKEN_BUDGETS_PATH.exists() else "not found")

def run_all_scenarios_headless():
    mapping = discover_scenarios()
    if not mapping:
        print("No scenarios to run.")
        return
    print("Running all scenarios (headless) with saved token budgets where available...")
    for name, path in sorted(mapping.items()):
        stem = Path(path).stem
        budget = get_token_budget_for_scenario(stem)
        print(f"\n== Running {stem} (token budget {budget}) ==")
        try:
            run_demo_from_path(path, use_llm=True, save_logs=True)
        except Exception as e:
            print("Error running scenario", stem, ":", e)
    print("\nAll scenarios attempted. Check logs/ for outputs.")

def print_banner():
    print("="*72)
    print("The Rook — Interactive Demo".center(72))
    print("="*72)
    print("Tips: ensure MULTI_GEMINI_KEYS environment variable is set with your keys.")
    print("Use token_budgets.json to tune per-scenario budgets automatically.")
    print("Logs are saved to logs/ directory for inspection.")
    print("="*72)

def main_menu():
    print_banner()
    menu = textwrap.dedent("""
    Menu:
      1) List demo scenarios
      2) Run a demo scenario (LLM)
      3) Generate an email draft (LLM, concurrent drafts + merge)
      4) Generate tasks (LLM -> simulated TaskAPI)
      5) Run all scenarios (headless)
      6) Show logs locations
      7) Exit
    """)
    while True:
        print(menu)
        choice = input("Enter choice: ").strip()
        if choice == "1":
            list_scenarios()
        elif choice == "2":
            run_scenario_interactive()
        elif choice == "3":
            generate_email_interactive()
        elif choice == "4":
            generate_tasks_interactive()
        elif choice == "5":
            run_all_scenarios_headless()
        elif choice == "6":
            show_logs_location()
        elif choice == "7" or choice.lower() in ("q", "quit", "exit"):
            print("Exiting. Goodbye.")
            break
        else:
            print("Unknown choice. Enter a number 1-7.")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
