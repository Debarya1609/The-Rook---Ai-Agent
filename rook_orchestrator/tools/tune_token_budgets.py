#!/usr/bin/env python3
"""
tools/tune_token_budgets.py

Probe Gemini for token usage on given scenario(s) and compute recommended max_output_tokens.

Usage:
  python tools/tune_token_budgets.py demo_inputs/low_budget.json
  python tools/tune_token_budgets.py --all
Outputs: prints recommendations and writes token_budgets.json
"""
# ensure project root is on sys.path so package imports work when running script directly
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[2]  # two levels up -> project root
sys.path.insert(0, str(repo_root))

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from rook_orchestrator.utils.llm_client import call_llm, get_text_from_sdk_resp

# Copy of your compact scenario builder (keeps it consistent)
def build_compact_scenario_from_file(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    notes = data.get("inputs", {}).get("notes") or data.get("notes") or ""
    if notes:
        parts.append(f"notes:{notes}")
    campaigns = (data.get("analytics") or {}).get("campaigns", [])
    for c in campaigns:
        cid = c.get("campaign_id") or c.get("campaignId") or c.get("id") or "campaign"
        channel = c.get("channel") or c.get("platform") or "unknown"
        spend = c.get("daily_spend") or c.get("spend") or ""
        cpa = c.get("cpa") or ""
        target = c.get("target_cpa") or ""
        trend = c.get("trend") or ""
        parts.append(f"{cid}|{channel}|spend:{spend}|cpa:{cpa}|target:{target}|trend:{trend}")
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

def extract_usage_info(raw_resp) -> dict:
    """
    Try to extract token usage info from raw_resp (SDK object or dict).
    Return dict with keys: prompt_tokens, thoughts_tokens, total_tokens
    """
    info = {"prompt_tokens": None, "thoughts_tokens": None, "total_tokens": None}
    if not raw_resp:
        return info

    # Sometimes raw_resp is an SDK object with usage metadata
    try:
        # try attribute access
        usage = getattr(raw_resp, "usage_metadata", None) or getattr(raw_resp, "usage", None) or getattr(raw_resp, "usage_metadata", None)
        if usage:
            # try common fields
            pt = getattr(usage, "prompt_token_count", None)
            tt = getattr(usage, "thoughts_token_count", None) or getattr(usage, "completion_token_count", None)
            tot = getattr(usage, "total_token_count", None)
            if pt: info["prompt_tokens"] = int(pt)
            if tt: info["thoughts_tokens"] = int(tt)
            if tot: info["total_tokens"] = int(tot)
            return info
    except Exception:
        pass

    # If it's a dict (HTTP) try common keys
    try:
        if isinstance(raw_resp, dict):
            # google generative API may nest usage info differently
            if "usage_metadata" in raw_resp:
                um = raw_resp["usage_metadata"]
                info["prompt_tokens"] = int(um.get("prompt_token_count") or um.get("prompt_tokens") or 0)
                info["thoughts_tokens"] = int(um.get("thoughts_token_count") or um.get("completion_tokens") or 0)
                info["total_tokens"] = int(um.get("total_token_count") or um.get("total_tokens") or 0)
                return info
            # fallback: try top-level known fields
            info["total_tokens"] = int(raw_resp.get("total_token_count") or raw_resp.get("total_tokens") or 0)
            return info
    except Exception:
        pass

    # fallback: no usage found
    return info

def compute_recommendation(usage: dict) -> int:
    prompt = usage.get("prompt_tokens") or 0
    thoughts = usage.get("thoughts_tokens") or 0
    total = usage.get("total_tokens") or 0

    if prompt and thoughts:
        base = prompt + thoughts
    elif total:
        base = total
    else:
        # fallback conservative default
        base = max(400, prompt + 300)

    buffer = max(200, int(0.2 * base))
    rec = int(base + buffer)
    # clamp
    rec = min(rec, 2000)
    return rec

def probe_scenario(path: Path, system_instruction: str = None, probe_tokens: int = 2000) -> dict:
    scenario_text = build_compact_scenario_from_file(path)
    # unwrap system instruction if present
    system_instruction = system_instruction or ""
    # build a short probe prompt: use system instruction + scenario
    prompt = (system_instruction + "\n\nSCENARIO:\n" + scenario_text) if system_instruction else scenario_text
    # call with a large max_output_tokens so it won't truncate
    resp = call_llm(prompt, model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), max_output_tokens=probe_tokens, temperature=0.0)
    raw_resp = resp.get("raw") or resp.get("text") or None
    textual = get_text_from_sdk_resp(raw_resp) if raw_resp is not None else resp.get("text", "")
    usage = extract_usage_info(raw_resp) or {}
    usage["text_snippet"] = (textual or "")[:800]
    usage["raw_meta"] = resp.get("meta", {})
    usage["recommended"] = compute_recommendation(usage)
    return usage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="scenario files (or --all)")
    parser.add_argument("--all", action="store_true", help="Probe all demo_inputs/*.json")
    parser.add_argument("--system", default="strategy_prompt.txt", help="path to system prompt")
    args = parser.parse_args()

    repo_root = Path.cwd()
    demo_dir = repo_root / "demo_inputs"
    scenario_files = []

    if args.all:
        scenario_files = sorted(demo_dir.glob("*.json"))
    else:
        if not args.paths:
            print("No scenarios specified. Use --all or provide file paths.")
            return
        for p in args.paths:
            scenario_files.append(Path(p))

    # read system instruction
    system_instruction = ""
    try:
        si_path = Path(args.system)
        if si_path.exists():
            system_instruction = si_path.read_text(encoding="utf-8")
    except Exception:
        system_instruction = ""

    results = {}
    for sf in scenario_files:
        print(f"\nProbing scenario: {sf}")
        usage = probe_scenario(sf, system_instruction=system_instruction, probe_tokens=2000)
        print("  prompt_tokens:", usage.get("prompt_tokens"))
        print("  thoughts_tokens:", usage.get("thoughts_tokens"))
        print("  total_tokens:", usage.get("total_tokens"))
        print("  recommended_max_output_tokens:", usage.get("recommended"))
        print("  raw_meta:", usage.get("raw_meta"))
        results[sf.name] = usage

    # save recommendations
    out_path = repo_root / "token_budgets.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("\nWrote recommendations to", out_path)
    except Exception as e:
        print("Failed to write token_budgets.json:", e)

if __name__ == "__main__":
    main()
