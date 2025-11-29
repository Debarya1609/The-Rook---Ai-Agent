#!/usr/bin/env python3
"""
submit_ready.py

One-click submission runner:
- Validates API keys (uses test_keys logic)
- Runs all demo scenarios in demo_inputs/ with live LLM (if available)
- Saves logs (llm_samples + decisions)
- Writes submission_summary.json and a zip archive submission_package.zip
"""

import os
import json
import time
import zipfile
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import sys

ROOT = Path(__file__).parent.resolve()
DEMOS_DIR = ROOT / "demo_inputs"
LOGS_DIR = ROOT / "logs"
LLM_SAMPLES = LOGS_DIR / "llm_samples"
DECISIONS = LOGS_DIR / "decisions"

# Ensure folders exist
for p in (LOGS_DIR, LLM_SAMPLES, DECISIONS):
    p.mkdir(parents=True, exist_ok=True)

def run_command(cmd_args, capture_output=False):
    """Run a command in-shell; return (returncode, stdout)."""
    try:
        if capture_output:
            out = subprocess.check_output(cmd_args, stderr=subprocess.STDOUT, shell=False, text=True)
            return 0, out
        else:
            rc = subprocess.call(cmd_args, shell=False)
            return rc, None
    except subprocess.CalledProcessError as e:
        return e.returncode, getattr(e, "output", str(e))

def run_test_keys():
    print("1) Running key health check (test_keys.py)...")
    rc, out = run_command([sys.executable, "test_keys.py"], capture_output=True)
    if rc != 0:
        print("test_keys.py failed to run. Return code:", rc)
        print(out or "")
        return False, out
    # Inspect output for at least one OK
    ok = "OK (gemini)" in out
    print(out)
    return ok, out

def discover_demos():
    demos = []
    if not DEMOS_DIR.exists():
        print("No demo_inputs/ directory found.")
        return demos
    for f in sorted(DEMOS_DIR.glob("*.json")):
        demos.append(f)
    return demos

def run_demo_file(path: Path, use_llm=True, save_logs=True):
    print(f"\nRunning demo: {path.name}")
    cmd = [sys.executable, "run_demo.py", str(path), "--use-llm"] if use_llm else [sys.executable, "run_demo.py", str(path)]
    if save_logs:
        cmd.append("--save-logs")
    rc, out = run_command(cmd, capture_output=True)
    return rc, out

def collect_latest_logs():
    # pick the most recent llm_sample and decision files
    samples = sorted(LLM_SAMPLES.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    decisions = sorted(DECISIONS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return samples, decisions

def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = {"run_at": ts, "demo_results": []}

    keys_ok, keys_output = run_test_keys()
    if not keys_ok:
        print("WARNING: Key health check did not find any live Gemini key. The run will still proceed and may use stub fallback.")
    else:
        print("At least one key appears healthy — proceeding with live demos.")

    demos = discover_demos()
    if not demos:
        print("No demo files found in demo_inputs/. Create JSON scenarios and re-run.")
        return

    for demo_path in demos:
        rc, out = run_demo_file(demo_path, use_llm=True, save_logs=True)
        demo_summary = {"scenario": demo_path.stem, "rc": rc, "raw_output": None, "llm_meta": None}
        if out:
            demo_summary["raw_output"] = out.strip()[:2000]
        # try to find the most recent sample file created by this run (best-effort)
        time.sleep(0.2)
        samples, decisions = collect_latest_logs()
        # attach most recent sample & decision file to summary if present
        if samples:
            demo_summary["latest_sample"] = str(samples[0].name)
            try:
                with open(samples[0], "r", encoding="utf-8") as f:
                    sample_json = json.load(f)
                    demo_summary["llm_meta"] = sample_json.get("meta", {})
            except Exception:
                demo_summary["llm_meta"] = None
        if decisions:
            demo_summary["latest_decision"] = str(decisions[0].name)
        summary["demo_results"].append(demo_summary)

    # Write submission_summary.json
    summary_path = LOGS_DIR / f"submission_summary_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Create a zip of logs + README snapshot
    zip_path = ROOT / f"submission_package_{ts}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # include recent logs (limit to last 20 files)
        all_samples = sorted(LLM_SAMPLES.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]
        all_decisions = sorted(DECISIONS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]
        for p in all_samples + all_decisions:
            zf.write(p, arcname=f"logs/{p.name}")
        # include README if exists
        readme = ROOT / "README.md"
        if readme.exists():
            zf.write(readme, arcname="README.md")
        # include the summary
        zf.write(summary_path, arcname=summary_path.name)

    print("\n=== Submission package created ===")
    print("Summary JSON:", summary_path)
    print("Package ZIP :", zip_path)
    print("\nPlease inspect logs/llm_samples and logs/decisions to include representative examples in your submission.")

if __name__ == "__main__":
    main()
