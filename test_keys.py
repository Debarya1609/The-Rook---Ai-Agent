# test_keys.py (project root)
import os
import json
import time
from rook_orchestrator.utils.key_loader import load_keys_from_env, mask_key
from rook_orchestrator.utils import llm_client

def quick_test_key(single_key: str, model: str):
    """
    Temporarily set MULTI_GEMINI_KEYS to a single key so call_llm reads it.
    Returns (ok: bool, info: dict_or_str)
    """
    # Backup original env var
    original = os.getenv("MULTI_GEMINI_KEYS", None)
    os.environ["MULTI_GEMINI_KEYS"] = single_key
    try:
        # Minimal prompt
        prompt = "Return JSON: {\"ok\": true}"
        resp = llm_client.call_llm(prompt, model=model, max_output_tokens=32, temperature=0.0)
        # If meta.source == "gemini", it's live
        src = resp.get("meta", {}).get("source")
        return (src == "gemini", resp)
    except Exception as e:
        return (False, str(e))
    finally:
        # Restore original
        if original is None:
            os.environ.pop("MULTI_GEMINI_KEYS", None)
        else:
            os.environ["MULTI_GEMINI_KEYS"] = original

def main():
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    keys = load_keys_from_env()
    results = []
    for k in keys:
        print(f"Testing key {mask_key(k)} ...", end=" ")
        ok, info = quick_test_key(k, model)
        if ok:
            print("OK (gemini)")
        else:
            print("FAILED or stub")
        results.append({"key_masked": mask_key(k), "ok": ok, "info_snippet": str(info)[:300]})
        time.sleep(0.4)
    print("\nSummary:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
