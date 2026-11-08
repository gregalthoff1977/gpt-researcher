"""Run the upstream offline suite with module-level process isolation.

This avoids upstream cross-module mock leakage and pytest-forked's incompatibility
with newer pytest versions. It excludes the same three live modules as upstream CI.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"tests/test_researcher_logging.py", "tests/test_logging_output.py", "tests/test_mcp.py"}


def main():
    directory = ROOT / "verification-results"
    directory.mkdir(exist_ok=True)
    paths = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").rglob("test_*.py")
                   if str(p.relative_to(ROOT)) not in EXCLUDED)
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    env = dict(os.environ, GPTR_BLOCK_NETWORK="1", OPENAI_API_KEY="offline-fixture-key",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    def one(path):
        name = path.replace("/", "_")
        xml = directory / (name + ".xml")
        try:
            result = subprocess.run([sys.executable, "-m", "pytest", path, "-q", "-o", "addopts=",
                "-p", "pytest_asyncio.plugin", "--tb=short", "--junitxml=" + str(xml)],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
            (directory / (name + ".log")).write_text(result.stdout + result.stderr)
            suite = ET.parse(xml).getroot() if xml.exists() else None
            cases = list(suite.iter("testcase")) if suite is not None else []
            skipped_only = result.returncode == 5 and all(c.find("skipped") is not None for c in cases) and suite is not None and any(
                int(s.get("skipped", "0")) > 0 for s in suite.iter("testsuite"))
            return {"module": path, "exit_code": 0 if skipped_only else result.returncode, "tests": len(cases),
                    "failed": sum(c.find("failure") is not None or c.find("error") is not None for c in cases),
                    "skipped": sum(c.find("skipped") is not None for c in cases)}
        except subprocess.TimeoutExpired as exc:
            (directory / (name + ".log")).write_text("Module timed out after 90 seconds")
            return {"module": path, "exit_code": 124, "tests": 0, "failed": 1, "skipped": 0}
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(one, paths), 1):
            results.append(result)
            if result["exit_code"]:
                print("FAIL " + result["module"], flush=True)
            elif i % 20 == 0:
                print(f"Verified {i}/{len(paths)} modules", flush=True)
    summary = {"modules": results, "live_modules_excluded": sorted(EXCLUDED),
        "tests": sum(r["tests"] for r in results), "failed": sum(r["failed"] for r in results),
        "skipped": sum(r["skipped"] for r in results),
        "failed_modules": [r["module"] for r in results if r["exit_code"]]}
    (directory / "offline-suite.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "modules"}), flush=True)
    return bool(summary["failed_modules"])


if __name__ == "__main__":
    sys.exit(main())
