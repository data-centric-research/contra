#!/usr/bin/env python3
"""Run one paper smoke-test job when system load is acceptable."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_smoke_lib import (
    COOLDOWN_SEC,
    MANIFEST,
    load_ok,
    load_state,
    pick_next,
    progress_counts,
    run_job,
    save_state,
    system_metrics,
    utc_now,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=str(REPO / ".venv" / "bin" / "python"))
    parser.add_argument("--force", action="store_true", help="ignore load guard")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    state = load_state()

    if args.status:
        done, total = progress_counts(manifest, state)
        print(json.dumps({"done": done, "total": total, "metrics": system_metrics()}, indent=2))
        return 0

    metrics = system_metrics()
    ok, reason = (True, "forced") if args.force else load_ok(metrics)
    if not ok:
        print(
            json.dumps(
                {
                    "action": "wait",
                    "reason": reason,
                    "metrics": metrics,
                    "next_sleep_sec": COOLDOWN_SEC,
                },
                ensure_ascii=False,
            )
        )
        return 2

    job = pick_next(manifest, state)
    if job is None:
        print(json.dumps({"action": "complete", "metrics": metrics}, ensure_ascii=False))
        return 0

    started = time.time()
    code, summary = run_job(job, manifest["defaults"], args.python)
    elapsed = int(time.time() - started)

    entry = {"at": utc_now(), "elapsed_sec": elapsed, "exit_code": code, "summary_tail": summary}
    if code == 0:
        state.setdefault("completed", {})[job["id"]] = entry
    else:
        state.setdefault("failed", {})[job["id"]] = entry
    save_state(state)

    done, total = progress_counts(manifest, state)

    sleep_sec = COOLDOWN_SEC
    if job["kind"] == "contra":
        sleep_sec = max(COOLDOWN_SEC, min(900, elapsed // 2))

    print(
        json.dumps(
            {
                "action": "ran",
                "job_id": job["id"],
                "kind": job["kind"],
                "exit_code": code,
                "elapsed_sec": elapsed,
                "progress": f"{done}/{total}",
                "next_sleep_sec": sleep_sec,
                "metrics": metrics,
            },
            ensure_ascii=False,
        )
    )
    return 0 if code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
