#!/usr/bin/env python3
"""Run multiple paper smoke-test jobs in parallel with load guards."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_smoke_lib import (
    COOLDOWN_SEC,
    MANIFEST,
    active_running,
    launch_job,
    load_ok,
    load_state,
    locked_state,
    pick_parallel_batch,
    progress_counts,
    suggest_max_workers,
    system_metrics,
    utc_now,
)

REPO = Path(__file__).resolve().parents[1]


def wait_for_slot(worker_count: int, force: bool) -> tuple[bool, str, dict]:
    metrics = system_metrics()
    if force:
        return True, "forced", metrics
    ok, reason = load_ok(metrics, extra_workers=max(0, worker_count - 1))
    return ok, reason, metrics


def finish_job(job_id: str, proc, log_path: Path, started: float) -> dict:
    code = proc.wait()
    elapsed = int(time.time() - started)
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
    except OSError:
        tail = []
    return {
        "job_id": job_id,
        "exit_code": code,
        "elapsed_sec": elapsed,
        "summary_tail": "\n".join(tail),
    }


def wait_batch(launched: list[tuple[dict, object, Path, float]]) -> list[dict]:
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(launched)) as pool:
        future_map = {
            pool.submit(finish_job, job["id"], proc, log_path, started): job
            for job, proc, log_path, started in launched
        }
        for future in as_completed(future_map):
            job = future_map[future]
            result = future.result()
            result["kind"] = job["kind"]
            results.append(result)
    return results


def apply_results(state: dict, results: list[dict]) -> None:
    for result in results:
        jid = result["job_id"]
        state.get("running", {}).pop(jid, None)
        entry = {
            "at": utc_now(),
            "elapsed_sec": result["elapsed_sec"],
            "exit_code": result["exit_code"],
            "summary_tail": result["summary_tail"],
        }
        if result["exit_code"] == 0:
            state.setdefault("completed", {})[jid] = entry
            state.get("failed", {}).pop(jid, None)
        else:
            state.setdefault("failed", {})[jid] = entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=str(REPO / ".venv" / "bin" / "python"))
    parser.add_argument("--max-workers", type=int, default=0, help="0 = auto from CPU/RAM")
    parser.add_argument("--force", action="store_true", help="ignore load guard")
    parser.add_argument("--dry-run", action="store_true", help="show planned batch only")
    parser.add_argument("--once", action="store_true", help="run one batch then exit")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="keep scheduling batches until complete or load blocks",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    exit_code = 0

    if args.status:
        state = load_state()
        done, total = progress_counts(manifest, state)
        running = active_running(state, manifest)
        print(
            json.dumps(
                {
                    "done": done,
                    "total": total,
                    "running": list(running.keys()),
                    "metrics": system_metrics(),
                    "suggested_workers": suggest_max_workers(system_metrics()),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    while True:
        metrics = system_metrics()
        max_workers = args.max_workers or suggest_max_workers(metrics)

        with locked_state() as state:
            running = active_running(state, manifest)
            external = [jid for jid, meta in running.items() if meta.get("external")]
            slots = max(0, max_workers - len(running))
            batch = pick_parallel_batch(manifest, state, slots) if slots > 0 else []
            done, total = progress_counts(manifest, state)

            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "action": "dry_run",
                            "max_workers": max_workers,
                            "running": list(running.keys()),
                            "external": external,
                            "planned": [j["id"] for j in batch],
                            "progress": f"{done}/{total}",
                            "metrics": metrics,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0

            if done >= total:
                print(json.dumps({"action": "complete", "progress": f"{done}/{total}"}, ensure_ascii=False))
                return 0

            if not batch:
                ok, reason, _ = wait_for_slot(max_workers, args.force)
                payload = {
                    "action": "wait" if not ok else "idle",
                    "reason": reason if not ok else "no runnable jobs while slots available",
                    "running": list(running.keys()),
                    "metrics": metrics,
                }
                if not ok:
                    payload["next_sleep_sec"] = COOLDOWN_SEC
                print(json.dumps(payload, ensure_ascii=False))
                if args.once or not args.loop:
                    return 2 if not ok else 0
                time.sleep(COOLDOWN_SEC if not ok else COOLDOWN_SEC // 2)
                continue

            ok, reason, metrics = wait_for_slot(len(batch) + len(running), args.force)
            if not ok:
                print(
                    json.dumps(
                        {
                            "action": "wait",
                            "reason": reason,
                            "would_run": [j["id"] for j in batch],
                            "running": list(running.keys()),
                            "next_sleep_sec": COOLDOWN_SEC,
                            "metrics": metrics,
                        },
                        ensure_ascii=False,
                    )
                )
                if args.once or not args.loop:
                    return 2
                time.sleep(COOLDOWN_SEC)
                continue

            launched = []
            started_at = time.time()
            for job in batch:
                proc, log_path = launch_job(job, manifest["defaults"], args.python)
                state.setdefault("running", {})[job["id"]] = {
                    "pid": proc.pid,
                    "started_at": utc_now(),
                    "launcher": "parallel",
                }
                launched.append((job, proc, log_path, started_at))
            planned = [j["id"] for j in batch]

        results = wait_batch(launched)

        with locked_state() as state:
            apply_results(state, results)
            done, total = progress_counts(manifest, state)

        for result in results:
            if result["exit_code"] != 0:
                exit_code = 1

        print(
            json.dumps(
                {
                    "action": "ran_batch",
                    "jobs": planned,
                    "results": [
                        {
                            "job_id": r["job_id"],
                            "exit_code": r["exit_code"],
                            "elapsed_sec": r["elapsed_sec"],
                        }
                        for r in results
                    ],
                    "progress": f"{done}/{total}",
                    "metrics": system_metrics(),
                },
                ensure_ascii=False,
            )
        )

        if args.once or not args.loop:
            return exit_code

        if done >= total:
            print(json.dumps({"action": "complete", "progress": f"{done}/{total}"}, ensure_ascii=False))
            return exit_code

        time.sleep(COOLDOWN_SEC)


if __name__ == "__main__":
    sys.exit(main())
