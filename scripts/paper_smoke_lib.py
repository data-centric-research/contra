"""Shared helpers for CONTRA paper smoke-test orchestration."""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "scripts" / "paper_smoke_manifest.json"
STATE_PATH = REPO / "runs" / "paper_smoke_state.json"
STATE_LOCK_PATH = REPO / "runs" / "paper_smoke_state.lock"
LOG_DIR = REPO / "logs" / "paper_smoke"

MAX_LOAD_RATIO = 0.70
MIN_FREE_GIB = 3.0
COOLDOWN_SEC = 120
GIB_PER_WORKER = 3.0
MAX_WORKERS_CAP = 4
BASELINE_PARALLEL_PER_CHAIN = 3

TRAINING_PATTERN = r"run_contra\.py|run_experiment\.py|colearn/main\.py|run_sotta|run_plf|tta\.py|gen_cifar|gen_pet|gen_corruption"

EXCLUSIVE_KINDS = frozenset(
    {"gen", "step0", "raw", "contra", "eval", "corrupt_gen", "corrupt_eval"}
)
PARALLEL_KINDS = frozenset({"rehearsal", "lnl", "tta"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return default_state()


def default_state() -> dict:
    return {
        "completed": {},
        "failed": {},
        "skipped": {},
        "running": {},
        "precompleted": {
            "cifar10_sym_gen": "dataset already at data/cifar-10/gen/nr_0.2_nt_symmetric_balanced",
            "cifar10_sym_s0": "prior smoke (2 epochs)",
            "cifar10_sym_s1_raw": "prior smoke",
            "cifar10_sym_s1_contra": "prior smoke",
        },
    }


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


@contextmanager
def locked_state():
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = STATE_LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        if STATE_PATH.exists():
            state = json.loads(STATE_PATH.read_text())
        else:
            state = default_state()
        state.setdefault("running", {})
        yield state
        save_state(state)
    finally:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        lock_fd.close()


def system_metrics() -> dict:
    ncpu = os.cpu_count() or 4
    load1 = os.getloadavg()[0]
    load_ratio = load1 / ncpu

    vm = subprocess.check_output(["vm_stat"], text=True)
    page_size = 16384
    m = re.search(r"page size of (\d+)", vm)
    if m:
        page_size = int(m.group(1))
    free_pages = 0
    for key in ("Pages free", "Pages speculative"):
        hit = re.search(rf"{re.escape(key)}:\s+(\d+)", vm)
        if hit:
            free_pages += int(hit.group(1).replace(".", ""))
    free_gib = free_pages * page_size / (1024**3)

    training_pids = list_training_pids()
    return {
        "ncpu": ncpu,
        "load1": load1,
        "load_ratio": load_ratio,
        "free_gib": free_gib,
        "training_running": bool(training_pids),
        "training_count": len(training_pids),
    }


def list_training_pids() -> list[int]:
    proc = subprocess.run(
        ["pgrep", "-f", TRAINING_PATTERN],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    return [int(x) for x in proc.stdout.split() if x.strip()]


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def suggest_max_workers(metrics: dict, cap: int | None = None) -> int:
    cap = MAX_WORKERS_CAP if cap is None else cap
    headroom_load = max(0.0, metrics["ncpu"] * MAX_LOAD_RATIO - metrics["load1"])
    by_cpu = max(1, int(headroom_load))
    reserve = MIN_FREE_GIB
    by_mem = max(1, int((metrics["free_gib"] - reserve) / GIB_PER_WORKER))
    return max(1, min(by_cpu, by_mem, cap))


def load_ok(metrics: dict, extra_workers: int = 0) -> tuple[bool, str]:
    needed_gib = MIN_FREE_GIB + extra_workers * GIB_PER_WORKER
    if metrics["load_ratio"] > MAX_LOAD_RATIO:
        return False, f"load {metrics['load1']:.1f} > {MAX_LOAD_RATIO:.0%} of {metrics['ncpu']} cores"
    if metrics["free_gib"] < needed_gib:
        return (
            False,
            f"free memory {metrics['free_gib']:.1f} GiB < {needed_gib:.1f} GiB "
            f"(base {MIN_FREE_GIB} + {extra_workers} workers)",
        )
    return True, "ok"


def case_name(job: dict, defaults: dict) -> str:
    balanced = job.get("balanced", defaults.get("balanced", False))
    suffix = "balanced" if balanced else "cvpr"
    return f"nr_{job['noise_ratio']}_nt_{job['noise_type']}_{suffix}"


def chain_key(job: dict) -> tuple:
    return (job["dataset"], job.get("noise_type"), job.get("noise_ratio"))


def dataset_ready(job: dict, defaults: dict) -> bool:
    if job["kind"] == "gen":
        return True
    case = case_name(job, defaults)
    ds = job["dataset"]
    gen_root = REPO / "data" / ds / "gen" / case
    if job["kind"] == "step0":
        return (gen_root / "step_0" / "train_data.npy").exists()
    step = job.get("step", 0)
    return (gen_root / f"step_{step}" / "train_data.npy").exists() or (
        job["kind"] in {"corrupt_gen", "corrupt_eval"}
        and (gen_root / "test_data.npy").exists()
    )


def job_done(state: dict, job_id: str) -> bool:
    return job_id in state.get("completed", {}) or job_id in state.get("precompleted", {})


def deps_ok(job: dict, state: dict) -> bool:
    for dep in job.get("requires_steps", []):
        if not job_done(state, dep):
            return False
    return True


def find_job(manifest: dict, dataset: str, noise_type: str, noise_ratio: float, kind: str, step=None):
    for job in manifest["jobs"]:
        if job["dataset"] != dataset:
            continue
        if job.get("noise_type") != noise_type or job.get("noise_ratio") != noise_ratio:
            continue
        if job["kind"] != kind:
            continue
        if step is not None and job.get("step") != step:
            continue
        return job
    return None


def infer_prerequisites(job: dict, manifest: dict) -> list[str]:
    prereqs = list(job.get("requires_steps", []))
    ds = job["dataset"]
    nt = job.get("noise_type")
    nr = job.get("noise_ratio")
    kind = job["kind"]
    step = job.get("step")

    def add(kind_name: str, step_no=None):
        found = find_job(manifest, ds, nt, nr, kind_name, step_no)
        if found:
            prereqs.append(found["id"])

    if kind != "gen":
        add("gen")
    if kind == "step0":
        pass
    elif kind == "raw" and step is not None:
        if step <= 1:
            add("step0")
        else:
            add("contra", step - 1)
    elif kind == "contra":
        add("raw", step)
    elif kind == "eval":
        add("contra", step)
    elif kind in PARALLEL_KINDS | {"rehearsal"}:
        add("step0")
        add("raw", step)
    elif kind == "corrupt_gen":
        pass
    elif kind == "corrupt_eval":
        add("corrupt_gen")
        add("contra", step)

    out: list[str] = []
    seen = set()
    for dep in prereqs:
        if dep not in seen:
            seen.add(dep)
            out.append(dep)
    return out


def prerequisites_ok(job: dict, manifest: dict, state: dict) -> bool:
    if not deps_ok(job, state):
        return False
    for dep in infer_prerequisites(job, manifest):
        if not job_done(state, dep):
            return False
    return True


def reconcile_running(state: dict) -> None:
    running = state.get("running", {})
    stale = []
    for jid, meta in running.items():
        pid = meta.get("pid", -1)
        if pid is None or pid <= 0:
            continue
        if not pid_alive(pid):
            stale.append(jid)
    for jid in stale:
        running.pop(jid, None)


def infer_job_from_cmdline(cmdline: str, manifest: dict) -> str | None:
    for job in manifest["jobs"]:
        ds = job["dataset"]
        if ds not in cmdline:
            continue
        step = job.get("step")
        if step is not None and f"--step {step}" not in cmdline and f"--step {step} " not in cmdline:
            continue
        kind = job["kind"]
        if kind == "gen":
            if "gen_" in cmdline and str(job.get("noise_ratio")) in cmdline:
                return job["id"]
        elif kind == "step0" and "run_experiment.py" in cmdline and "--step 0" in cmdline:
            return job["id"]
        elif kind == "raw" and "run_experiment.py" in cmdline and "worker_raw" in cmdline:
            return job["id"]
        elif kind == "contra" and "run_contra.py" in cmdline:
            return job["id"]
        elif kind == "lnl" and job.get("uni_name") and job["uni_name"] in cmdline:
            return job["id"]
        elif kind == "tta" and job.get("uni_name"):
            uni = job["uni_name"]
            if uni == "CoTTA" and "tta.py" in cmdline:
                return job["id"]
            if uni == "PLF" and "run_plf.py" in cmdline:
                return job["id"]
            if uni == "SoTTA" and "run_sotta.py" in cmdline:
                return job["id"]
        elif kind == "rehearsal" and "Rehearsal" in cmdline:
            return job["id"]
    return None


def external_running_jobs(manifest: dict, state: dict) -> dict[str, dict]:
    tracked_pids = {meta.get("pid") for meta in state.get("running", {}).values()}
    external: dict[str, dict] = {}
    for pid in list_training_pids():
        if pid in tracked_pids:
            continue
        if not pid_alive(pid):
            continue
        cmdline = ""
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_text()
        except OSError:
            proc = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
            if proc.returncode != 0:
                continue
            cmdline = proc.stdout.strip()
        job_id = infer_job_from_cmdline(cmdline, manifest)
        if job_id:
            external[job_id] = {"pid": pid, "started_at": utc_now(), "external": True}
    return external


def active_running(state: dict, manifest: dict) -> dict[str, dict]:
    reconcile_running(state)
    merged = dict(state.get("running", {}))
    for jid, meta in external_running_jobs(manifest, state).items():
        merged.setdefault(jid, meta)
    return merged


def pick_next(manifest: dict, state: dict) -> dict | None:
    defaults = manifest["defaults"]
    for job in manifest["jobs"]:
        jid = job["id"]
        if job_done(state, jid) or jid in state.get("skipped", {}):
            continue
        if jid in active_running(state, manifest):
            continue
        if not prerequisites_ok(job, manifest, state):
            continue
        if job["kind"] != "gen" and not dataset_ready(job, defaults):
            gen_job = find_job(
                manifest,
                job["dataset"],
                job.get("noise_type"),
                job.get("noise_ratio"),
                "gen",
            )
            if gen_job and not job_done(state, gen_job["id"]):
                return gen_job
            continue
        return job
    return None


def chain_conflict(job: dict, running: dict[str, dict], manifest: dict) -> bool:
    jobs_by_id = {j["id"]: j for j in manifest["jobs"]}
    ck = chain_key(job)
    kind = job["kind"]

    if kind in EXCLUSIVE_KINDS:
        for rid, _ in running.items():
            rjob = jobs_by_id.get(rid)
            if rjob and chain_key(rjob) == ck:
                return True
        return False

    if kind in PARALLEL_KINDS:
        parallel_count = 0
        for rid, _ in running.items():
            rjob = jobs_by_id.get(rid)
            if rjob and chain_key(rjob) == ck and rjob["kind"] in PARALLEL_KINDS:
                parallel_count += 1
        return parallel_count >= BASELINE_PARALLEL_PER_CHAIN
    return False


def pick_parallel_batch(
    manifest: dict,
    state: dict,
    max_workers: int,
    running: dict[str, dict] | None = None,
) -> list[dict]:
    running = running if running is not None else active_running(state, manifest)
    defaults = manifest["defaults"]
    batch: list[dict] = []
    batch_ids: set[str] = set()

    for job in manifest["jobs"]:
        if len(batch) >= max_workers:
            break
        jid = job["id"]
        if jid in batch_ids:
            continue
        if job_done(state, jid) or jid in state.get("skipped", {}):
            continue
        if jid in running:
            continue
        if not prerequisites_ok(job, manifest, state):
            continue
        if job["kind"] != "gen" and not dataset_ready(job, defaults):
            gen_job = find_job(
                manifest,
                job["dataset"],
                job.get("noise_type"),
                job.get("noise_ratio"),
                "gen",
            )
            if gen_job and not job_done(state, gen_job["id"]) and gen_job["id"] not in running:
                if gen_job["id"] not in batch_ids and not chain_conflict(gen_job, running, manifest):
                    batch.append(gen_job)
                    batch_ids.add(gen_job["id"])
            continue
        virtual_running = dict(running)
        for picked in batch:
            virtual_running[picked["id"]] = {"pid": -1}
        if chain_conflict(job, virtual_running, manifest):
            continue
        batch.append(job)
        batch_ids.add(jid)

    return batch


def build_command(job: dict, defaults: dict, python: str) -> tuple[list[str], dict]:
    d = {**defaults, **job}
    kind = job["kind"]
    env = {"PYTHONPATH": str(REPO)}

    if kind == "gen":
        script = REPO / "gen_dataset" / job["generator"]
        cmd = [
            python,
            str(script),
            "--data_dir",
            str(REPO / "data" / job["dataset"] / "normal"),
            "--noise_type",
            job["noise_type"],
            "--noise_ratio",
            str(job["noise_ratio"]),
            "--num_versions",
            "4",
            "--retention_ratios",
            "0.5",
            "0.3",
            "0.1",
            "0.05",
            "--balanced",
        ]
        return cmd, env

    if kind == "corrupt_gen":
        cmd = [
            python,
            str(REPO / "gen_dataset" / "gen_corruption_data.py"),
            "--dataset",
            job["dataset"],
            "--noise_ratio",
            str(job["noise_ratio"]),
            "--noise_type",
            job["noise_type"],
            "--balanced",
            "--corruptions",
            "gaussian_noise",
            "gaussian_blur",
            "jpeg",
            "contrast",
        ]
        return cmd, env

    base = [
        "--dataset",
        d["dataset"],
        "--model",
        d["model"],
        "--noise_ratio",
        str(d["noise_ratio"]),
        "--noise_type",
        d["noise_type"],
        "--num_epochs",
        str(d.get("num_epochs", defaults["num_epochs"])),
        "--learning_rate",
        str(d.get("learning_rate", defaults["learning_rate"])),
        "--optimizer",
        d.get("optimizer", defaults["optimizer"]),
        "--batch_size",
        str(d.get("batch_size", defaults["batch_size"])),
        "--seed",
        str(d.get("seed", defaults["seed"])),
    ]
    if d.get("balanced", defaults.get("balanced")):
        base.append("--balanced")

    if kind == "corrupt_eval":
        cmd = [
            python,
            str(REPO / "scripts" / "evaluate_corruptions.py"),
            *base,
            "--step",
            str(job["step"]),
            "--model_suffix",
            job.get("model_suffix", "worker_tta"),
        ]
        if job.get("eval_map"):
            cmd.append("--eval_map")
        return cmd, env

    if kind == "eval":
        cmd = [
            python,
            str(REPO / "scripts" / "evaluate_checkpoint.py"),
            *base,
            "--step",
            str(job["step"]),
            "--model_suffix",
            job.get("model_suffix", "worker_tta"),
        ]
        if job.get("eval_map"):
            cmd.append("--eval_map")
        return cmd, env

    if kind == "step0":
        cmd = [python, str(REPO / "run_experiment.py"), *base, "--step", "0"]
        return cmd, env

    if kind == "raw":
        cmd = [
            python,
            str(REPO / "run_experiment.py"),
            *base,
            "--step",
            str(job["step"]),
            "--model_suffix",
            "worker_raw",
        ]
        return cmd, env

    if kind == "contra":
        cmd = [
            python,
            str(REPO / "run_contra.py"),
            *base,
            "--step",
            str(job["step"]),
            "--adapt_epochs",
            str(d.get("adapt_epochs", defaults.get("adapt_epochs", 1))),
        ]
        if job.get("eval_map"):
            cmd.append("--eval_map")
        return cmd, env

    if kind == "rehearsal":
        raw_ckpt = REPO / "ckpt" / job["dataset"] / case_name(job, defaults) / f"step_{job['step']}"
        raw_path = raw_ckpt / f"{job['model']}_worker_raw.pth"
        if not raw_path.exists():
            cmd = [
                python,
                str(REPO / "run_experiment.py"),
                *base,
                "--step",
                str(job["step"]),
                "--model_suffix",
                "worker_raw",
                "--uni_name",
                "Rehearsal",
            ]
            return cmd, env
        cmd = [
            python,
            str(REPO / "run_experiment.py"),
            *base,
            "--step",
            str(job["step"]),
            "--train_aux",
            "--uni_name",
            "Rehearsal",
        ]
        return cmd, env

    if kind == "lnl":
        cmd = [
            python,
            str(REPO / "baseline_code" / "colearn" / "main.py"),
            *base,
            "--step",
            str(job["step"]),
            "--uni_name",
            job["uni_name"],
        ]
        return cmd, env

    if kind == "tta":
        script_map = {
            "CoTTA": REPO / "baseline_code" / "cotta-main" / "cifar" / "tta.py",
            "PLF": REPO / "baseline_code" / "PLF-main" / "cifar" / "run_plf.py",
            "SoTTA": REPO / "baseline_code" / "sotta" / "run_sotta.py",
        }
        cmd = [
            python,
            str(script_map[job["uni_name"]]),
            *base,
            "--step",
            str(job["step"]),
            "--uni_name",
            job["uni_name"],
        ]
        return cmd, env

    raise ValueError(f"Unknown job kind: {kind}")


def run_job(job: dict, defaults: dict, python: str) -> tuple[int, str]:
    cmd, env = build_command(job, defaults, python)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{job['id']}.log"
    env = {**os.environ, **env}
    with log_path.open("w") as logf:
        logf.write(f"# {utc_now()} job={job['id']}\n")
        logf.write("# " + " ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT)
    tail = log_path.read_text().splitlines()[-8:]
    summary = "\n".join(tail)
    return proc.returncode, summary


def launch_job(job: dict, defaults: dict, python: str) -> tuple[subprocess.Popen, Path]:
    cmd, env = build_command(job, defaults, python)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{job['id']}.log"
    logf = log_path.open("w")
    logf.write(f"# {utc_now()} job={job['id']}\n")
    logf.write("# " + " ".join(cmd) + "\n\n")
    logf.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=REPO,
        env={**os.environ, **env},
        stdout=logf,
        stderr=subprocess.STDOUT,
    )
    return proc, log_path


def progress_counts(manifest: dict, state: dict) -> tuple[int, int]:
    total = len(manifest["jobs"])
    done = sum(1 for j in manifest["jobs"] if job_done(state, j["id"]))
    return done, total
