# Paper Smoke Runs

This folder contains scripts for running a small set of paper-related smoke checks. The jobs are defined in `paper_smoke_manifest.json`, state is written to `runs/paper_smoke_state.json`, and logs are written to `logs/paper_smoke/<job_id>.log`.

## Single-Job Runner

Run at most one pending job:

```bash
python scripts/run_paper_smoke_tick.py
```

Check state:

```bash
python scripts/run_paper_smoke_tick.py --status
```

Run one job even when the load check would normally wait:

```bash
python scripts/run_paper_smoke_tick.py --force
```

When the script returns `action=wait`, wait for the reported `next_sleep_sec` before starting another job. When it returns `action=complete`, the manifest has finished.

## Parallel Runner

Inspect the queue:

```bash
python scripts/run_paper_smoke_parallel.py --status
python scripts/run_paper_smoke_parallel.py --dry-run
```

Run one batch:

```bash
python scripts/run_paper_smoke_parallel.py --once --max-workers 2
```

Run until the manifest is complete:

```bash
python scripts/run_paper_smoke_parallel.py --loop --max-workers 2
```

Use `--max-workers 0` to let the script choose a worker count from the current CPU and memory load. The runner keeps dataset chains in order, so generation, step-0 training, raw-worker jobs, CONTRA jobs, and evaluation jobs are not started out of sequence.

## Notes

- Do not run the tick runner and the parallel runner against the same state file at the same time.
- Rehearsal jobs require the stage raw-worker checkpoint for the same case and step. The manifest includes the raw-worker step before the rehearsal fine-tuning step.
