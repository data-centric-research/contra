# CONTRA 论文 Smoke 套件编排

54 个 job 定义见 `paper_smoke_manifest.json`，状态写入 `runs/paper_smoke_state.json`，日志在 `logs/paper_smoke/<job_id>.log`。

## 串行 tick（原有，兼容 /loop）

每轮最多跑 **1** 个 job；若已有训练进程（含外部 PLF/CoTTA 等）则 `action=wait`。

```bash
cd /Users/suizhihao/Documents/phd/thesis/papers/ICANN/contra_accept/contra
.venv/bin/python scripts/run_paper_smoke_tick.py          # 跑 1 个
.venv/bin/python scripts/run_paper_smoke_tick.py --status
.venv/bin/python scripts/run_paper_smoke_tick.py --force  # 忽略负载门控
```

**/loop 唤醒命令（串行）**

```bash
cd /Users/suizhihao/Documents/phd/thesis/papers/ICANN/contra_accept/contra && \
.venv/bin/python scripts/run_paper_smoke_tick.py
```

返回 `action=wait` 时按 `next_sleep_sec`（默认 120s）重新 arm；`action=complete` 时停止。

## 并行 runner（推荐）

```bash
.venv/bin/python scripts/run_paper_smoke_parallel.py --status
.venv/bin/python scripts/run_paper_smoke_parallel.py --dry-run
.venv/bin/python scripts/run_paper_smoke_parallel.py --once              # 跑一批后退出
.venv/bin/python scripts/run_paper_smoke_parallel.py --once --max-workers 2
.venv/bin/python scripts/run_paper_smoke_parallel.py --loop              # 持续调度直到完成
```

- **`--max-workers 0`（默认）**：按 CPU 负载（<70% 核数）与空闲内存（每 worker 约 3GiB，保留 3GiB）自动计算，上限 4。
- **依赖**：同数据集链路上 `gen → s0 → raw → contra → eval` 串行；同 step 的 TTA/LNL 基线最多 **3** 个并行。
- **跨数据集**：CIFAR-100 / Pet-37 等不同 chain 可同时跑（例如 `cifar100_asy_gen` 与 `pet37_sym_gen`）。
- **外部训练**：通过 `pgrep` 识别已在跑的 job（如 tick 启动的 PLF），占用 worker 槽位，**不会重复启动**。
- **状态锁**：`runs/paper_smoke_state.lock` + `running` 字段，避免并发写 state。

**/loop 唤醒命令（并行，一批）**

```bash
cd /Users/suizhihao/Documents/phd/thesis/papers/ICANN/contra_accept/contra && \
.venv/bin/python scripts/run_paper_smoke_parallel.py --once --max-workers 2
```

负载高时返回 `action=wait`；可与串行 tick **不要同时**对同一 state 发起新 job（并行 runner 会识别外部进程并等待）。

## 并行度建议

| 阶段 | 建议 `--max-workers` |
|------|----------------------|
| CIFAR-10 剩余基线（PLF/SoTTA） | 1（PLF 占用时）或 2 |
| CIFAR-100 gen + Pet-37 gen 起步 | 2 |
| 单数据集 raw/contra 链 | 1（链内独占） |
| 多 Pet-37 noise ratio 实验 | 2–3（不同 chain） |

## 已知 prep

- **Rehearsal**：若报 `Rehearsal/..._worker_raw.pth not found`，需先把 `step_1/cifar-resnet18_worker_raw.pth` 复制到 `step_1/Rehearsal/`（或重跑 rehearsal job，脚本会先训 raw）。
