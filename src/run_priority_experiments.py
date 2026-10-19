"""Launch the 300k-seed comparison and stronger alternating experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from compare_algorithms import (
    _write_comparison_csv,
    _write_comparison_plots,
    _write_report,
    priority_config,
)
from config import AlternatingConfig, MAX_STEPS


ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).resolve()
SEEDS = (42, 43, 44)
MAX_WORKERS = 3
RESULTS_DIR = ROOT / "results/priority_300k"
LOGS_DIR = ROOT / "logs/experiments/priority_300k/launcher"


def _run_child(arguments: list[str], log_path: Path) -> None:
    """Execute one independent experiment process and preserve its output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as output:
        subprocess.run(
            [sys.executable, str(WORKER), *arguments],
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _run_jobs(jobs: list[tuple[list[str], Path]]) -> None:
    """Run bounded subprocess jobs concurrently and fail if any child fails."""
    errors: list[tuple[Path, Exception]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_run_child, arguments, log_path): log_path
            for arguments, log_path in jobs
        }
        for future in as_completed(futures):
            log_path = futures[future]
            try:
                future.result()
                print(f"Completed: {log_path.name}")
            except Exception as error:
                errors.append((log_path, error))
    if errors:
        detail = "; ".join(f"{path}: {error}" for path, error in errors)
        raise RuntimeError(f"One or more experiment workers failed: {detail}")


def run_comparison() -> None:
    """Run six PPO/DQN seed workers, then aggregate their evaluations."""
    config = priority_config()
    jobs: list[tuple[list[str], Path]] = []
    for algorithm in ("ppo", "dqn"):
        for seed in SEEDS:
            log_path = LOGS_DIR / f"comparison_{algorithm}_seed_{seed}.log"
            jobs.append((
                [
                    "--worker",
                    "comparison",
                    "--seed",
                    str(seed),
                    "--algorithm",
                    algorithm,
                ],
                log_path,
            ))
    print(f"Launching {len(jobs)} comparison workers, at most {MAX_WORKERS} at once.")
    _run_jobs(jobs)

    results_dir = LOGS_DIR / "worker_results"
    records: list[dict[str, str | int | float]] = []
    for algorithm in ("ppo", "dqn"):
        for seed in SEEDS:
            result_path = results_dir / f"comparison_{algorithm}_seed_{seed}.json"
            records.extend(json.loads(result_path.read_text(encoding="utf-8")))
    _write_comparison_csv(config, records)
    _write_comparison_plots(config, records)
    _write_report(config, records)
    print(f"Comparison results: {config.results_path}")


def _priority_alternating_config(seed: int) -> AlternatingConfig:
    """Build one isolated seed configuration with baseline restarts."""
    seed_results = RESULTS_DIR / "alternating" / f"seed_{seed}.csv"
    return replace(
        AlternatingConfig(),
        n_rounds=4,
        pacman_steps_per_round=100_000,
        ghost_steps_per_round=100_000,
        seeds=(seed,),
        initial_pacman_model_path=Path("models/best_model.zip"),
        restart_pacman_each_round=True,
        evaluation_episodes=30,
        max_steps=MAX_STEPS,
        eval_freq=50_000,
        models_dir=Path("models/experiments/priority_300k/alternating"),
        logs_dir=Path("logs/experiments/priority_300k/alternating"),
        results_path=seed_results,
    )


def run_alternating() -> None:
    """Launch three isolated alternating training processes and merge results."""
    jobs = []
    for seed in SEEDS:
        jobs.append((
            ["--worker", "alternating", "--seed", str(seed)],
            LOGS_DIR / f"alternating_seed_{seed}.log",
        ))
    print(f"Launching {len(jobs)} alternating workers, at most {MAX_WORKERS} at once.")
    _run_jobs(jobs)

    output_path = RESULTS_DIR / "alternating_rounds.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer: csv.DictWriter | None = None
        for seed in SEEDS:
            seed_path = RESULTS_DIR / "alternating" / f"seed_{seed}.csv"
            with seed_path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames is None:
                    raise ValueError(f"Alternating worker output has no header: {seed_path}")
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
    catch_rows: list[dict[str, str]] = []
    for seed in SEEDS:
        catch_path = (
            RESULTS_DIR
            / "alternating"
            / f"baseline_ghost_catch_rates_seed_{seed}.csv"
        )
        if catch_path.exists():
            with catch_path.open(newline="", encoding="utf-8") as source:
                catch_rows.extend(csv.DictReader(source))
    if catch_rows:
        catch_path = RESULTS_DIR / "baseline_ghost_catch_rates.csv"
        with catch_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=catch_rows[0].keys())
            writer.writeheader()
            seen_rule_row = False
            for row in catch_rows:
                if row["ghost_type"] == "rule":
                    if seen_rule_row:
                        continue
                    seen_rule_row = True
                writer.writerow(row)
    print(f"Alternating results: {output_path}")


def _worker(args: argparse.Namespace) -> None:
    """Execute one seed/algorithm worker and persist its structured output."""
    if args.kind == "comparison":
        from compare_algorithms import _run_seed

        config = priority_config()
        rows = _run_seed(
            args.seed,
            replace(config, seeds=(args.seed,)),
            algorithms=(args.algorithm,),
        )
        result_name = f"comparison_{args.algorithm}_seed_{args.seed}.json"
    else:
        from train_alternating import run_alternating

        run_alternating(_priority_alternating_config(args.seed))
        return
    result_path = LOGS_DIR / "worker_results" / result_name
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("comparison", "alternating"))
    parser.add_argument("--worker", choices=("comparison", "alternating"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--algorithm", choices=("ppo", "dqn"))
    args = parser.parse_args()
    if args.worker:
        if args.seed is None or (
            args.worker == "comparison" and args.algorithm is None
        ):
            parser.error("Worker mode requires --seed and comparison requires --algorithm.")
        args.kind = args.worker
        _worker(args)
        return
    if args.task == "comparison":
        run_comparison()
    elif args.task == "alternating":
        run_alternating()
    else:
        parser.error("Choose --task comparison or --task alternating.")


if __name__ == "__main__":
    main()
