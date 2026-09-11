from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Sequence

from rich.console import Console
from rich.table import Table

from nnunetv2.paths import nnUNet_results


EXPECTED_FOLDS = tuple(range(5))


@dataclass(frozen=True)
class ExperimentResult:
    identifier: str
    configuration: str
    completed_folds: tuple[int, ...]
    mean_dsc: float
    created_at: float


def configuration_from_identifier(identifier: str) -> str:
    """Return the configuration component of an nnU-Net experiment identifier."""
    parts = identifier.split("__")
    return parts[2] if len(parts) == 3 else identifier


def load_fold_dsc(summary_file: Path) -> float | None:
    try:
        with summary_file.open() as file:
            dsc = json.load(file)["foreground_mean"]["Dice"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        warnings.warn(f"Skipping invalid validation summary {summary_file}: {error}", stacklevel=2)
        return None

    if isinstance(dsc, bool) or not isinstance(dsc, (int, float)) or not math.isfinite(dsc):
        warnings.warn(
            f"Skipping invalid validation summary {summary_file}: foreground_mean.Dice is not a finite number",
            stacklevel=2,
        )
        return None
    return float(dsc)


def resolve_results_dataset(dataset_id: int, results_directory: Path) -> Path:
    dataset_prefix = f"Dataset{dataset_id:03d}"
    candidates = sorted(
        directory
        for directory in results_directory.iterdir()
        if directory.is_dir() and directory.name.startswith(dataset_prefix)
    )
    if not candidates:
        raise RuntimeError(
            f"Could not find a results directory for dataset ID {dataset_id} under {results_directory}"
        )
    if len(candidates) > 1:
        formatted = "\n".join(f"  - {candidate.name}" for candidate in candidates)
        raise RuntimeError(
            f"More than one results directory matches dataset ID {dataset_id} under {results_directory}:\n{formatted}"
        )
    return candidates[0]


def collect_experiment_results(dataset_id: int) -> list[ExperimentResult]:
    """Collect completed cross-validation results for a dataset in ``nnUNet_results``."""
    results_directory = Path(nnUNet_results.require())
    dataset_directory = resolve_results_dataset(dataset_id, results_directory)

    experiments = []
    for experiment_directory in sorted(dataset_directory.iterdir()):
        if not experiment_directory.is_dir():
            continue

        completed_folds = []
        fold_scores = []
        for fold in EXPECTED_FOLDS:
            summary_file = experiment_directory / f"fold_{fold}" / "validation" / "summary.json"
            if not summary_file.is_file():
                continue
            dsc = load_fold_dsc(summary_file)
            if dsc is not None:
                completed_folds.append(fold)
                fold_scores.append(dsc)

        if not fold_scores:
            continue
        directory_stat = experiment_directory.stat()
        experiments.append(
            ExperimentResult(
                identifier=experiment_directory.name,
                configuration=configuration_from_identifier(experiment_directory.name),
                completed_folds=tuple(completed_folds),
                mean_dsc=fmean(fold_scores),
                created_at=getattr(directory_stat, "st_birthtime", directory_stat.st_ctime),
            )
        )
    return experiments


def sort_experiment_results(
    experiments: Sequence[ExperimentResult], sort_by: str
) -> list[ExperimentResult]:
    if sort_by == "date":
        return sorted(experiments, key=lambda result: (-result.created_at, result.configuration, result.identifier))
    if sort_by == "dsc":
        return sorted(experiments, key=lambda result: (-result.mean_dsc, result.configuration, result.identifier))
    raise ValueError(f"Unknown sort criterion: {sort_by}")


def build_results_table(experiments: Sequence[ExperimentResult]) -> Table:
    table = Table(title="nnU-Net validation results")
    table.add_column("Configuration")
    table.add_column("Folds Completed")
    table.add_column("Folds Remaining")
    table.add_column("Mean DSC", justify="right")

    for experiment in experiments:
        completed = ", ".join(map(str, experiment.completed_folds))
        remaining = ", ".join(str(fold) for fold in EXPECTED_FOLDS if fold not in experiment.completed_folds)
        table.add_row(experiment.configuration, completed, remaining, f"{experiment.mean_dsc:.4f}")
    return table


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report completed nnU-Net validation results for a dataset.")
    parser.add_argument("-d", "--dataset", type=int, required=True, help="Dataset ID")
    parser.add_argument(
        "-s",
        "--sort",
        choices=("date", "dsc"),
        default="date",
        help="Sort by newest experiment directory (date, default) or highest mean DSC (dsc)",
    )
    return parser.parse_args(argv)


def report_results_entry(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiments = sort_experiment_results(collect_experiment_results(args.dataset), args.sort)
    console = Console()
    if not experiments:
        console.print("No completed validation summaries found.")
        return 0
    console.print(build_results_table(experiments))
    return 0


if __name__ == "__main__":
    report_results_entry()
