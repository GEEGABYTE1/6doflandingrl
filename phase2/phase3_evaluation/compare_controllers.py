"""Build paper-ready Phase 3 comparison tables from Monte Carlo outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from phase3_evaluation.metrics import (
        count_failure_modes,
        summarize_controller_overall,
        summarize_episode_rows,
        write_csv,
        write_json,
    )
else:
    from .metrics import (
        count_failure_modes,
        summarize_controller_overall,
        summarize_episode_rows,
        write_csv,
        write_json,
    )

def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if value in {"True", "False"}:
                    parsed[key] = value == "True"
                else:
                    try:
                        parsed[key] = int(value)
                    except ValueError:
                        try:
                            parsed[key] = float(value)
                        except ValueError:
                            parsed[key] = value
            rows.append(parsed)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_rows = read_rows(args.input_dir / "episodes.csv")
    summary_rows = summarize_episode_rows(episode_rows)
    overall_rows = summarize_controller_overall(episode_rows)
    failure_rows = count_failure_modes(episode_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary_by_controller_level.csv", summary_rows)
    write_json(args.output_dir / "summary_by_controller_level.json", summary_rows)
    write_csv(args.output_dir / "controller_overall_summary.csv", overall_rows)
    write_json(args.output_dir / "controller_overall_summary.json", overall_rows)
    write_csv(args.output_dir / "failure_mode_counts.csv", failure_rows)
    write_json(args.output_dir / "failure_mode_counts.json", failure_rows)


if __name__ == "__main__":
    main()

