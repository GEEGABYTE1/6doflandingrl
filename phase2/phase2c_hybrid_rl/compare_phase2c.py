from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "METRICS_JSON"),
        required=True,
        help="Label and metrics.json path for one compared run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for label, metrics_path in args.run:
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        rows.append(
            {
                "label": label,
                "success": bool(metrics["success"]),
                "failure_modes": metrics["failure_modes"],
                "vertical_touchdown_velocity_mps": float(metrics["vertical_touchdown_velocity_mps"]),
                "horizontal_touchdown_velocity_mps": float(metrics["horizontal_touchdown_velocity_mps"]),
                "landing_position_error_m": float(metrics["landing_position_error_m"]),
                "tilt_angle_deg": float(metrics["tilt_angle_deg"]),
                "angular_rate_norm_radps": float(metrics["angular_rate_norm_radps"]),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "phase2c_comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, sort_keys=True)
    with (args.output_dir / "phase2c_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()

