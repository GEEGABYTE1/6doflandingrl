from __future__ import annotations

import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_dir / "failure_mode_counts.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({(row["controller_label"], row["disturbance_level"]) for row in rows})
    modes = sorted({row["failure_mode"] for row in rows})
    x = np.arange(len(keys), dtype=float)
    bottoms = np.zeros(len(keys), dtype=float)
    fig, ax = plt.subplots(figsize=(12.0, 4.5), constrained_layout=True)
    for mode in modes:
        counts = []
        for controller_label, disturbance_level in keys:
            count = 0
            for row in rows:
                if (
                    row["controller_label"] == controller_label
                    and row["disturbance_level"] == disturbance_level
                    and row["failure_mode"] == mode
                ):
                    count = int(row["count"])
                    break
            counts.append(count)
        counts_array = np.asarray(counts, dtype=float)
        ax.bar(x, counts_array, bottom=bottoms, label=mode)
        bottoms += counts_array
    ax.set_xticks(x)
    ax.set_xticklabels([f"{controller}\n{level}" for controller, level in keys], rotation=45, ha="right")
    ax.set_ylabel("episode count")
    ax.legend(fontsize=7, ncols=3)
    fig.savefig(args.output_dir / "failure_modes.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()

