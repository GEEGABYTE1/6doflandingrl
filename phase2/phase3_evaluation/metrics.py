from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import numpy as np

PRIMARY_METRICS: tuple[str, ...] = (
    "landing_position_error_m",
    "vertical_touchdown_velocity_mps",
    "tilt_angle_deg",
    "fuel_used_kg",
)

RAW_METRICS: tuple[str, ...] = (
    "success",
    "landing_position_error_m",
    "vertical_touchdown_velocity_mps",
    "horizontal_touchdown_velocity_mps",
    "touchdown_speed_mps",
    "tilt_angle_deg",
    "angular_rate_norm_radps",
    "fuel_used_kg",
    "termination_reason",
    "failure_modes",
    "touchdown_detected",
    "duration_s",
)

METRIC_DEFINITIONS: dict[str, str] = {
    "success_rate": "Fraction of evaluated episodes satisfying all touchdown success criteria.",
    "landing_position_error_m": "Euclidean lateral landing error at touchdown in meters.",
    "vertical_touchdown_velocity_mps": "Signed vertical touchdown velocity in m/s; negative is downward.",
    "tilt_angle_deg": "Touchdown tilt angle between body +z and inertial +z in degrees.",
    "fuel_used_kg": "Initial vehicle mass minus final vehicle mass in kilograms.",
    "horizontal_touchdown_velocity_mps": "Horizontal touchdown speed magnitude in m/s.",
    "touchdown_speed_mps": "Total touchdown speed magnitude in m/s.",
    "angular_rate_norm_radps": "Touchdown body-rate vector norm in rad/s.",
    "termination_reason": "Terminal rollout status such as touchdown, timeout, or divergence.",
    "failure_modes": "Semicolon-delimited failure taxonomy labels from the shared touchdown metric function.",
    "touchdown_detected": "Whether the rollout ended by ground contact rather than timeout/divergence.",
    "duration_s": "Elapsed rollout duration in seconds.",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def normalized_failure_modes(value: str) -> tuple[str, ...]:
    modes = [mode.strip() for mode in str(value).split(";") if mode.strip()]
    return ("none",) if not modes else tuple(modes)


def _quantile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return float("nan")
    return float(np.quantile(array, q))


@dataclass(frozen=True)
class SummaryKey:
    controller_id: str
    disturbance_level: str


def summarize_episode_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[SummaryKey, list[dict[str, object]]] = {}
    for row in rows:
        key = SummaryKey(
            controller_id=str(row["controller_id"]),
            disturbance_level=str(row["disturbance_level"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: (item.controller_id, item.disturbance_level)):
        group = grouped[key]
        first = group[0]
        out: dict[str, object] = {
            "controller_id": key.controller_id,
            "controller_label": first["controller_label"],
            "controller_type": first["controller_type"],
            "disturbance_level": key.disturbance_level,
            "disturbance_index": int(first["disturbance_index"]),
            "episode_count": len(group),
        }
        success_values = [bool(row["success"]) for row in group]
        out["success_count"] = int(sum(success_values))
        out["success_rate"] = float(np.mean(success_values))
        for metric in PRIMARY_METRICS:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = float(mean(values))
            out[f"{metric}_median"] = float(median(values))
            out[f"{metric}_std"] = float(np.std(values))
        for metric in ("landing_position_error_m", "vertical_touchdown_velocity_mps", "fuel_used_kg"):
            values = [float(row[metric]) for row in group]
            out[f"{metric}_p10"] = _quantile(values, 0.10)
            out[f"{metric}_p90"] = _quantile(values, 0.90)
        summary_rows.append(out)
    return summary_rows


def summarize_controller_overall(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["controller_id"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for controller_id in sorted(grouped):
        group = grouped[controller_id]
        first = group[0]
        out: dict[str, object] = {
            "controller_id": controller_id,
            "controller_label": first["controller_label"],
            "controller_type": first["controller_type"],
            "episode_count": len(group),
        }
        success_values = [bool(row["success"]) for row in group]
        out["success_count"] = int(sum(success_values))
        out["success_rate"] = float(np.mean(success_values))
        for metric in PRIMARY_METRICS:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = float(mean(values))
            out[f"{metric}_median"] = float(median(values))
            out[f"{metric}_std"] = float(np.std(values))
        summary_rows.append(out)
    return summary_rows

def count_failure_modes(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[SummaryKey, Counter[str]] = {}
    first_rows: dict[SummaryKey, dict[str, object]] = {}
    for row in rows:
        key = SummaryKey(
            controller_id=str(row["controller_id"]),
            disturbance_level=str(row["disturbance_level"]),
        )
        first_rows.setdefault(key, row)
        counter = grouped.setdefault(key, Counter())
        for mode in normalized_failure_modes(str(row["failure_modes"])):
            counter[mode] += 1
    failure_rows: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: (item.controller_id, item.disturbance_level)):
        meta = first_rows[key]
        for mode, count in sorted(grouped[key].items()):
            failure_rows.append(
                {
                    "controller_id": key.controller_id,
                    "controller_label": meta["controller_label"],
                    "controller_type": meta["controller_type"],
                    "disturbance_level": key.disturbance_level,
                    "disturbance_index": int(meta["disturbance_index"]),
                    "failure_mode": mode,
                    "count": int(count),
                }
            )
    return failure_rows

__all__ = [
    "METRIC_DEFINITIONS",
    "PRIMARY_METRICS",
    "RAW_METRICS",
    "count_failure_modes",
    "normalized_failure_modes",
    "summarize_controller_overall",
    "summarize_episode_rows",
    "write_csv",
    "write_json",
]

