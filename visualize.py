
from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from mpi_utils import EMPTY


def save_space_time_diagrams(history_speed: list[np.ndarray], history_emergency: list[np.ndarray], out_dir: Path) -> None:
    if not history_speed:
        return
    speed_stack = np.array(history_speed, dtype=np.int32)
    emergency_stack = np.array(history_emergency, dtype=np.int32)

    occupancy = (speed_stack != EMPTY).astype(int)
    emergency_occ = emergency_stack.astype(int)

    lanes = occupancy.shape[1]
    for lane in range(lanes):
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(
            occupancy[:, lane, :],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(f"Space-Time Diagram - Lane {lane + 1}")
        ax.set_xlabel("Road Position")
        ax.set_ylabel("Time Step")
        fig.tight_layout()
        fig.savefig(out_dir / f"space_time_lane_{lane + 1}.png", dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(
        emergency_occ.sum(axis=1),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    ax.set_title("Emergency Vehicle Heatmap (All Lanes)")
    ax.set_xlabel("Road Position")
    ax.set_ylabel("Time Step")
    fig.tight_layout()
    fig.savefig(out_dir / "emergency_heatmap.png", dpi=220)
    plt.close(fig)


def save_metrics_csv(rows: list[dict[str, float]], out_path: Path) -> None:
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_metrics_plot(rows: list[dict[str, float]], out_path: Path) -> None:
    if not rows:
        return
    steps = [r["step"] for r in rows]
    density = [r["density"] for r in rows]
    avg_speed = [r["avg_speed"] for r in rows]
    flow = [r["flow"] for r in rows]
    queue = [r["global_queue"] for r in rows]
    green = [r["green_signals"] for r in rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(steps, density, label="Density")
    ax.plot(steps, avg_speed, label="Average Speed")
    ax.plot(steps, flow, label="Flow")
    ax.plot(steps, queue, label="Queue Near Signals")
    ax.plot(steps, green, label="Green Signals")
    ax.set_title("Traffic Metrics Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_benchmark_csv(rows: list[dict[str, float]], out_path: Path) -> None:
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
