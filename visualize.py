from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from mpi_utils import EMPTY

# 🏎️ Upgrade: Apply a dark "telemetry" style preferred in motorsport data analysis
plt.style.use("dark_background")

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
        # Upgrade: Use 'bone' colormap for a sleek radar/telemetry look
        im = ax.imshow(
            occupancy[:, lane, :],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="bone"
        )
        ax.set_title(f"Space-Time Diagram - Lane {lane + 1}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Road Position", fontsize=12)
        ax.set_ylabel("Time Step", fontsize=12)
        ax.grid(False) # Turn off grid for heatmaps to keep it clean
        fig.tight_layout()
        fig.savefig(out_dir / f"space_time_lane_{lane + 1}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    # Upgrade: Use 'inferno' colormap to make emergency hotspots glow intensely
    ax.imshow(
        emergency_occ.sum(axis=1),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="inferno"
    )
    # Emoji removed here to prevent terminal font warnings
    ax.set_title("Emergency Vehicle Heatmap (All Lanes)", fontsize=14, fontweight="bold", color="#FF4B4B")
    ax.set_xlabel("Road Position", fontsize=12)
    ax.set_ylabel("Time Step", fontsize=12)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / "emergency_heatmap.png", dpi=300, bbox_inches='tight')
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
    
    # Upgrade: High-visibility neon colors common in telemetry dashboards
    ax.plot(steps, density, label="Density", color="#00E5FF", linewidth=2)
    ax.plot(steps, avg_speed, label="Average Speed", color="#76FF03", linewidth=2)
    ax.plot(steps, flow, label="Flow", color="#D500F9", linewidth=2)
    ax.plot(steps, queue, label="Queue Near Signals", color="#FF3D00", linewidth=2)
    ax.plot(steps, green, label="Green Signals", color="#FFEA00", linewidth=2, linestyle="--")
    
    ax.set_title("Traffic Flow & Control Metrics Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    
    # Add subtle grid lines for readability
    ax.grid(True, color="#333333", linestyle="-", alpha=0.7)
    
    # Position legend outside the main data area
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_benchmark_csv(rows: list[dict[str, float]], out_path: Path) -> None:
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)