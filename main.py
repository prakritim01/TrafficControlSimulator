
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from mpi4py import MPI

from config import SimulationConfig
from mpi_utils import (
    build_counts_displs,
    exchange_boundary_ghosts,
    gather_global_state,
    lane_occupancy_in_window,
    nonempty_count,
    transfer_cross_boundary,
)
from traffic import (
    SmartSignal,
    inject_vehicles_at_entry,
    initialize_local_state,
    step_local_traffic,
)
from visualize import save_metrics_csv, save_metrics_plot, save_space_time_diagrams


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cfg = SimulationConfig()
    if rank == 0:
        cfg.validate()
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = comm.bcast(cfg, root=0)

    if cfg.road_length < size * (cfg.vmax + 2):
        if rank == 0:
            raise ValueError("Road is too short for the chosen number of MPI ranks.")
        comm.Abort(1)

    counts, displs = build_counts_displs(cfg.road_length, size)
    local_n = counts[rank]
    global_start = displs[rank]

    rng = np.random.default_rng(cfg.seed + rank * 1000 + 13)
    local_speeds, local_emergency = initialize_local_state(
        cfg.lanes,
        local_n,
        cfg.initial_density,
        cfg.emergency_density,
        cfg.vmax,
        cfg.seed,
        rank,
    )

    signals = [SmartSignal(position=p, green=True) for p in cfg.signal_positions]
    signal_states = {s.position: True for s in signals}

    history_speed: list[np.ndarray] = []
    history_emergency: list[np.ndarray] = []
    metrics_rows: list[dict[str, float]] = []

    t0 = time.perf_counter()

    for step in range(cfg.steps):
        ghost = exchange_boundary_ghosts(comm, local_speeds, local_emergency, cfg.halo_width)

        local_queue = np.zeros(len(signals), dtype=np.int32)
        local_em = np.zeros(len(signals), dtype=np.int32)

        for i, sig in enumerate(signals):
            _, stopped, em = lane_occupancy_in_window(
                local_speeds,
                local_emergency,
                global_start,
                sig.position,
                cfg.signal_window,
            )
            local_queue[i] = stopped
            local_em[i] = em

        global_queue = comm.allreduce(local_queue, op=MPI.SUM)
        global_em = comm.allreduce(local_em, op=MPI.SUM)
        _ = comm.allreduce(nonempty_count(local_speeds), op=MPI.SUM)

        if rank == 0:
            for i, sig in enumerate(signals):
                density = float(global_queue[i]) / float((2 * cfg.signal_window + 1) * cfg.lanes)
                sig.green = sig.update(
                    queue_count=int(global_queue[i]),
                    density=density,
                    emergency_count=int(global_em[i]),
                    min_green=cfg.min_green,
                    min_red=cfg.min_red,
                    queue_threshold=cfg.queue_threshold,
                    emergency_threshold=cfg.emergency_threshold,
                )
            signal_states = {sig.position: sig.green for sig in signals}
        else:
            signal_states = None  # type: ignore[assignment]

        signal_states = comm.bcast(signal_states, root=0)

        next_speeds, next_emergency, outgoing_right, step_stats = step_local_traffic(
            local_speeds=local_speeds,
            local_emergency=local_emergency,
            ghost=ghost,
            global_start=global_start,
            vmax=cfg.vmax,
            emergency_speed_boost=cfg.emergency_speed_boost,
            slow_prob=cfg.slow_prob,
            emergency_slow_prob=cfg.emergency_slow_prob,
            signal_states=signal_states,
            signal_positions=cfg.signal_positions,
            rng=rng,
        )

        incoming_left = transfer_cross_boundary(comm, outgoing_right)

        for global_pos, lane, speed, em_flag in incoming_left:
            local_idx = global_pos - global_start
            if 0 <= local_idx < local_n and next_speeds[lane, local_idx] == -1:
                next_speeds[lane, local_idx] = speed
                next_emergency[lane, local_idx] = em_flag

        if rank == 0:
            inject_vehicles_at_entry(
                next_speeds,
                next_emergency,
                cfg.spawn_rate,
                cfg.emergency_spawn_rate,
                cfg.vmax,
                rng,
            )

        local_speeds = next_speeds
        local_emergency = next_emergency

        global_speed, global_em = gather_global_state(
            comm, local_speeds, local_emergency, counts, displs, cfg.road_length
        )

        if rank == 0 and global_speed is not None and global_em is not None:
            history_speed.append(global_speed.copy())
            history_emergency.append(global_em.copy())

            occupied = int(np.count_nonzero(global_speed != -1))
            total_speed = int(np.sum(global_speed[global_speed != -1])) if occupied else 0
            avg_speed = float(total_speed / occupied) if occupied else 0.0
            flow = int(step_stats["moved"])
            green_signals = int(sum(1 for s in signals if s.green))
            global_queue_sum = int(np.sum(global_queue))

            metrics_rows.append(
                {
                    "step": float(step),
                    "density": float(occupied / (cfg.road_length * cfg.lanes)),
                    "avg_speed": avg_speed,
                    "flow": float(flow),
                    "global_queue": float(global_queue_sum),
                    "green_signals": float(green_signals),
                    "total_vehicles": float(occupied),
                    "emergency_vehicles": float(np.count_nonzero(global_em == 1)),
                }
            )

        comm.Barrier()

    elapsed = time.perf_counter() - t0

    if rank == 0:
        out_dir = Path(cfg.output_dir)
        save_space_time_diagrams(history_speed, history_emergency, out_dir)
        save_metrics_plot(metrics_rows, out_dir / "traffic_metrics.png")
        save_metrics_csv(metrics_rows, out_dir / "metrics.csv")

        summary = out_dir / "summary.txt"
        summary.write_text(
            "\n".join(
                [
                    "Distributed Real-Time Smart City Traffic Flow & Crowd Optimizer",
                    f"MPI ranks: {size}",
                    f"Road length: {cfg.road_length}",
                    f"Lanes: {cfg.lanes}",
                    f"Steps: {cfg.steps}",
                    f"Elapsed time: {elapsed:.4f} s",
                ]
            ),
            encoding="utf-8",
        )

        print("Simulation complete.")
        print(f"Elapsed time: {elapsed:.4f} s")
        print(f"Outputs saved in: {out_dir.resolve()}")
        print("Files:")
        print(" - summary.txt")
        print(" - traffic_metrics.png")
        print(" - metrics.csv")
        for lane in range(cfg.lanes):
            print(f" - space_time_lane_{lane + 1}.png")
        print(" - emergency_heatmap.png")


if __name__ == "__main__":
    main()
