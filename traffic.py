
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpi_utils import EMPTY


@dataclass
class SmartSignal:
    position: int
    green: bool = True
    phase_timer: int = 0

    def update(
        self,
        queue_count: int,
        density: float,
        emergency_count: int,
        min_green: int,
        min_red: int,
        queue_threshold: int,
        emergency_threshold: int,
    ) -> bool:
        if self.green:
            self.phase_timer += 1
            if self.phase_timer < min_green:
                return True
            if queue_count >= queue_threshold or emergency_count >= emergency_threshold or density >= 0.40:
                return True
            self.green = False
            self.phase_timer = 0
            return False
        self.phase_timer += 1
        if self.phase_timer < min_red:
            return False
        if queue_count >= queue_threshold or emergency_count >= emergency_threshold or density >= 0.30:
            self.green = True
            self.phase_timer = 0
            return True
        return False


def initialize_local_state(
    lanes: int,
    local_n: int,
    density: float,
    emergency_density: float,
    vmax: int,
    seed: int,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + rank * 997)
    speeds = np.full((lanes, local_n), EMPTY, dtype=np.int32)
    emergency = np.zeros((lanes, local_n), dtype=np.int32)

    mask = rng.random((lanes, local_n)) < density
    for lane in range(lanes):
        for i in range(local_n):
            if mask[lane, i]:
                speeds[lane, i] = int(rng.integers(0, vmax + 1))
                emergency[lane, i] = int(rng.random() < emergency_density)
                if emergency[lane, i]:
                    speeds[lane, i] = int(rng.integers(max(1, vmax - 1), vmax + 1))
    return speeds, emergency


def choose_next_signal(current_pos: int, signal_positions: tuple[int, ...]) -> int | None:
    for pos in signal_positions:
        if pos > current_pos:
            return pos
    return None


def build_extended(
    speeds: np.ndarray,
    emergency: np.ndarray,
    right_ghost_speed: np.ndarray,
    right_ghost_emergency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([speeds, right_ghost_speed], axis=1),
        np.concatenate([emergency, right_ghost_emergency], axis=1),
    )


def gap_ahead(
    ext_speed: np.ndarray,
    lane: int,
    idx: int,
    max_lookahead: int,
) -> int:
    gap = 0
    for step in range(1, max_lookahead + 1):
        if idx + step >= ext_speed.shape[1]:
            break
        if ext_speed[lane, idx + step] != EMPTY:
            break
        gap += 1
    return gap


def place_vehicle(
    next_speed: np.ndarray,
    next_emergency: np.ndarray,
    lane: int,
    pos: int,
    speed: int,
    emergency_flag: int,
) -> None:
    while pos >= 0 and next_speed[lane, pos] != EMPTY:
        pos -= 1
    if pos >= 0:
        next_speed[lane, pos] = speed
        next_emergency[lane, pos] = emergency_flag


def step_local_traffic(
    local_speeds: np.ndarray,
    local_emergency: np.ndarray,
    ghost: dict[str, np.ndarray],
    global_start: int,
    vmax: int,
    emergency_speed_boost: int,
    slow_prob: float,
    emergency_slow_prob: float,
    signal_states: dict[int, bool],
    signal_positions: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]], dict[str, int]]:
    lanes, local_n = local_speeds.shape
    next_speeds = np.full_like(local_speeds, EMPTY)
    next_emergency = np.zeros_like(local_emergency)

    ext_speed, ext_em = build_extended(local_speeds, local_emergency, ghost["right_speed"], ghost["right_em"])

    outgoing_right: list[tuple[int, int, int, int]] = []
    moved = 0
    stopped = 0
    emergency_moved = 0

    for idx in range(local_n - 1, -1, -1):
        global_pos = global_start + idx
        next_signal = choose_next_signal(global_pos, signal_positions)

        for lane in range(lanes):
            if local_speeds[lane, idx] == EMPTY:
                continue

            is_emergency = int(local_emergency[lane, idx])
            base_vmax = vmax + (emergency_speed_boost if is_emergency else 0)
            speed = int(local_speeds[lane, idx])
            speed = min(speed + 1, base_vmax)

            current_gap = gap_ahead(ext_speed, lane, idx, base_vmax)

            target_lane = lane
            best_gap = current_gap

            candidates = []
            if lane > 0:
                candidates.append(lane - 1)
            if lane < lanes - 1:
                candidates.append(lane + 1)

            for cand in candidates:
                safe_here = ext_speed[cand, idx] == EMPTY
                safe_next = (idx + 1 < ext_speed.shape[1]) and (ext_speed[cand, idx + 1] == EMPTY)
                if not safe_here or not safe_next:
                    continue
                cand_gap = gap_ahead(ext_speed, cand, idx, base_vmax)
                if cand_gap > best_gap + 1:
                    best_gap = cand_gap
                    target_lane = cand

            speed = min(speed, best_gap)

            if next_signal is not None:
                green = signal_states.get(next_signal, True)
                if not green and not is_emergency and global_pos < next_signal:
                    speed = min(speed, next_signal - global_pos - 1)

            if speed < 0:
                speed = 0

            slow_probability = emergency_slow_prob if is_emergency else slow_prob
            if speed > 0 and rng.random() < slow_probability:
                speed -= 1

            new_global = global_pos + speed
            if new_global >= global_start + local_n:
                outgoing_right.append((new_global, target_lane, speed, is_emergency))
                moved += int(speed > 0)
                emergency_moved += is_emergency
                continue

            place_vehicle(next_speeds, next_emergency, target_lane, new_global - global_start, speed, is_emergency)
            moved += int(speed > 0)
            emergency_moved += is_emergency
            if speed == 0:
                stopped += 1

    return next_speeds, next_emergency, outgoing_right, {
        "moved": moved,
        "stopped": stopped,
        "emergency_moved": emergency_moved,
    }


def inject_vehicles_at_entry(
    local_speeds: np.ndarray,
    local_emergency: np.ndarray,
    spawn_rate: float,
    emergency_spawn_rate: float,
    vmax: int,
    rng: np.random.Generator,
) -> None:
    lanes, local_n = local_speeds.shape
    for lane in range(lanes):
        if local_speeds[lane, 0] == EMPTY and rng.random() < spawn_rate:
            local_speeds[lane, 0] = int(rng.integers(0, vmax + 1))
            local_emergency[lane, 0] = int(rng.random() < emergency_spawn_rate)
            if local_emergency[lane, 0] == 1:
                local_speeds[lane, 0] = max(local_speeds[lane, 0], max(1, vmax - 1))
