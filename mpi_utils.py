
from __future__ import annotations

from typing import Any

import numpy as np
from mpi4py import MPI

EMPTY = -1


def build_counts_displs(n: int, size: int) -> tuple[list[int], list[int]]:
    base = n // size
    rem = n % size
    counts: list[int] = []
    displs: list[int] = []
    cursor = 0
    for r in range(size):
        length = base + (1 if r < rem else 0)
        counts.append(length)
        displs.append(cursor)
        cursor += length
    return counts, displs


def exchange_boundary_ghosts(
    comm: MPI.Comm,
    local_speeds: np.ndarray,
    local_emergency: np.ndarray,
    halo_width: int,
) -> dict[str, Any]:
    """
    Lightweight ghost-cell exchange using an allgather of boundary strips.
    Each rank receives a small left and right halo for every lane.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    lanes, local_n = local_speeds.shape

    width = min(halo_width, local_n)

    def pad_strip(arr: np.ndarray) -> np.ndarray:
        if width == halo_width:
            return arr.copy()
        out = np.full((lanes, halo_width), EMPTY, dtype=arr.dtype)
        out[:, :width] = arr[:, :width]
        return out

    left_strip = pad_strip(local_speeds[:, :width])
    right_strip = pad_strip(local_speeds[:, -width:])
    left_em = pad_strip(local_emergency[:, :width])
    right_em = pad_strip(local_emergency[:, -width:])

    gathered = comm.allgather(
        {
            "rank": rank,
            "left_speed": left_strip,
            "right_speed": right_strip,
            "left_em": left_em,
            "right_em": right_em,
        }
    )

    ghost_left_speed = np.full((lanes, halo_width), EMPTY, dtype=np.int32)
    ghost_right_speed = np.full((lanes, halo_width), EMPTY, dtype=np.int32)
    ghost_left_em = np.zeros((lanes, halo_width), dtype=np.int32)
    ghost_right_em = np.zeros((lanes, halo_width), dtype=np.int32)

    if rank > 0:
        ghost_left_speed = gathered[rank - 1]["right_speed"].copy()
        ghost_left_em = gathered[rank - 1]["right_em"].copy()
    if rank < size - 1:
        ghost_right_speed = gathered[rank + 1]["left_speed"].copy()
        ghost_right_em = gathered[rank + 1]["left_em"].copy()

    return {
        "left_speed": ghost_left_speed,
        "right_speed": ghost_right_speed,
        "left_em": ghost_left_em,
        "right_em": ghost_right_em,
    }


def gather_global_state(
    comm: MPI.Comm,
    local_speeds: np.ndarray,
    local_emergency: np.ndarray,
    counts: list[int],
    displs: list[int],
    road_length: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    rank = comm.Get_rank()
    lanes, local_n = local_speeds.shape
    flat_speed = local_speeds.reshape(-1)
    flat_em = local_emergency.reshape(-1)

    cell_counts = [c * lanes for c in counts]
    cell_displs = [d * lanes for d in displs]

    if rank == 0:
        global_speed = np.empty(lanes * road_length, dtype=np.int32)
        global_em = np.empty(lanes * road_length, dtype=np.int32)
    else:
        global_speed = None
        global_em = None

    comm.Gatherv(flat_speed, [global_speed, cell_counts, cell_displs, MPI.INT], root=0)
    comm.Gatherv(flat_em, [global_em, cell_counts, cell_displs, MPI.INT], root=0)

    if rank == 0:
        return global_speed.reshape(lanes, road_length), global_em.reshape(lanes, road_length)
    return None, None


def transfer_cross_boundary(
    comm: MPI.Comm,
    outgoing_right: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """
    Point-to-point transfer of vehicles leaving a segment to the right neighbor.
    Each tuple is: (global_pos, lane, speed, emergency_flag)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    dest = rank + 1 if rank < size - 1 else MPI.PROC_NULL
    source = rank - 1 if rank > 0 else MPI.PROC_NULL
    incoming = comm.sendrecv(outgoing_right, dest=dest, source=source, sendtag=91, recvtag=91)
    return incoming if incoming is not None else []


def nonempty_count(speeds: np.ndarray) -> int:
    return int(np.count_nonzero(speeds != EMPTY))


def lane_occupancy_in_window(
    local_speeds: np.ndarray,
    local_emergency: np.ndarray,
    global_start: int,
    signal_pos: int,
    window: int,
) -> tuple[int, int, int]:
    left = max(0, signal_pos - window)
    right = signal_pos + window
    occupied = 0
    stopped = 0
    emergency = 0
    lanes, local_n = local_speeds.shape
    for lane in range(lanes):
        for idx in range(local_n):
            g = global_start + idx
            if left <= g <= right and local_speeds[lane, idx] != EMPTY:
                occupied += 1
                if local_speeds[lane, idx] == 0:
                    stopped += 1
                if local_emergency[lane, idx] == 1:
                    emergency += 1
    return occupied, stopped, emergency
