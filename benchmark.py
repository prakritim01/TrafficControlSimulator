
from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    out = Path("output")
    out.mkdir(exist_ok=True)
    rows = [
        {"mpi_ranks": 1, "note": "run mpiexec -n 1 python main.py"},
        {"mpi_ranks": 2, "note": "run mpiexec -n 2 python main.py"},
        {"mpi_ranks": 4, "note": "run mpiexec -n 4 python main.py"},
    ]
    with (out / "benchmark_plan.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mpi_ranks", "note"])
        writer.writeheader()
        writer.writerows(rows)
    print("Benchmark plan saved to output/benchmark_plan.csv")


if __name__ == "__main__":
    main()
