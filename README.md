# Distributed Real-Time Smart City Traffic Flow & Crowd Optimizer

A high-performance, MPI-based traffic simulation designed to model complex urban vehicle dynamics. This project utilizes domain decomposition to simulate multi-lane traffic flow, adaptive signal control, and emergency vehicle prioritization across distributed computing nodes.

## Key Features
* **Distributed Architecture:** Implements a distributed memory model using `mpi4py` for scalable traffic simulation across multiple computing ranks.
* **Domain Decomposition:** The road is segmented into local sections, where each MPI rank manages its own segment and coordinates with neighbors.
* **Adaptive Traffic Control:** Smart signals adjust phases based on real-time queue pressure, vehicle density, and emergency vehicle detection.
* **Emergency Priority Handling:** Special logic for emergency vehicles, including speed boosts and prioritized signal switching to minimize response times.
* **Ghost-Cell Boundary Exchange:** Lightweight halo-cell synchronization ensures seamless vehicle transitions between distributed MPI ranks.
* **Visual Analytics:** Automated generation of space-time diagrams for each lane and emergency vehicle heatmaps for performance analysis.

## Technical Stack
* **Language:** Python 3.x
* **Parallel Computing:** MPI (via `mpi4py`)
* **State Management & Logic:** NumPy
* **Visualization:** Matplotlib

## Setup & Installation
Ensure you have an MPI implementation (like MS-MPI for Windows) installed, then install the required Python libraries:
```bash
pip install mpi4py numpy matplotlib
```

## 🏃 Execution
To run the simulation with 4 distributed processes:
```bash
mpiexec -n 4 python main.py
```

## Project Structure & Outputs
The simulation populates the `output/` directory with the following assets:
* **`summary.txt`**: Logs the execution time, number of MPI ranks, and road configuration.
* **`metrics.csv`**: Time-step data for density, average speed, flow, and queue lengths.
* **`traffic_metrics.png`**: Visual trend analysis of global traffic health metrics.
* **`space_time_lane_X.png`**: Diagrams showing vehicle progression over time for each individual lane.
* **`emergency_heatmap.png`**: A specialized heatmap focusing on emergency vehicle movement patterns.

##  Highlights
* **Synchronization Logic:** Uses `sendrecv` and `allgather` to manage vehicle transfers and ghost-cell updates across rank boundaries.
* **Adaptive Algorithms:** The `SmartSignal.update()` logic triggers green light extensions based on a `queue_threshold` of 5 or density above 0.40.
* **Cellular Automata Principles:** Implemented vehicle movement rules including gap-ahead calculations and lane-changing logic.
* **Scalability:** Performance benchmarking capabilities via `benchmark.py` across different MPI rank counts to measure computational efficiency.