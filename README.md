# Distributed Real-Time Smart City Traffic Flow & Crowd Optimizer

A high-performance, MPI-based traffic simulation designed to model complex urban vehicle dynamics. This project utilizes domain decomposition to simulate multi-lane traffic flow, adaptive signal control, and emergency vehicle prioritization across distributed computing nodes.

##  Telemetry & Visual Analytics
<img width="4164" height="1764" alt="image" src="https://github.com/user-attachments/assets/2ce2241e-4e2f-4f58-9bf6-85e6f6913211" />
 <img width="4164" height="1764" alt="image" src="https://github.com/user-attachments/assets/3f717753-00b1-4c66-834c-2722adc1302d" />
<img width="4164" height="1764" alt="image" src="https://github.com/user-attachments/assets/b4896bb4-cc4f-4d27-b1d1-e4c086bdf932" />
<img width="4164" height="1764" alt="image" src="https://github.com/user-attachments/assets/c71007ba-44f7-4374-889f-f51242887040" />

<img width="4164" height="1764" alt="image" src="https://github.com/user-attachments/assets/c0556ddf-0738-4df2-9a04-1f87775f5746" />

<img width="4164" height="1764" alt="image" src="https://github.com/user-attachments/assets/4d12d434-9d85-435d-8874-8cd7ff1e327f" />


##  Why Distributed Computing (HPC)?
Unlike standard Python simulations that hit a CPU bottleneck, this architecture uses MPI domain decomposition. By splitting the road into segments and using ghost-cell boundary exchanges, the workload scales across computing clusters. This mirrors the High-Performance Computing (HPC) techniques used in enterprise fluid dynamics (CFD) and motorsport grid simulations.

## 🔑 Key Features
* **Distributed Architecture:** Implements a distributed memory model using `mpi4py` for scalable traffic simulation across multiple computing ranks.
* **Domain Decomposition:** The road is segmented into local sections, where each MPI rank manages its own segment and coordinates with neighbors.
* **Adaptive Traffic Control:** Smart signals adjust phases based on real-time queue pressure, vehicle density, and emergency vehicle detection.
* **Emergency Priority Handling:** Special logic for emergency vehicles, including speed boosts and prioritized signal switching to minimize response times.
* **Ghost-Cell Boundary Exchange:** Lightweight halo-cell synchronization ensures seamless vehicle transitions between distributed MPI ranks.
* **Dynamic Configuration:** Simulation parameters (lanes, density, steps) are injected at runtime via CLI arguments for flexible experimental setups.

##  Technical Stack
* **Language:** Python 3.11+
* **Parallel Computing:** MPI (via `mpi4py`)
* **State Management & Logic:** NumPy
* **Visualization:** Matplotlib (Dark Mode Telemetry Styling)

##  Setup & Installation
Ensure you have an MPI implementation (like MS-MPI for Windows or OpenMPI for Linux/macOS) installed, then install the required Python libraries:
```bash
pip install -r requirements.txt
# OR manually: pip install mpi4py numpy matplotlib
