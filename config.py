
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SimulationConfig:
    road_length: int = 240
    lanes: int = 3
    steps: int = 180
    vmax: int = 5

    initial_density: float = 0.30
    emergency_density: float = 0.06
    spawn_rate: float = 0.18
    emergency_spawn_rate: float = 0.04
    slow_prob: float = 0.18
    emergency_slow_prob: float = 0.05
    seed: int = 42

    halo_width: int = 8
    signal_positions: tuple[int, ...] = (60, 120, 180)
    min_green: int = 8
    min_red: int = 5
    queue_threshold: int = 5
    emergency_threshold: int = 2
    signal_window: int = 10

    emergency_speed_boost: int = 2

    output_dir: Path = field(default_factory=lambda: Path("output"))

    def validate(self) -> None:
        if self.road_length <= 0:
            raise ValueError("road_length must be positive")
        if self.lanes < 2:
            raise ValueError("lanes must be at least 2")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.vmax <= 0:
            raise ValueError("vmax must be positive")
        if self.halo_width <= 0:
            raise ValueError("halo_width must be positive")
        if not (0.0 <= self.initial_density <= 1.0):
            raise ValueError("initial_density must be between 0 and 1")
        if not (0.0 <= self.emergency_density <= 1.0):
            raise ValueError("emergency_density must be between 0 and 1")
        if not (0.0 <= self.spawn_rate <= 1.0):
            raise ValueError("spawn_rate must be between 0 and 1")
        if not (0.0 <= self.emergency_spawn_rate <= 1.0):
            raise ValueError("emergency_spawn_rate must be between 0 and 1")
        if not (0.0 <= self.slow_prob <= 1.0):
            raise ValueError("slow_prob must be between 0 and 1")
        if not (0.0 <= self.emergency_slow_prob <= 1.0):
            raise ValueError("emergency_slow_prob must be between 0 and 1")
        if not self.signal_positions:
            raise ValueError("signal_positions must not be empty")
        if any(s <= 0 or s >= self.road_length for s in self.signal_positions):
            raise ValueError("signal_positions must be inside the road")
        if list(self.signal_positions) != sorted(self.signal_positions):
            raise ValueError("signal_positions must be sorted")
