"""Rayleigh-Taylor nondimensionalisation (Dalziel et al. 1999).

    tau = t sqrt(A g / H),  box size / H,  velocity / sqrt(A g H)

Merges FractalParameterAI/core/rt_physics.py (2D) and fractal3d/rt_physics.py
(3D); W (depth) is None in 2D.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RTPhysics:
    A: float
    g: float
    H: float
    L: float
    W: Optional[float] = None
    tau_factor: float = field(init=False, repr=False)
    velocity_scale: float = field(init=False, repr=False)

    def __post_init__(self):
        if not 0 <= self.A <= 1:
            raise ValueError(f"Atwood number must be in [0, 1], got {self.A}")
        if self.A == 0:
            # no buoyancy: keep dimensional time
            self.tau_factor = 1.0
            self.velocity_scale = 1.0
        else:
            self.tau_factor = np.sqrt(self.A * self.g / self.H)
            self.velocity_scale = np.sqrt(self.A * self.g * self.H)

    def nondim_time(self, t):
        return np.asarray(t) * self.tau_factor

    def nondim_box_size(self, delta):
        return np.asarray(delta) / self.H

    def nondim_velocity(self, u):
        return np.asarray(u) / self.velocity_scale

    def summary(self):
        return (f"RT physics: A={self.A:.6f}, g={self.g}, H={self.H}, L={self.L}"
                + (f", W={self.W}" if self.W else "")
                + f"; tau = t * {self.tau_factor:.6f}")
