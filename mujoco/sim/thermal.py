"""Hip motor temperatures estimated from the simulated joint torques.

An estimate, not a measurement: two thermal masses per motor, the winding and
the housing, heated by the copper loss of the current each torque needs and
cooled through the housing to the air:

    P = k tau^2                       (k = 1.5 R / Kt^2 at the output)
    C_w dT_w/dt = P - (T_w - T_h) / R_wh
    C_h dT_h/dt = (T_w - T_h) / R_wh - (T_h - T_air) / R_ha

T_w is what a motor driver reports (Unitree LowState motor temperature). The
parameters are rough guesses from the motors' size and peak torque (Go2 23.5,
As2 60, B2 200 Nm), not identified: the curves show which hip works harder and
how the load builds up, the absolute degrees need a fit to the real robot's
logged temperatures and torques. Iron and friction losses are left out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

T_AIR = 25.0   # deg C, the air and every motor at spawn


@dataclass(frozen=True)
class MotorThermalParams:
    loss_w_per_nm2: float      # k: copper loss per squared output torque
    winding_j_per_k: float     # C_w
    housing_j_per_k: float     # C_h
    winding_housing_k_per_w: float   # R_wh
    housing_air_k_per_w: float       # R_ha


# Go2-size joint motor: ~660 W of copper loss at its 23.5 Nm peak, the winding's
# time constant ~40 s, the housing's ~12 min, ~2.8 K per watt held.
_GO2 = MotorThermalParams(1.2, 50.0, 350.0, 0.8, 2.0)
# As2: a larger motor (60 Nm peak, ~1 kW at peak), twice the thermal mass.
_AS2 = MotorThermalParams(0.28, 100.0, 700.0, 0.5, 1.4)
# B2: 200 Nm peak (~2.5 kW at peak), about five times the Go2's mass.
_B2 = MotorThermalParams(0.0625, 250.0, 2000.0, 0.25, 0.7)
MOTORS = {"go2": _GO2, "go2w": _GO2, "as2": _AS2, "as2w": _AS2, "b2": _B2, "b2w": _B2}


class MotorThermal:
    """The two-mass model for n motors: step() once per control tick."""

    def __init__(self, params: MotorThermalParams, n: int, t_air: float = T_AIR) -> None:
        self.p = params
        self.t_air = float(t_air)
        self.winding = np.full(n, self.t_air)
        self.housing = np.full(n, self.t_air)

    def step(self, torque, dt: float) -> None:
        p = self.p
        heat = p.loss_w_per_nm2 * np.square(np.asarray(torque, dtype=np.float64))
        into_housing = (self.winding - self.housing) / p.winding_housing_k_per_w
        into_air = (self.housing - self.t_air) / p.housing_air_k_per_w
        self.winding += dt * (heat - into_housing) / p.winding_j_per_k
        self.housing += dt * (into_housing - into_air) / p.housing_j_per_k


class HipThermal:
    """The four hip motors of one plant (sim.plant.Bindings): their joint
    torques read once per control tick, legs in FL, FR, RL, RR order."""

    def __init__(self, data, b, robot: str, t_air: float = T_AIR) -> None:
        names = list(b.joint_names)
        hips = sorted(j for j in names if j.endswith("_hip_joint"))
        self.labels = tuple(j.split("_")[0] for j in hips)
        self._ids = np.asarray(b.actuator_ids)[[names.index(j) for j in hips]]
        self._data = data
        self.motors = MotorThermal(MOTORS[robot], len(hips), t_air)
        self.time = 0.0   # s since spawn

    def step(self, dt: float) -> None:
        self.motors.step(self._data.actuator_force[self._ids], dt)
        self.time += dt

    @property
    def temperatures(self) -> np.ndarray:
        """Winding temperatures (deg C), as a driver would report them."""
        return self.motors.winding.copy()


__all__ = ["MOTORS", "T_AIR", "HipThermal", "MotorThermal", "MotorThermalParams"]
