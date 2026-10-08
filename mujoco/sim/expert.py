#!/usr/bin/env python3
"""Dual-expert selector: stairs veteran + flat omni expert.

The unified-training campaign verdict: the two skills erase each other at
the ~1000-iteration scale, so deployment carries BOTH experts plus a
selector. A switch resets the target's GRU memory to zero -- episode-fresh
state is the training distribution's own reset semantics, and hidden
states do not transfer across differently-weighted nets (their latent
spaces are unrelated). Observation histories (proprio/depth) are world
state and stay shared across switches, so the incoming expert is not
blind.

Modes: 'auto' (backward command -> flat expert, otherwise stairs; +/-0.05
hysteresis band, MIN_DWELL_TICKS control ticks between switches) or a
forced 'stairs' / 'flat'. cycle() (keyboard E / web button) rotates
auto -> stairs -> flat -> auto. The dwell counts select() CALLS (= control
ticks at 50 Hz by contract), NOT wall time: unpaced/fast-sim hosts run
wall time at a very different rate than the control loop (a bare tick
loop hit ~40x realtime in testing), so a wall-clock dwell is meaningless
there.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .policy import PieOnnxPolicy

VX_DEADBAND = 0.05        # |vx| below this keeps the current expert (hysteresis)
MIN_DWELL_TICKS = 50      # 1 sim-second at the 50 Hz control contract
MODES = ("auto", "stairs", "flat")


class DualExpertPolicy:
    """Drop-in PieOnnxPolicy stand-in: same __call__/reset surface."""

    def __init__(self, stairs_path: Path, flat_path: Path,
                 mode: str = "auto", provider: str = "cpu") -> None:
        if mode not in MODES:
            raise SystemExit(
                f"unknown expert mode {mode!r}; choose {list(MODES)}")
        self.mode = mode
        self.stairs = PieOnnxPolicy(Path(stairs_path), provider)
        self.flat = PieOnnxPolicy(Path(flat_path), provider)
        self.active_name = "flat" if mode == "flat" else "stairs"
        self._dwell = 0

    @property
    def active(self) -> PieOnnxPolicy:
        return self.flat if self.active_name == "flat" else self.stairs

    @property
    def providers(self) -> list[str]:
        return self.active.providers

    def select(self, vx: float) -> str | None:
        """Auto rule ('auto' mode only). Returns the new active name on a
        switch, else None; called once per control tick by the host loop.
        The dwell counts calls (= control ticks), immune to pacing."""
        if self.mode != "auto":
            return None
        if self._dwell > 0:
            self._dwell -= 1
            return None
        want = "flat" if vx < -VX_DEADBAND else "stairs"
        if want != self.active_name:
            return self._switch(want)
        return None

    def cycle(self) -> str:
        """Rotate auto -> stairs -> flat -> auto (keyboard E / web button).
        Returns the new mode; also reports it on the status line via
        active_name."""
        self.mode = {"auto": "stairs", "stairs": "flat",
                     "flat": "auto"}[self.mode]
        if self.mode in ("stairs", "flat"):
            self._switch(self.mode)
        return self.mode

    def _switch(self, name: str) -> str:
        if name == self.active_name:
            return name
        self.active_name = name
        self.active.reset()     # episode-fresh GRU; histories stay shared
        self._dwell = MIN_DWELL_TICKS
        return name

    def reset(self) -> None:
        self.stairs.reset()
        self.flat.reset()
        self._dwell = 0

    def __call__(self, proprio: np.ndarray, proprio_history: np.ndarray,
                 depth_history: np.ndarray) -> np.ndarray:
        return self.active(proprio, proprio_history, depth_history)


__all__ = ["DualExpertPolicy", "MODES", "VX_DEADBAND"]
