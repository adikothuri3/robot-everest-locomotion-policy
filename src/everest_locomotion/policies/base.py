"""Policy interface used by evaluation.

A policy maps an observation dict to target joint positions (canonical control
joint order from the manifest). The stability suite is policy-agnostic: it can
evaluate the trivial PD stand baseline today and Holosoma ONNX exports later.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    def reset(self) -> None: ...

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Return target joint positions for the manifest's control joints."""
        ...


class PDStandPolicy:
    """Holds the default pose. Baseline for E01 / pipeline validation."""

    def __init__(self, default_pose_ctrl: np.ndarray):
        self.target = default_pose_ctrl.copy()

    def reset(self) -> None:
        pass

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return self.target


class OnnxPolicy:
    """Retired 2026-08-14 — this was never a valid Holosoma consumer.

    It assumed a caller-supplied observation order and a phase-free observation.
    The real exported policy takes a 100-dim vector concatenated in
    **alphabetical term order** including a 2-dim per-leg gait clock, and the
    rollout harness this class plugged into supplies neither. Feeding it a real
    checkpoint produces a policy that falls over instantly, which is far too
    easy to misread as a bad checkpoint.

    Use `everest_locomotion.evaluation.sim2sim.HolosomaPolicy` instead; it reads
    the contract out of the ONNX metadata. Contract: `docs/onnx_policy_interface.md`.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "OnnxPolicy is retired — its observation layout does not match Holosoma "
            "exports. Use everest_locomotion.evaluation.sim2sim.HolosomaPolicy "
            "(scripts/eval/sim2sim_suite.py)."
        )
