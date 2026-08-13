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
    """Runs an exported ONNX locomotion policy.

    obs_layout: ordered list of obs-dict keys concatenated into the network
    input; must match the training-time observation config (record it in the
    checkpoint metadata when exporting from Holosoma).
    """

    def __init__(
        self,
        onnx_path: str,
        obs_layout: list[str],
        default_pose_ctrl: np.ndarray,
        action_scale: float,
    ):
        import onnxruntime as ort  # lazy: only needed when evaluating ONNX

        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.obs_layout = obs_layout
        self.default = default_pose_ctrl.copy()
        self.action_scale = action_scale

    def reset(self) -> None:
        pass

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        x = np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in self.obs_layout])
        action = self.session.run(None, {self.input_name: x[None]})[0][0]
        return self.default + self.action_scale * action
