"""Manifest consistency tests: the repo must fail loudly on joint-order drift."""

import numpy as np
import pytest

from everest_locomotion.robots.manifest import load_manifest


@pytest.fixture(scope="module")
def manifest():
    return load_manifest("a3_ultra")


def test_counts(manifest):
    assert manifest.num_joints == 31
    assert manifest.num_actions == 29
    assert set(manifest.frozen_joints) == {"head_yaw_joint", "head_pitch_joint"}


def test_joint_order_is_canonical(manifest):
    # legs (L,R) -> waist -> arms (L,R) -> head; spot-check anchors
    assert manifest.joint_order[0] == "left_hip_pitch_joint"
    assert manifest.joint_order[6] == "right_hip_pitch_joint"
    assert manifest.joint_order[12] == "waist_yaw_joint"
    assert manifest.joint_order[15] == "left_shoulder_pitch_joint"
    assert manifest.joint_order[22] == "right_shoulder_pitch_joint"
    assert manifest.joint_order[29] == "head_yaw_joint"
    assert len(set(manifest.joint_order)) == 31


def test_limits_shape_and_sanity(manifest):
    lim = manifest.joint_limits()
    assert lim.shape == (31, 2)
    assert (lim[:, 0] < lim[:, 1]).all()
    eff = manifest.effort_limits()
    assert eff.min() > 0 and eff.max() <= 320.0
    vel = manifest.velocity_limits()
    assert (vel > 0).all()


def test_default_pose_within_limits(manifest):
    pose = manifest.default_pose_vector()
    lim = manifest.joint_limits()
    assert (pose >= lim[:, 0]).all() and (pose <= lim[:, 1]).all()


def test_pd_gains_cover_all_joints(manifest):
    kp, kd = manifest.pd_gains()
    assert kp.shape == (31,) and (kp > 0).all() and (kd > 0).all()


def test_holosoma_preset_matches_manifest():
    """The Holosoma extension's literal arrays must match the manifest."""
    import ast
    from pathlib import Path

    from everest_locomotion import REPO_ROOT

    src = (REPO_ROOT / "src/everest_locomotion/holosoma_ext/a3_ultra_presets.py").read_text()
    tree = ast.parse(src)
    consts = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("_LOWER", "_UPPER", "_VEL", "_EFFORT"):
                consts[name] = ast.literal_eval(node.value)

    manifest = load_manifest("a3_ultra")
    ctrl = manifest.control_joints
    lim = manifest.joint_limits(ctrl)
    np.testing.assert_allclose(consts["_LOWER"], lim[:, 0], atol=1e-3)
    np.testing.assert_allclose(consts["_UPPER"], lim[:, 1], atol=1e-3)
    np.testing.assert_allclose(consts["_EFFORT"], manifest.effort_limits(ctrl), atol=1e-2)
    np.testing.assert_allclose(consts["_VEL"], manifest.velocity_limits(ctrl), atol=1e-2)
