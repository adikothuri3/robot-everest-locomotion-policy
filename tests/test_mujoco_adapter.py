"""MuJoCo adapter + generated Holosoma asset tests."""

import numpy as np
import pytest
import mujoco

from everest_locomotion import REPO_ROOT
from everest_locomotion.robots.manifest import load_manifest
from everest_locomotion.sim_adapters.mujoco_adapter import MujocoRobot


@pytest.fixture(scope="module")
def robot():
    return MujocoRobot(load_manifest("a3_ultra"))


def test_joint_mapping_roundtrip(robot):
    robot.reset_to_default()
    pose = robot.manifest.default_pose_vector()
    np.testing.assert_allclose(robot.joint_pos(), pose, atol=1e-9)


def test_actuator_mapping_is_named_consistently(robot):
    m = robot.model
    for name, aid in zip(robot.manifest.joint_order, robot.act_ids):
        jid = m.actuator_trnid[aid, 0]
        assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) == name


def test_total_mass(robot):
    assert abs(mujoco.mj_getTotalmass(robot.model) - robot.manifest.total_mass_kg) < 0.01


def test_pd_torque_respects_limits(robot):
    robot.reset_to_default()
    target = robot.default_pose + 10.0  # absurd target
    tau = robot.pd_torque(target)
    assert (np.abs(tau) <= robot.effort_limit + 1e-9).all()


def test_deterministic_stepping(robot):
    results = []
    for _ in range(2):
        robot.reset_to_default()
        for _ in range(100):
            robot.apply_pd(robot.default_pose)
            robot.step()
        results.append(robot.data.qpos.copy())
    np.testing.assert_array_equal(results[0], results[1])


def test_holosoma_asset_29dof():
    path = REPO_ROOT / "assets" / "a3_ultra" / "holosoma" / "a3_ultra_29dof.xml"
    if not path.exists():
        pytest.skip("generated asset missing; run scripts/convert/make_holosoma_asset.py")
    m = mujoco.MjModel.from_xml_path(str(path))
    hinges = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    assert len(hinges) == 29 and m.nu == 29
    manifest = load_manifest("a3_ultra")
    assert sorted(hinges) == sorted(manifest.control_joints)
    assert abs(mujoco.mj_getTotalmass(m) - manifest.total_mass_kg) < 0.01
    for b in ("left_foot_contact_point", "right_foot_contact_point"):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) >= 0
