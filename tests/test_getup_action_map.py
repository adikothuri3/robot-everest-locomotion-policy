"""A get-up policy must be graded through the action map it was trained through.

`A3UltraGetupManager._pre_physics_step` scales every action by the authority
curriculum (HoST's beta, 2.0 -> 1.0) and the wrists by the manifest's soft
freeze, *then* hands the result to the action manager — so that scaled action is
also what holosoma's `actions` observation term reports back. A harness that
replays the raw ONNX output at authority 1.0 commands half the joint travel the
policy learned to produce and feeds it an action history it never saw. These
tests pin both halves of that contract, plus the neutrality that keeps the
recorded 68/68 locomotion gate valid.
"""

import sys

import numpy as np
import pytest

from everest_locomotion import REPO_ROOT
from everest_locomotion.evaluation.sim2sim import A3Sim, HolosomaPolicy, ObsLayout
from everest_locomotion.robots.manifest import load_manifest

GETUP = REPO_ROOT / "checkpoints" / "v1_getup_28k_plateau" / "model_27999.onnx"

pytestmark = pytest.mark.skipif(
    not GETUP.exists(), reason="shipped get-up policy missing (see README)"
)


@pytest.fixture(scope="module")
def sim():
    policy = HolosomaPolicy(GETUP)
    return A3Sim(policy, load_manifest())


# -- the action map ---------------------------------------------------------


def test_apply_action_is_identity_at_defaults(sim):
    """Neutrality: every locomotion grade on record was taken through this path."""
    assert sim.action_authority == 1.0 and sim.wrist_action_factor == 1.0
    a = np.linspace(-1.0, 1.0, sim.policy.n_dof)
    assert np.array_equal(sim.apply_action(a), a)


def test_apply_action_scales_authority_and_freezes_wrists(sim):
    sim.action_authority, sim.wrist_action_factor = 2.0, 0.2
    try:
        a = np.ones(sim.policy.n_dof)
        applied = sim.apply_action(a)
        wrist = sim._wrist_idx
        assert wrist.size, "A3 Ultra has wrists; the index must not be empty"
        other = np.setdiff1d(np.arange(sim.policy.n_dof), wrist)
        assert np.allclose(applied[other], 2.0)
        assert np.allclose(applied[wrist], 2.0 * 0.2)
        assert np.array_equal(a, np.ones(sim.policy.n_dof)), "must not mutate its input"
    finally:
        sim.action_authority, sim.wrist_action_factor = 1.0, 1.0


def test_wrist_factor_is_derivable_from_the_manifest(sim):
    """The training constant WRIST_ACTION_FACTOR is manifest/action_scale — the
    harness derives it rather than duplicating the number."""
    manifest = load_manifest()
    wrist = sim._wrist_idx
    factor = manifest.raw["getup"]["wrist_action_scale"] / np.mean(
        sim.policy.action_scale[wrist]
    )
    assert factor == pytest.approx(0.2)


# -- the observation term ---------------------------------------------------


def test_action_authority_observation_reports_the_applied_authority(sim):
    sim.action_authority = 2.0
    try:
        frame = sim._term_frame(
            "action_authority", np.zeros(sim.policy.n_dof), 0, np.zeros(3), 0.0,
            np.zeros(len(sim.arm_idx)),
        )
        assert np.asarray(frame).shape == (1,)
        assert float(np.asarray(frame)[0]) == 2.0
    finally:
        sim.action_authority = 1.0


def test_harness_can_reproduce_every_v3_getup_observation_term(sim):
    """The v3 get-up actor adds `action_authority` and a 5-frame history.

    Mirrors `getup_observation` in the training extension. If a term is added
    there without a producer here, grading dies on a KeyError mid-rollout —
    which is exactly how the first v3 checkpoint would have been unreadable.
    """
    cfg = {"observation": {"groups": {"actor_obs": {
        "history_length": 5,
        "terms": {
            "base_ang_vel": {"scale": 0.25},
            "projected_gravity": {"scale": 1.0},
            "dof_pos": {"scale": 1.0},
            "dof_vel": {"scale": 0.05},
            "actions": {"scale": 1.0},
            "action_authority": {"scale": 1.0},
        },
    }}}}
    layout = ObsLayout.from_experiment_config(cfg, 29)
    # 29+3+29+29+3+1 = 94 per frame, five frames deep
    assert layout.total_dim == 94 * 5
    # '_' sorts before 's', so the beta channel leads the vector
    assert [t.name for t in layout.terms][0] == "action_authority"

    for term in layout.terms:
        frame = sim._term_frame(
            term.name, np.zeros(sim.policy.n_dof), 0, np.zeros(3), 0.0,
            np.zeros(len(sim.arm_idx)),
        )
        assert np.asarray(frame).shape == (term.dim,), term.name


# -- the success gates must be reachable ------------------------------------
#
# A gate no policy can satisfy reports 0 forever and reads as "the policy is
# broken". That is what pinned `getup_success_rate` at 0 for two cloud runs, so
# every gate this repo scores against gets a reachability test.


@pytest.fixture(scope="module")
def rollover_term(sim):
    import mujoco  # noqa: F401  (imported for parity with the grader)
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))
    import getup_video

    return getup_video.RolloverTerminal(load_manifest(), sim.policy), getup_video


def _settle(sim, qpos, steps=50):
    import mujoco

    sim.reset_fallen(qpos) if hasattr(sim, "reset_fallen") else None
    for _ in range(steps):
        mujoco.mj_step(sim.model, sim.data)


def test_rollover_gate_accepts_its_own_target_state(rollover_term):
    """Settled supine poses ARE the rollover terminal state; the gate must pass them."""
    term, gv = rollover_term
    sim = gv.GetupSim(HolosomaPolicy(GETUP), load_manifest())
    bank = np.load(REPO_ROOT / "assets" / "a3_ultra" / "getup" / "fallen_poses_supine.npy")
    rng = np.random.default_rng(0)
    ok = 0
    for idx in rng.choice(len(bank), 20, replace=False):
        _settle(sim, bank[idx])
        ok += term.check(sim)[0]
    # Not 20/20: a few bank poses settle twisted (torso off-axis), which the
    # base-AND-torso criterion is right to reject.
    assert ok >= 15, f"only {ok}/20 settled supine poses satisfy the rollover gate"


def test_rollover_gate_rejects_the_state_it_rolls_out_of(rollover_term):
    term, gv = rollover_term
    sim = gv.GetupSim(HolosomaPolicy(GETUP), load_manifest())
    bank = np.load(REPO_ROOT / "assets" / "a3_ultra" / "getup" / "fallen_poses_prone.npy")
    rng = np.random.default_rng(1)
    rejected = 0
    for idx in rng.choice(len(bank), 20, replace=False):
        _settle(sim, bank[idx])
        rejected += not term.check(sim)[0]
    assert rejected == 20, f"{20 - rejected}/20 prone poses wrongly pass the rollover gate"


def test_the_two_gates_are_not_interchangeable(rollover_term):
    """Grading one task with the other's gate is the trap this flag exists to avoid."""
    term, gv = rollover_term
    assert gv.TERMINALS["getup"] is gv.Terminal
    assert gv.TERMINALS["rollover"] is gv.RolloverTerminal
    # rollover wants the pelvis LOW, get-up wants it HIGH — opposite directions
    getup = gv.Terminal(load_manifest(), HolosomaPolicy(GETUP))
    assert term.height_max < getup.height - getup.height_tol
