"""The smoothness penalties must not charge the policy for commanded arm motion.

`A3UltraLocoV2Manager._pre_physics_step` writes the upper-body skill's arm targets
*into* the action vector before the action manager stores it, so
`action_manager.action` contains motion the policy did not choose. Differencing
that vector — which is what `penalty_action_rate` and `penalty_action_jerk` do —
taxes the policy for a disturbance it cannot reduce.

That is not hypothetical: it is what broke S2/S3/S4 on the first ladder. Raw
`penalty_action_rate` went 132 (S1) -> 571 (S2) -> 720 (S4) at weight -2.0, which
alone exceeded the whole `alive` budget, drove net per-step reward negative, and
made falling over early the optimal policy (`notes/experiments.md` E09).

These tests pin both halves of the fix:
  * with no arm command the terms reduce EXACTLY to holosoma's own
    ``sum((a_t - a_{t-1})^2)``, so S0/S1 stay refactor-neutral;
  * with an arm command they ignore the arm channels on exactly the envs the
    skill owns, and no others.
"""

import importlib.util
import sys

import pytest

from everest_locomotion import REPO_ROOT

torch = pytest.importorskip("torch", reason="torch is cloud/WSL-side only")
pytest.importorskip("holosoma", reason="holosoma is cloud/WSL-side only")

EXT = REPO_ROOT / "src" / "everest_locomotion" / "holosoma_ext" / "a3_ultra_loco_v2.py"


@pytest.fixture(scope="module")
def v2():
    spec = importlib.util.spec_from_file_location("everest_loco_v2", EXT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["everest_loco_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


class _Upper:
    def __init__(self, active):
        self.active = active


class _CommandManager:
    def __init__(self, upper=None):
        self._upper = upper

    def get_state(self, key):
        return self._upper if key == "upper_body_command" else None


class _ActionManager:
    def __init__(self, action, prev_action):
        self.action = action
        self.prev_action = prev_action


class _Sim:
    def __init__(self, dof_pos):
        self.dof_pos = dof_pos


class _Env:
    """Just enough env for the action-difference and pose reward terms."""

    def __init__(self, v2, n=8, upper_active=None):
        self.num_envs = n
        self.num_dof = v2.NUM_DOF
        self.device = "cpu"
        self.dof_names = list(v2.DOF_NAMES)          # config order == this order here
        self._arm_idx = torch.tensor(v2.ARM_DOF_IDX, dtype=torch.long)
        g = torch.Generator().manual_seed(0)
        a = torch.randn(n, self.num_dof, generator=g)
        a1 = torch.randn(n, self.num_dof, generator=g)
        a2 = torch.randn(n, self.num_dof, generator=g)
        self.action_manager = _ActionManager(a, a1)
        self._action_tm2 = a2
        self._raw_arm_action = torch.randn(n, v2.NUM_ARM_DOF, generator=g)
        self.simulator = _Sim(torch.randn(n, self.num_dof, generator=g))
        self.default_dof_pos = torch.randn(n, self.num_dof, generator=g)
        upper = _Upper(upper_active) if upper_active is not None else None
        self.command_manager = _CommandManager(upper)

    # the extension caches the name-mapped weight tensor on the env
    def _pose_weight_tensor(self, pose_weights):
        return torch.tensor(list(pose_weights), dtype=torch.float).unsqueeze(0)


def _upstream_action_rate(env):
    """holosoma.managers.reward.terms.locomotion:penalty_action_rate, verbatim."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )


def test_action_rate_is_upstream_when_no_arm_command(v2):
    env = _Env(v2)
    assert v2._policy_owned_mask(env) is None
    torch.testing.assert_close(v2.penalty_action_rate(env), _upstream_action_rate(env))


def test_action_rate_ignores_arms_only_where_the_skill_owns_them(v2):
    n = 8
    active = torch.zeros(n, dtype=torch.bool)
    active[::2] = True  # every other env handed to the skill
    env = _Env(v2, n=n, upper_active=active)

    got = v2.penalty_action_rate(env)
    full = _upstream_action_rate(env)

    d = env.action_manager.action - env.action_manager.prev_action
    arm_part = torch.sum(torch.square(d[:, env._arm_idx]), dim=1)

    # policy-owned envs are untouched; skill-owned envs lose exactly the arm term
    torch.testing.assert_close(got[~active], full[~active])
    torch.testing.assert_close(got[active], (full - arm_part)[active])
    assert torch.all(got <= full + 1e-6)


def test_commanded_arm_motion_costs_the_policy_nothing(v2):
    """The regression that broke S2: a huge arm swing must not change the term."""
    n = 4
    active = torch.ones(n, dtype=torch.bool)
    env = _Env(v2, n=n, upper_active=active)
    before = v2.penalty_action_rate(env)
    jerk_before = v2.penalty_action_jerk(env)

    # slam the arm channels around as an upper-body skill would
    env.action_manager.action[:, env._arm_idx] += 50.0
    env._action_tm2[:, env._arm_idx] -= 30.0

    torch.testing.assert_close(v2.penalty_action_rate(env), before)
    torch.testing.assert_close(v2.penalty_action_jerk(env), jerk_before)


def test_leg_motion_is_still_penalised_under_an_arm_command(v2):
    """Masking must not become a blanket exemption."""
    n = 4
    env = _Env(v2, n=n, upper_active=torch.ones(n, dtype=torch.bool))
    before = v2.penalty_action_rate(env)
    leg = [i for i in range(env.num_dof) if i not in set(v2.ARM_DOF_IDX)]
    env.action_manager.action[:, leg] += 1.0
    assert torch.all(v2.penalty_action_rate(env) > before)


def test_jerk_is_upstream_second_difference_without_a_command(v2):
    env = _Env(v2)
    expect = torch.sum(
        torch.square(
            env.action_manager.action
            - 2.0 * env.action_manager.prev_action
            + env._action_tm2
        ),
        dim=1,
    )
    torch.testing.assert_close(v2.penalty_action_jerk(env), expect)


def test_pose_matches_upstream_when_no_arm_command(v2):
    """S0/S1 must see holosoma's own `pose`, bit for bit."""
    from holosoma.managers.reward.terms.locomotion import pose as upstream_pose

    env = _Env(v2)
    w = [float(i % 7) + 0.5 for i in range(v2.NUM_DOF)]
    torch.testing.assert_close(v2.pose(env, w), upstream_pose(env, w))


def test_pose_releases_arms_only_where_the_skill_owns_them(v2):
    n = 8
    active = torch.zeros(n, dtype=torch.bool)
    active[::2] = True
    env = _Env(v2, n=n, upper_active=active)
    w = [float(i % 7) + 0.5 for i in range(v2.NUM_DOF)]

    got = v2.pose(env, w)
    wt = env._pose_weight_tensor(w)
    full = torch.sum(torch.square(env.simulator.dof_pos - env.default_dof_pos) * wt, dim=1)
    arm_part = torch.sum(
        (torch.square(env.simulator.dof_pos - env.default_dof_pos) * wt)[:, env._arm_idx],
        dim=1,
    )
    torch.testing.assert_close(got[~active], full[~active])
    torch.testing.assert_close(got[active], (full - arm_part)[active])


def test_arm_pose_weight_is_softened_not_removed(v2):
    """Zeroing it is what let the arm channels drift to |a| ~ 11 on the first S2."""
    free = v2._pose_weights(arms_free=True)
    pinned = v2._pose_weights(arms_free=False)
    for i in v2.ARM_DOF_IDX:
        assert free[i] > 0.0, "arm pose weight must not be zero — see penalty_arm_off_target"
        assert free[i] < pinned[i], "arm pose weight must be softened under a skill"
    for i in range(v2.NUM_DOF):
        if i not in set(v2.ARM_DOF_IDX):
            assert free[i] == pinned[i], "non-arm pose weights must be untouched"


def test_arm_off_target_is_zero_without_a_command_and_positive_with_one(v2):
    env = _Env(v2)
    assert torch.all(v2.penalty_arm_off_target(env) == 0.0)

    n = 6
    active = torch.zeros(n, dtype=torch.bool)
    active[:3] = True
    env = _Env(v2, n=n, upper_active=active)
    got = v2.penalty_arm_off_target(env)
    assert torch.all(got[~active] == 0.0), "policy-owned envs already constrained"
    assert torch.all(got[active] > 0.0), "skill-owned envs must get a gradient"

    # and it actually tracks the discrepancy it is meant to measure
    env._raw_arm_action[:] = env.action_manager.action[:, env._arm_idx]
    torch.testing.assert_close(
        v2.penalty_arm_off_target(env), torch.zeros(n)
    )


def test_ladder_stages_share_one_observation_contract(v2):
    """S2+ resume S1's weights, which is only possible if the vector never widens."""
    from holosoma.config_values.experiment import EXPERIMENT_REGISTRY

    dims = {
        "actions": 29, "base_ang_vel": 3, "base_lin_vel": 3, "command_ang_vel": 1,
        "command_lin_vel": 2, "cos_phase": 2, "dof_pos": 29, "dof_vel": 29,
        "heading_error": 1, "projected_gravity": 3, "sin_phase": 2,
        "upper_body_target": 14, "height_scan": 117,
    }

    def width(groups):
        total = 0
        for g in groups.values():
            total += sum(dims[t] * g.history_length for t in g.terms)
        return total

    shapes = set()
    for stage in ("s1", "s2", "s3", "s4"):
        exp = EXPERIMENT_REGISTRY.get(f"a3_ultra_loco_v2_{stage}")
        groups = exp.observation.groups
        actor = {k: v for k, v in groups.items() if k != "critic_obs"}
        critic = {k: v for k, v in groups.items() if k != "actor_obs"}
        shapes.add((width(actor), width(critic)))
    assert len(shapes) == 1, f"observation contract differs across S1..S4: {shapes}"


def test_iterations_are_cumulative_across_the_chain(v2):
    """A resumed run continues `global_step`, so each stage's count must exceed the last."""
    from holosoma.config_values.experiment import EXPERIMENT_REGISTRY

    counts = [
        EXPERIMENT_REGISTRY.get(f"a3_ultra_loco_v2_{s}").algo.config.num_learning_iterations
        for s in ("s1", "s2", "s3", "s4")
    ]
    assert counts == sorted(counts) and len(set(counts)) == len(counts), counts
