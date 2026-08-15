"""The observation layout a policy is graded with must come from the policy.

`ObservationManager.compute_group` concatenates terms in **alphabetically
sorted** order, so a harness that assumes config order silently scores a good
policy as broken (`docs/final_rl_policy.md` §5 — it already cost one run). These
tests pin that against the two shipped policies, which is why they are in git.
"""

import pytest

from everest_locomotion import REPO_ROOT
from everest_locomotion.evaluation.sim2sim import HolosomaPolicy, ObsLayout

LOCO = (REPO_ROOT / "checkpoints" /
        "cloud_20260814_012617-a3_ultra_fast_sac-locomotion" / "model_0050000.onnx")
GETUP = REPO_ROOT / "checkpoints" / "v1_getup_28k_plateau" / "model_27999.onnx"

pytestmark = pytest.mark.skipif(
    not (LOCO.exists() and GETUP.exists()),
    reason="shipped policies missing (see README 'Shipped policies')",
)


@pytest.fixture(scope="module")
def loco():
    return HolosomaPolicy(LOCO)


@pytest.fixture(scope="module")
def getup():
    return HolosomaPolicy(GETUP)


def test_config_derived_layout_reproduces_v1_default(loco):
    """The 68/68 locomotion policy must grade identically either way.

    `v1_default` is the hand-written table the recorded gate was produced with;
    deriving the same columns from the embedded training config is what lets a
    non-locomotion v1 policy (get-up: 93 dims) be graded at all.
    """
    ref = ObsLayout.v1_default(loco.n_dof)
    assert loco.layout.source == "experiment_config"
    assert loco.layout.total_dim == ref.total_dim == 100
    for got, want in zip(loco.layout.terms, ref.terms, strict=True):
        assert (got.name, got.dim, got.scale, got.start) == (
            want.name, want.dim, want.scale, want.start
        )


def test_terms_are_alphabetical(loco, getup):
    for policy in (loco, getup):
        names = [t.name for t in policy.layout.terms]
        assert names == sorted(names)


def test_layout_matches_the_onnx_input_width(loco, getup):
    # HolosomaPolicy raises on mismatch, so reaching here is most of the check;
    # assert the widths explicitly so a regression names the number it broke.
    assert loco.obs_dim == loco.layout.total_dim == 100
    assert getup.obs_dim == getup.layout.total_dim == 93


def test_getup_has_no_locomotion_only_terms(getup):
    """Get-up is command-free: no velocity command, no gait clock."""
    for absent in ("command_lin_vel", "command_ang_vel", "sin_phase", "cos_phase"):
        assert not getup.layout.has(absent)
    assert [t.name for t in getup.layout.terms] == [
        "actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"
    ]


def test_unknown_term_is_a_loud_error():
    cfg = {"observation": {"groups": {"actor_obs": {
        "history_length": 1,
        "terms": {"dof_pos": {"scale": 1.0}, "telepathy": {"scale": 1.0}},
    }}}}
    with pytest.raises(KeyError, match="telepathy"):
        ObsLayout.from_experiment_config(cfg, 29)
