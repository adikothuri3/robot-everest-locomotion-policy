# Get-up / fall-recovery recipe selection for the A3 Ultra (2026-08-13)

Ranked evaluation of every credible open get-up training recipe, judged by
real-hardware evidence first, code quality second, paper claims last.
Decision: **reimplement HoST's recipe as a Holosoma extension; HumanUP's
slow-down stage as smoothness insurance; throwaway HoST Isaac Gym probe first.**
Full execution plan lives in the approved mission plan; asset/keyframe/pose-bank
work items are P0.x in the experiment ladder (E11+).

## The honest headline

No recipe cleanly fits a 60 kg, 1.74 m robot. Both open hardware-proven recipes
(HoST, HumanUP — RSS 2025) were demonstrated only on the 35 kg Unitree G1, in
legacy Isaac Gym, and are now unmaintained. Nothing above 47 kg (sim-only H1)
has a public learned-get-up datapoint; a community H1-2 HoST port "stood up but
behaved strangely" (HoST issue #35). All matching 2026 advances (UniReLo,
Stubborn, VIGOR, RecoverFormer) have **no released code**. Expect significant
reward re-tuning. Mitigating: AgiBot demoed the base A3 doing a kip-up (the
hardware can do it), and the A3's legs are torque-rich (320 Nm knees vs G1's
~139 Nm). The weak point is the arms: 60 Nm shoulders / 24 Nm elbows / 6 Nm
wrists on a 60 kg body — G1's arms are proportionally ~2× stronger per kg.

## Scorecard (evidence-ranked)

| Recipe | HW evidence | Poses | Non-flat | Code | Verdict |
|---|---|---|---|---|---|
| **HoST** ([InternRobotics/HoST](https://github.com/InternRobotics/HoST), MIT, [arXiv:2502.08378](https://arxiv.org/abs/2502.08378)) | G1: 5/5 on ground/platform/wall/slope, 1–15° incl. slippery, pushes, 12 kg payload | supine/prone/side/seated/leaning | **slope/platform/wall training envs released** | Isaac Gym; abandoned 6/2025, 23 open issues unanswered | **Base recipe** |
| **HumanUP** ([RunpeiDong/HumanUP](https://github.com/RunpeiDong/HumanUP), Apache-2.0, [arXiv:2502.12152](https://arxiv.org/abs/2502.12152)) | G1: 78 % get-up / 98 % roll-over on 6 surfaces incl. snow, ~10° grass slope | supine + prone (two policies) | flat-trained (slope-tested, snow/slope failures logged) | Isaac Gym; idle since 4/2025 | Stage-II slow-down donor |
| UniReLo ([arXiv:2606.08922](https://arxiv.org/abs/2606.08922)) | G1 outdoors: gravel, grass, 10–15° slopes, unified recovery→walk | diverse | **yes, real** | none | Design donor: terrain-relative support features, gated handoff |
| HiFAR ([arXiv:2502.20061](https://arxiv.org/abs/2502.20061)) | Booster T1 (118 cm): slopes, obstacles, μ=0.1 | diverse | yes | none | DR-ranges donor (friction/compliance) |
| BeyondMimic tracking ([whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking), MIT, Isaac Lab) | LAFAN1 fallAndGetUp on real G1 (paper table only) | reference-start only | flat lab | maintained | Fallback; no initial-pose robustness |
| ASAP / delta-action | not a get-up method; broke 2 G1s | — | — | MIT | Drop |
| Unitree stacks | closed firmware; no recovery task in any public repo | — | — | — | Drop |
| Isaac Lab / Holosoma / AgiBot official | no get-up env exists anywhere; AgiBot ships no A3 RL code | — | — | — | Build-it-yourself confirmed |

2026 papers without code, tracked for ideas: Stubborn (Bernoulli termination for
recovery inside a tracker, [arXiv:2606.12814](https://arxiv.org/abs/2606.12814)),
VIGOR (depth-conditioned fall safety on non-flat ground,
[arXiv:2602.16511](https://arxiv.org/abs/2602.16511)), RecoverFormer
(obstacle-assisted recovery, sim-only, [arXiv:2604.22911](https://arxiv.org/abs/2604.22911)),
SD-AMP unified walk/run/recovery ([arXiv:2605.18611](https://arxiv.org/abs/2605.18611)),
APEX ratchet-progress reward ([arXiv:2602.11143](https://arxiv.org/abs/2602.11143)),
heavier-robot recoveries by other methods: H1-2 via balance-metric RL
([arXiv:2603.08619](https://arxiv.org/abs/2603.08619)), Booster T1 in AGILE
([arXiv:2603.20147](https://arxiv.org/abs/2603.20147)).

## Why HoST over HumanUP as the base

1. Pose coverage: supine + prone + side + seated + leaning vs HumanUP's two
   single-purpose policies.
2. Non-flat ground: HoST ships slope/platform/wall training envs and showed real
   slope recovery — the rare property we need for alpine.
3. Documented new-robot scaling heuristics (README): pull force ≈ 60 % weight,
   curriculum threshold ≈ 70 % height, stage heights ≈ 35 %/70 %, head target
   ≈ 75 %, hardware kp ×1.33–1.5, primitive collisions. For A3: ~350 N, 1.2 m,
   0.6/1.2 m, ~1.3 m.
4. Single-stage → simpler to reimplement in Holosoma.
5. Ablation-verified smoothness machinery with hardware evidence.

## Smoothness (what actually works)

- HoST, ablation-ranked: **multi-critic** (remove → zero success),
  **pull-force curriculum** (remove → 6.8 % vs 99.8 % off-platform),
  **action rescaler β** 1.0→0.25 curriculum (deploy at 0.3; sim success stays
  high without it but smoothness degrades 9.52 vs 2.90 — the hardware killer),
  **L2C2 smoothness loss** (11/20 → 20/20 on real G1). Plus action-rate,
  torque, dof_acc, dof_vel penalties and hard DOF/base velocity terminations.
  Watch HoST issue #42: verify `smoothness_lower_bound` ≠ 0 no-op, and issue
  #37 (reward bug report).
- HumanUP: Stage-I discovery trajectory re-tracked **8× slower** (4× already
  saturated the G1; 10× fails to converge) with heavy regularizers
  (action_rate −0.1, torques −0.003, torque-limit −5). This is the strongest
  known fix if HoST-style regularization alone is too jerky at 60 kg.
- Both flag 1 kHz sim dt as important — consistent with our armature/200 Hz PD
  integration findings from the locomotion port.

## Fallen-pose coverage

- HoST: per-posture randomized joints + base pose (not drop-and-settle);
  unactuated first ~0.5 s of each episode.
- HumanUP: 20 K supine + 20 K prone via randomize-drop-settle (script not
  public). Ours: `scripts/getup/generate_fallen_poses.py` (drop 0.5 m, settle,
  classify by projected gravity against the lying keyframes baked into
  `a3_ultra_29dof.xml`).
- Slopes/obstacles: HoST slope env (train 0–15°); HiFAR DR ranges (friction to
  0.1, terrain compliance, obstacles under body). Nobody has published
  snow/rock-field get-up; >15° slope get-up is unreported. That frontier is ours.

## The handoff (success ≠ "torso vertical")

No public recipe ships a get-up→locomotion handoff. HoST success is emergent
(base height > 0.7 m sustained, G1); HumanUP is head height ≥ 1.1 m. Our
convention (manifest `getup.terminal`): terminal = locomotion `default_pose`
at 1.063 m within pose/velocity tolerances, held 2 s; chained
get-up→locomotion ONNX test in the MuJoCo gate from day one. UniReLo's
continuous gating is the eventual upgrade path.

## Porting surface & failure modes at 60 kg

Robot-specific: asset + full-body primitive collisions (done, asset v5),
fallen-pose bank (done), PD gains under load (assumed values need get-up-phase
validation), reward height targets rescaled, contact body names, torque budget
asymmetry (arms weak → bias to leg-dominant strategies; waist pitch limit
−0.49..0.42 disfavors sit-up routes; wrists soft-frozen, see manifest).
Known failure modes: knee/hip saturation at sit-to-squat; kp sim-to-real gap
(×1.33–1.5); arm/torso contact fidelity; "stands but behaves strangely"
(HoST #35) — motion quality, not success rate, is the hard part at scale.

## Update 2026-08-15 — the multi-policy architecture (post run #1)

Run #1 (one policy, all postures) plateaued at 30% success ≈ the bank's supine
share. Deep-dive confirmed the literature predicted this: HoST's limitations
section states training with supine+prone together "negatively impacted
performance due to interference between sampled rollouts", and HoST ships one
policy per posture (`g1_ground`, `g1_ground_prone`, plus per-terrain variants);
HumanUP ships a prone→supine rollover policy (98.3% hardware) plus a supine
get-up policy (78.3%). Side-lying coverage in HoST came from the PRONE policy
zero-shot (its reset code has no roll randomization — sides were generalization).

**Our architecture (implemented in `a3_ultra_getup.py`):**
- `a3-ultra-rollover`: prone+side starts → settled supine. 5 s episodes, no
  assist. Success = base AND torso projected-gravity-x < −0.85 (HumanUP's
  multi-body cosine ≥ 0.9), pelvis < 0.45 m, settled, held 1 s. Rewards: dense
  face-up cosine (base+torso averaged), gated roll-rate bonus (HoST's prone
  config weights its ang-vel style term 25× the supine value — the term that
  makes rolling emerge), settle-on-back stillness, anti-rise penalty, standard
  smoothness stack under PenaltyCurriculum.
- `a3-ultra-getup`: supine starts only (+5–30% standing anchors, HumanUP's
  `standing_init_prob` pattern) → manifest handoff pose. Assist curriculum
  `up_threshold` 0.45.
- Runtime router (E14, numbers from the literature): fall = tilt >37°
  (|pg_z+1|>0.6, SD-AMP) → damp window (Unitree FSM pattern) → chest-up ? getup
  : rollover; rollover→getup on face-up ≥0.9 held 0.5 s (FRASA debounce);
  getup→locomotion on handoff gate held 0.5 s, then ready-pose before commands.

**Exploration lesson (why run #1's std collapse wasn't the root cause):** no
recipe fixes PPO exploration with noise floors; all of them make the task easy
first — HoST's 200 N pull force, HumanUP's weak-regularization Stage I +
simplified collisions, HiFAR's sagittal-plane stage + keyframe resets. Our
assist curriculum plays that role for get-up; the rollover task is dense enough
(cosine gradient + roll-rate) not to need one. FRASA's CrossQ (max-entropy
off-policy) is the fallback precedent if discovery still stalls.
