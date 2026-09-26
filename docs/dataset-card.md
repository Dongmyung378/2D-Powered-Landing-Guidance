# Planar BC Dataset Card

[Korean dataset card](dataset-card.ko.md)

## Status and purpose

`planar-bc-v1-final` is the finalized offline dataset for training and evaluating a Behavior Cloning policy for the project's seven-state planar powered-landing problem. It contains 1,007 train, 100 IID validation, and 191 hard OOD test trajectories. Day 24 added a wider hard-biased train supplement, repaired marginal coverage, fitted train-only normalization, and verified the final physical shards.

The intended supervised mapping is one nonterminal state to the optimal-control action applied over the following interval. The terminal state has no action target. The frozen `planar-teacher-v1` solver is the only accepted label source for this dataset version.

## Intended use

- Train a float32 student policy that predicts throttle and gimbal from the current planar state.
- Select architecture and training settings on an in-distribution validation split.
- Measure generalization on a deliberately harder, out-of-distribution test split.
- Compare learned-policy success, fuel use, constraint violations, and evaluation time against the frozen Teacher and PID baselines.

The dataset is not evidence of flight safety, hardware readiness, six-degree-of-freedom performance, robustness to disturbances, or behavior outside the documented envelopes.

## State and action schema

All stored physical values use float64. Training code may cast batches to float32 only after loading and validation.

| Kind | Ordered fields | Units |
|---|---|---|
| State | `x, z, vx, vz, theta, omega, mass` | `m, m, m/s, m/s, rad, rad/s, kg` |
| Action | `throttle, gimbal_angle` | `1, rad` |

Each accepted trajectory records:

- a stable trajectory ID and split-independent initial-condition ID;
- source case index, duration, time nodes, state nodes, and interval actions;
- state and action offsets into packed arrays;
- solver success, Stage A and B iteration counts, hard and terminal residuals;
- replay result, final and node error ratios, fuel use, and attempt count.

Failed optimization attempts are never inserted into training shards. They remain explicit failure records in the manifest so that dataset yield is not overstated.

## Packed storage contract

Each split uses one `packed_npz_v1` shard with `allow_pickle=False`. Variable-length trajectories are concatenated and recovered with separate state and action offset arrays. For one trajectory, the first `T` states are Behavior Cloning inputs, the `T` actions are targets, and the final state is retained only for trajectory and landing validation.

The required arrays are `trajectory_ids`, `initial_condition_ids`, `source_case_indices`, `state_offsets`, `action_offsets`, `times_s`, `states`, `actions`, `durations_s`, `solver_success`, `stage_a_iterations`, `stage_b_iterations`, `max_hard_violation`, `max_terminal_violation`, `replay_passed`, `max_final_error_ratio`, `max_node_error_ratio`, `fuel_used_kg`, and `attempt_count`.

## Split design

Splits are generated independently with distinct seeds. Validation retains the original easy envelope for in-distribution model selection. Final train combines the easy Day 23 shard with a wider challenge supplement and targeted coverage repair. Test remains a harder OOD envelope with disjoint horizontal offset, altitude, descent speed, and mass ranges plus wider horizontal-speed, attitude, and angular-rate ranges.

| Field | Easy train and validation | Challenge train | Hard test |
|---|---:|---:|---:|
| Absolute horizontal offset | 2 to 10 m | 8 to 11 m | 12 to 20 m |
| Altitude | 70 to 105 m | 90 to 108 m | 110 to 140 m |
| Horizontal velocity | -1.5 to 1.5 m/s | -3 to 3 m/s | -4 to 4 m/s |
| Vertical velocity | -22 to -15 m/s | -23.5 to -20 m/s | -32 to -24 m/s |
| Attitude | -3 to 3 deg | -7 to 7 deg | -10 to 10 deg |
| Angular rate | -1.5 to 1.5 deg/s | -3 to 3 deg/s | -5 to 5 deg/s |
| Mass | 970 to 1000 kg | 962 to 980 kg | 900 to 960 kg |

| Split | Difficulty | Planned trajectories | Seed |
|---|---|---:|---:|
| Train | easy nominal | 800 | 20260922 |
| Train supplement | difficulty-oversampled challenge | 160 from 640 candidates | 20260924 |
| Train coverage | adaptive deficient-bin repair | at most 160 | 20261024 |
| Validation | easy IID | 100 | 20261022 |
| Test | hard OOD | 200 | 20261122 |

The generated Day 22 plan contains no duplicate initial-condition IDs within a split and no overlap across splits. The frozen initial-state SHA-256 values are `06e0203d6ffb9f26687ce6d734fb9010c969b9cb7d139bbef22a1231311937d1` for train, `ca45656bf8785808716bb522f201bfa72827f45af21d67e98c1485e2652070eb` for validation, and `b244481859f99fd19c3ba44235fbd8f6afb4d31b04e59fa37e9d67c173420725` for test.

The hard split is not a random tail of the training distribution. Even after the train envelope was expanded, every test case starts farther from the pad, higher, faster downward, and with less available propellant than every train case. Its remaining motion envelopes are also wider.

## Leakage prevention

The unit of splitting is a complete trajectory, never an individual state-action row. One trajectory can appear in exactly one physical shard. Stable initial-condition IDs are calculated before split labeling from canonical little-endian float64 state bytes. Generation fails when an ID is duplicated within a split or appears across splits. Normalization statistics and any future sampling weights are fitted on train only.

Validation and test trajectories must never be used for normalization, checkpoint selection gradients, replay-buffer augmentation, DAgger collection, or hyperparameter fitting. Validation may select a checkpoint. Test is evaluated only after the training procedure is fixed.

## Normalization contract

Normalization uses standard scores fitted from aligned nonterminal state-action pairs in accepted train trajectories:

~~~text
normalized = (value - train_mean) / max(train_population_std, 1e-6)
~~~

Statistics are accumulated and stored as float64. Each state and action field records count, mean, population standard deviation, applied scale, minimum, and maximum. Angles remain in radians. At inference, the state is normalized, the predicted action is denormalized, and the physical actuator limits are applied. Terminal states are excluded because they do not have an action label.

The final train shard provides 100,700 aligned nonterminal pairs. `normalization.json` stores the resulting float64 statistics and is final for `planar-bc-v1-final`.

## Quality gates

A trajectory is eligible for a training shard only when both optimization stages meet their configured tolerances and a fresh simulator replay passes final-state agreement, landing constraints, altitude, propellant reserve, tilt, angular rate, throttle, and gimbal checks. NaN or infinite values, malformed offsets, duplicate IDs, nonmonotonic time, state-action count mismatches, and failed replay records are rejected.

The dataset manifest reports generated, accepted, retried, timed-out, failed, and filtered counts separately. A high accepted count must not hide optimization failures.

The Day 23 run executed all 800 easy train conditions and accepted 797 trajectories. Four cases were retried; one recovered and three were excluded after fine-step replay exceeded the tilt limit. No accepted solver result was later rejected for duplicate identity, nonfinite data, malformed structure, or another constraint failure. The packed shard passed all dtype, offset, finite-value, uniqueness, time-grid, identity, and accepted-only checks.

Day 24 selected 160 challenge cases from 640 candidates, including the 120 highest difficulty scores; all 160 were accepted. Coverage analysis used eight bins per feature and a minimum of 30 trajectories per bin. One targeted round accepted 50 of 51 cases and reduced the total marginal deficit from 51 to zero. Validation accepted 100 of 100 and hard OOD test accepted 191 of 200. The nine hard-test failures and the one coverage failure are retained in the manifest, while all final shards contain accepted trajectories only. There were no cross-split initial-condition overlaps, timeouts, or post-solver filter rejections.

## First model use (Day 25)

The first `7-64-64-2` Behavior Cloning policy used 100,700 train and 10,000 IID validation state-action pairs. Its small-subset overfit check passed, and the best epoch 79 checkpoint reached normalized validation MSE `0.030977`, throttle MAE `0.032989`, and gimbal MAE `0.272693 deg`. The hard OOD test shard was not read during training or checkpoint selection. These are action-prediction measurements, not landing outcomes; closed-loop evaluation follows later.

## Known limitations

- The model is planar 3-DoF, not a 3D or 6-DoF vehicle.
- Training labels are nominal and wind-free unless a later dataset version says otherwise.
- Control-rate limits are soft objective terms, not hard actuator slew constraints.
- Ground contact has no landing-leg or structural-impact model.
- The hard test is harder by initial-condition design but does not include sensor noise, engine lag, turbulence, or model mismatch.
- Teacher trajectories close to actuator or state bounds may give the student little robustness margin.

## Reproduction

~~~bash
python scripts/design_offline_dataset.py
python scripts/generate_offline_dataset.py
python scripts/finalize_offline_dataset.py
~~~

The final command verifies the Day 23 source and writes the version manifest, normalization statistics, distribution PNG, and three physical shards under `artifacts/day24-final-dataset/`. Generated directories are ignored by Git and commands refuse to overwrite outputs. Final shard SHA-256 values are train `4161bc93b6753bdcc84cfb6eec6afe1c45bbf814b34525817a7ec24835691e0d`, validation `e77b561f63f5738ec6e09e24b9dbffe2e4712ff33e7d186d12b4945e2aaa618c`, and test `6b67c7866e202b6719f07c733cd2cd82fdd3e2f1638c5bb33ec10ea3f3f5ba17`.
