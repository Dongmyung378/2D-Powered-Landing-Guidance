# 2D Powered Landing Guidance

[Korean README](README.ko.md)

A reproducible 2D reusable-rocket powered-landing project that progresses from rigid-body simulation to classical control, optimal-control teaching, imitation learning, and robustness evaluation.

## Project snapshot

| Item | Current status |
|---|---|
| Roadmap | Week 4 final offline dataset complete (Day 24) |
| Model | Planar 3-DoF with variable mass |
| State | x, z, vx, vz, theta, omega, mass |
| Action | throttle, gimbal angle |
| Current controllers | Suicide-burn, vertical PID, horizontal-attitude, and integrated landing |
| Runtime | Python 3.12.7 |

The repository currently provides a tested simulation environment, frozen PID and optimal-control Teacher baselines, reproducible tuning, isolated-disturbance evaluation, and full planar 3-DoF optimal control. Day 24 finalized a leakage-free offline dataset with 1,007 train, 100 validation, and 191 hard OOD test trajectories. Its wider hard-biased train supplement, coverage repair, train-only normalization, and distribution audit are complete. Behavior Cloning, DAgger, combined uncertainty, and broader Monte Carlo evaluation remain later roadmap stages.

## Optimal-control teacher formulation

Day 15 defines a seven-state, two-control nominal landing problem with 100 control intervals and a free final time between 5 and 25 seconds. It includes simulator-matched smooth dynamics, hard altitude/fuel/actuator bounds, landing-state constraints, an initial feasibility objective, and a second-stage fuel/touchdown/smoothness objective. See the [equation-to-code table](docs/spec.md#21-day-15-optimal-control-teacher-formulation) and its [Korean counterpart](docs/spec.ko.md#15일차-최적제어-teacher-문제-정식화).

Day 16 solves the vertical `z, vz, mass` subset with CasADi/IPOPT direct multiple shooting. The command below writes a JSON report with both solver stages and constraint residuals plus a PNG comparing optimization nodes with an independent simulator rollout. It refuses to overwrite existing output files. Install the optional dependency first:

~~~bash
python -m pip install -e ".[optimization]"
python scripts/solve_vertical_landing.py --output-dir artifacts/day16-vertical
~~~

For the nominal `z=100 m`, `vz=-20 m/s`, `mass=1000 kg` case, the solver converges and the independent nominal rollout reaches the ground at about `-0.832 m/s` after `5.000 s`, using about `27.435 kg` of propellant. This is one vertical test case, not a 2D teacher success rate or a wind-robustness result. See the [Day 16 specification](docs/spec.md#22-day-16-vertical-casadiipopt-teacher) or [Korean version](docs/spec.ko.md#16일차-수직-casadiipopt-teacher).

Day 17 adds horizontal position and velocity plus direct thrust-vector angle as a second control. The five-state solver supports either an analytic initial guess or a sampled trajectory from the frozen integrated PID controller. The default demonstration begins 10 m to the right of the pad and writes a 2D flight-path PNG and a JSON constraint report:

~~~bash
python scripts/solve_translation_landing.py --output-dir artifacts/day17-translation
python scripts/solve_translation_landing.py --guess pid --output-dir artifacts/day17-pid
~~~

In the default case, the ideal-vector replay ends about `0.003 m` from the pad, with horizontal velocity `-0.005 m/s` and vertical velocity `-0.902 m/s`. This model directly commands the absolute thrust direction and resets the replay heading between control intervals; it does **not** prove a feasible attitude/gimbal trajectory. See the [Day 17 specification](docs/spec.md#23-day-17-planar-translation-teacher) or [Korean version](docs/spec.ko.md#17일차-평면-병진-teacher).

Day 18 replaces that idealization with the complete `x, z, vx, vz, theta, omega, mass` state and physical throttle/gimbal controls. The solver enforces the 20-degree body-tilt, 15-degree gimbal, and 30 deg/s angular-rate limits, including final attitude and rate constraints. A PID episode supplies only the initial guess; both optimization stages and the independent full-model replay are checked separately.

~~~bash
python scripts/solve_planar_landing.py --output-dir artifacts/day18-planar-3dof
~~~

The default demonstration starts at `x=10 m`, `z=100 m`, `vz=-20 m/s`, `theta=3 deg`, and `omega=-1 deg/s`. With the Day 19 rate-normalized objective, the 100-interval solution finishes in `5.000 s` at approximately `x=0.012 m`, `vx=-0.023 m/s`, `vz=-0.974 m/s`, `theta=0.101 deg`, and `omega=-0.028 deg/s`, using `27.907 kg` of propellant. Peak body tilt is `15.749 deg`, peak angular rate is `21.680 deg/s`, and peak gimbal is `14.671 deg`. This is one nominal, wind-free solution with instantaneous actuators, not a robustness or hardware-safety result. See the [Day 18 specification](docs/spec.md#24-day-18-full-planar-3-dof-teacher) or [Korean version](docs/spec.ko.md#18일차-전체-평면-3자유도-teacher).

## Day 19 objective and numerical study

Day 19 makes every full-planar Stage B term dimensionless with explicit physical scales. Fuel is divided by available propellant, touchdown errors by their landing limits, and consecutive control changes by mesh step and the `2 throttle/s` and `60 deg/s` controller reference rates. This removes the previous mesh-size dependence from the smoothness cost. Three weight profiles expose the fuel/smoothness trade-off, while the baseline profile is solved on 25, 50, 100, and 200 control intervals. Every result is replayed at `0.005 s` through the independent simulator.

~~~bash
python scripts/analyze_optimal_control.py --output-dir artifacts/day19-study
~~~

| Smoothness weight | Fuel used | Peak throttle rate | Peak gimbal rate |
|---|---:|---:|---:|
| 0.001, fuel priority | 27.858 kg | 1.479/s | 50.013 deg/s |
| 0.01, baseline | 27.907 kg | 0.697/s | 23.824 deg/s |
| 0.1, smooth control | 28.003 kg | 0.478/s | 11.685 deg/s |

All four mesh replays passed the per-state final-difference tolerances of `0.001 m`, `0.001 m/s`, `0.01 deg`, `0.01 deg/s`, and `0.001 kg`. The largest final error-to-tolerance ratio fell from `0.231` at 25 intervals to `8.70e-4` at 100 intervals; the measured solve time increased from `0.613 s` to `2.618 s` between 25 and 200 intervals on the development machine. These timings are comparative measurements, not runtime guarantees. Smoothness remains a soft objective, so the reference rates are not hard actuator constraints. See the [Day 19 specification](docs/spec.md#25-day-19-objective-and-numerical-stability-study) or [Korean version](docs/spec.ko.md#19일차-목적함수와-수치-안정성-연구).

## Day 20 multi-initial-condition teacher pipeline

Day 20 samples a deterministic batch with `integrated_uniform_v1`, runs every optimal-control solve in a reusable subprocess, and enforces a 30-second timeout per attempt. The first case uses the frozen PID controller for its initial guess. Later cases first reintegrate the previous successful control sequence from the new initial state; a failed warm start or timeout is retried once with a fresh PID guess. The worker is recycled after 25 attempts to bound long-run solver memory. Every failed attempt preserves its stage, IPOPT status, named constraint diagnostics, exception type, or timeout metadata.

~~~bash
python scripts/generate_teacher_pipeline.py --output-dir artifacts/day20-teacher-pipeline
~~~

The configured seed `20260920` produced batch SHA-256 `7a58dc505c4535e8f081544cea22de17743bff0d80f34debc9454b2029a9f9b9`. All 100 cases converged and passed independent simulator replay on their first attempt: one PID seed and 99 previous-success warm starts. There were no retries, timeouts, or failures. Mean propellant use was `27.349 kg` with a `24.669-30.571 kg` range, and total measured wall time was `126.246 s`. This passes the Day 20 execution gate; it is a finite nominal sample, not a disturbance-robustness guarantee.

The command writes one JSON manifest and one compressed NPZ. The manifest contains the batch hash, policies, every attempt, solver/replay metrics, and an explicit failures list. The NPZ stores only accepted fixed-mesh trajectories with case indices, initial states, time nodes, seven-state nodes, throttle/gimbal controls, and durations. Both files are under the ignored `artifacts/` directory. See the [Day 20 specification](docs/spec.md#26-day-20-multi-initial-condition-teacher-pipeline) or [Korean version](docs/spec.ko.md#20일차-다중-초기조건-teacher-pipeline).

## Day 21 teacher validation and freeze

Day 21 freezes `planar-teacher-v1` with hashes for its complete generation configuration, the 100-case nominal set, and both Day 20 source artifacts. Validation rejects changed dynamics, controller seed policy, objective, mesh, replay tolerance, initial-condition range, episode count, seed, report, or trajectory archive. It also checks the JSON/NPZ schema, array shapes and metadata, accepted case indices, initial states, time nodes, actuator bounds, and explicit failure accounting before any comparison begins.

~~~bash
python scripts/validate_teacher.py
~~~

The command independently reintegrates all accepted control sequences at `0.01 s` and then runs the integrated PID on the exact same 100 initial states. The Teacher solved `100/100`; every independent replay passed terminal, altitude, propellant, tilt, angular-rate, throttle, gimbal, and node-agreement checks. PID landed `99/100`, with one crash that exceeded the final position, horizontal-speed, and angular-rate limits. On the 99 cases both methods landed, mean fuel use was `27.347 kg` for Teacher and `51.907 kg` for PID, a paired Teacher reduction of `24.560 kg`.

Teacher generation took `126.246 s` wall time, with `1.236 s` mean accepted-attempt time. PID nominal evaluation took `20.661 s` including simulation; the controller itself used `5.609 s`, or about `0.483 ms` per control update. These clocks describe offline optimization and online command computation respectively, so the report preserves them separately instead of presenting a direct speed ratio.

The Week 3 gate requires at least 90% solver success, every accepted trajectory to pass a fresh simulator replay, and every optimization failure to remain explicitly listed. All checks passed. Case 22, closest to the median Teacher fuel use, is saved as a standard replay log and a 240-frame GIF. The command produces `teacher-validation.json`, `representative-teacher.json`, and `representative-teacher.gif` under the ignored `artifacts/day21-teacher-validation/` directory and refuses to overwrite them. See the [Day 21 specification](docs/spec.md#27-day-21-teacher-validation-and-freeze) or [Korean version](docs/spec.ko.md#21일차-teacher-검증과-동결).

## Day 22 dataset contract and split design

Day 22 defines `planar-bc-v1` before large-scale trajectory generation begins. The stored sample is a seven-state nonterminal node paired with the following two-control interval. Complete trajectories, not individual rows, are assigned to one of three physical shards. Stable initial-condition IDs detect duplicates within a split and overlap across splits. Failed solver attempts stay in the manifest and never become training labels.

The deterministic plan contains 800 easy train, 100 easy IID validation, and 200 hard OOD test initial conditions. Every test case is farther from the pad, higher, descending faster, and has less available propellant than every train case; its horizontal-speed, attitude, and angular-rate envelopes are also wider. Normalization uses only accepted nonterminal train pairs, stores float64 population statistics, and applies a `1e-6` scale floor. Numeric statistics will be computed after the accepted train trajectories exist on Days 23 and 24.

The generated plan passed all five completion checks with zero duplicate IDs inside a split and zero overlaps between splits. Its configuration digest is `7a45d09cc8d21bc795615538e6cf9c69a6a08eee58ffd1561228ecbc62f929f9`. The train, validation, and test initial-state digests are `06e0203d6ffb9f26687ce6d734fb9010c969b9cb7d139bbef22a1231311937d1`, `ca45656bf8785808716bb522f201bfa72827f45af21d67e98c1485e2652070eb`, and `b244481859f99fd19c3ba44235fbd8f6afb4d31b04e59fa37e9d67c173420725`.

~~~bash
python scripts/design_offline_dataset.py
~~~

The command validates all five completion checks and writes `dataset-design.json` plus `initial-condition-plan.npz` under the ignored `artifacts/day22-dataset-design/` directory. It refuses to overwrite either file. See the [dataset card](docs/dataset-card.md), [Day 22 specification](docs/spec.md#28-day-22-dataset-contract-and-split-design), and their [Korean dataset card](docs/dataset-card.ko.md) and [specification](docs/spec.ko.md#22일차-데이터셋-규약과-split-설계).

## Day 23 large-scale easy Teacher generation

Day 23 reuses the timeout-isolated Teacher worker for the exact 800-condition train plan from Day 22. Each case first uses the previous accepted control sequence as a dynamically reintegrated warm start and retries once from a fresh PID trajectory when necessary. Progress output records elapsed wall time, throughput, and whole-run ETA every 10 cases. The post-solve filter rejects duplicate identities, nonfinite values, malformed state-action alignment or time grids, failed replay, terminal violations, state-bound violations, and actuator violations before packing.

~~~bash
python scripts/generate_offline_dataset.py
~~~

All 800 planned cases were executed in `947.487 s`. The solver and independent replay accepted 797 trajectories, or `99.625%`. Four cases were retried; one retry recovered and three cases remained rejected because the fine-step replay exceeded the tilt bound. There were no timeouts and no additional duplicate, NaN, structure, or constraint-filter rejections among solver successes. The completion gate required all 800 cases to be executed, at least 500 verified trajectories, and a valid packed shard; all checks passed.

The accepted shard contains 80,497 state rows and 79,700 aligned action rows. Its canonical SHA-256 is `04c0f2c0ff3c2e77a28276b37cd8005d2274dad041fbca2e145cd0aec9fe0876`. The manifest and shard remain under the Git-ignored `artifacts/day23-easy-dataset/` directory. Day 24 will generate the wider and harder ranges, inspect coverage, and create the final offline dataset version. See the [Day 23 specification](docs/spec.md#29-day-23-large-scale-easy-teacher-generation) or [Korean version](docs/spec.ko.md#23일차-대규모-쉬운-teacher-궤적-생성).

## Day 24 final offline dataset

Day 24 keeps the verified 797-trajectory Day 23 train shard and adds a wider challenge range that remains strictly easier than the frozen hard OOD test in horizontal distance, altitude, descent speed, and available propellant. It draws 640 deterministic candidates, scores joint difficulty from position, velocity, attitude, angular rate, altitude, and mass, then selects 120 of the hardest candidates plus 40 random candidates from the remainder. All 160 challenge trajectories passed solver and fine-replay validation.

Coverage is measured over eight bins for absolute horizontal offset, altitude, horizontal and vertical speed, attitude, angular rate, and mass. The first analysis found 51 missing marginal counts relative to the required 30 trajectories per bin. One targeted generation round accepted 50 of 51 trajectories; random cross-feature placement also filled the failed target, leaving zero deficits and a worst-bin count of 30. The final train shard therefore contains 1,007 trajectories and 100,700 aligned state-action pairs.

~~~bash
python scripts/finalize_offline_dataset.py
~~~

The IID validation split accepted `100/100`. The hard OOD test accepted `191/200` (`95.5%`); all nine failures were preserved in the manifest after both warm-start and PID attempts failed fine replay, and no timeout or post-solver filter rejection occurred. The final shards have no duplicate initial-condition IDs within or across splits. Normalization is fitted only on the final train transitions. The command writes three packed shards, a versioned manifest, normalization statistics, and a distribution PNG under the Git-ignored `artifacts/day24-final-dataset/` directory and refuses to overwrite them. See the [Day 24 specification](docs/spec.md#30-day-24-final-offline-dataset) or [Korean version](docs/spec.ko.md#24일차-최종-offline-dataset).

## Frozen Week 2 PID baseline

The Day 12 grid-search winner is now the immutable `pid-baseline-v1` comparison point. Its selected position-gain, velocity-gain, and descent-profile scales `(1.10, 0.95, 1.10)` are materialized in `configs/pid-baseline-v1.yaml`, rather than reapplied at runtime. The protocol stores SHA-256 digests for both the integrated-controller settings and the generated initial-condition matrix. Evaluation stops if either changes.

The fixed comparison set contains 1,000 medium-difficulty, wind-free initial conditions generated by `integrated_uniform_v1` with seed 20260914. It covers 3 to 15 m horizontal error, 80 to 120 m altitude, -2 to 2 m/s horizontal speed, -25 to -15 m/s vertical speed, -5 to 5 degrees attitude, -2 to 2 deg/s angular rate, and 950 to 1000 kg mass. Its digest is `a9593a383acc0017738ed383a123ca743a32bce46c64d249b08ca577a448a952`.

| Metric | Result |
|---|---:|
| Safe landings | 999 / 1,000 (99.9%) |
| Mean / maximum absolute touchdown position | 0.4098 / 1.3494 m |
| Mean / maximum horizontal touchdown speed | 0.1656 / 0.5915 m/s |
| Mean / 95th percentile / maximum vertical touchdown speed | 0.7802 / 0.7810 / 0.7814 m/s |
| Mean / maximum touchdown attitude | 0.5991 / 3.2498 degrees |
| Mean / maximum touchdown angular rate | 1.3178 / 3.6500 deg/s |
| Mean propellant use | 51.4460 kg |
| Maximum observed throttle / gimbal slew | 1.8658 per second / 60.0000 deg/s |

The 70% Week 2 gate passed. One case missed the 1 m pad-position limit and was classified as a crash; the other touchdown limits remained within their configured maxima. This is a simulation benchmark, not a hardware-safety result. The artifact command below also renders a nominal success and a clearly labeled stress failure using the same initial state with a 40 m/s constant wind. The stress case is illustrative and is not included in the nominal 999-of-1,000 score.

## Week 2 result summary

| Stage | Fixed evaluation | Result |
|---|---|---:|
| Vertical suicide-burn | Three boundary examples | 3 / 3 safe landings |
| Vertical-velocity PID | 100 vertical, wind-free states | 100 / 100 safe landings |
| Horizontal-attitude loop | 100 eight-second hold cases | 100 / 100 converged toward the pad |
| Integrated PID, tuned validation | 100 unseen states | 100 / 100 safe landings |
| Frozen integrated PID | 1,000 nominal states | 999 / 1,000 safe landings |

All later controller, optimal-control, and learned-policy comparisons must load `configs/pid-baseline-v1.yaml` through `frozen_baseline_initial_states`. This keeps the initial conditions identical and makes configuration drift detectable.

## Reproducible baseline tuning

The integrated controller can be tuned with either Cartesian grid search or seeded random search. The default grid evaluates 13 candidates, including the unscaled reference. Candidate selection uses 16 initial conditions from seed 20260912; the selected candidate is then evaluated once on a separate set of 100 conditions from seed 20260913. The validation set does not influence selection.

The score is minimized and combines failure rate, propellant use normalized by the 250 kg capacity, and mean normalized touchdown error across x, vx, vz, attitude, and angular rate:

~~~text
score = 1000 * failure_rate + 5 * fuel_fraction + 10 * normalized_landing_error
~~~

The selected scales were 1.10 for horizontal position gain, 0.95 for horizontal velocity gain, and 1.10 for descent-profile deceleration. On the training batch, this candidate and the unscaled reference both landed 16 of 16 cases; the selected candidate reduced the score from 4.179125 to 3.716630 and mean propellant use from 52.9862 kg to 51.1532 kg. On the untouched validation batch it landed 100 of 100 cases, with 51.9394 kg mean propellant use and 0.3942 m mean absolute position error.

The generated report records every candidate, objective term, seed, and validation metric. The generated YAML is a complete loadable configuration. `configs/default.yaml` remains the pre-freeze tuning reference, while `configs/pid-baseline-v1.yaml` is the tracked, selected Week 2 baseline used for later comparisons.

## Isolated-disturbance benchmark

The tuned integrated PID was evaluated against one disturbance at a time using 30 paired initial conditions from seed 20260915. The test covers quadratic aerodynamic response to constant wind and a two-second half-sine gust, independent Gaussian sensor noise, actual thrust-scale error, and first-order throttle lag. A curve is marked as collapsed at the first tested strength below a 95% safe-landing rate.

| Disturbance | Last tested strength at or above 95% | First tested collapse strength | Success at collapse |
|---|---:|---:|---:|
| Constant wind | 10 m/s | 20 m/s | 26 / 30 |
| Gust amplitude | 40 m/s | 60 m/s | 26 / 30 |
| Sensor-noise scale | 1x | 2x | 26 / 30 |
| Actual thrust scale | 1.00x | 0.95x | 28 / 30 |
| Throttle-lag time constant | 0.2 s | 0.5 s | 27 / 30 |

The zero-disturbance batch landed 29 of 30 cases. This differs from the earlier 100-of-100 validation because it uses a new seed and exposes a boundary case; it is evidence that a finite successful batch does not prove universal robustness. Disturbance directions, gust start times, sensor-noise streams, and initial states are paired across strength levels, so each curve changes only one modeled factor. These thresholds describe the configured test grid, not certified operating limits.

## Horizontal-attitude control benchmark

The horizontal controller converts position and velocity error into a bounded target tilt, then uses attitude PD control to generate the gimbal command. It compensates throttle for the vertical component lost while the thrust vector is tilted. The configured limits are 2.5 m/s² horizontal acceleration, 15 degrees target tilt, and the vehicle's 15-degree gimbal limit.

A fixed batch of 100 wind-free initial conditions used 5 to 20 m horizontal offsets, -2 to 2 m/s horizontal velocity, -5 to 5 degrees attitude, -2 to 2 deg/s angular rate, and 950 to 1000 kg mass. Each trial held altitude near 100 m for 8 seconds so horizontal and attitude behavior could be isolated from the vertical landing controller.

| Metric | Result |
|---|---:|
| Trials completed | 100 / 100 |
| Trials converging toward the pad | 100 / 100 |
| Mean initial absolute position error | 12.4712 m |
| Mean final absolute position error | 5.4894 m |
| Maximum target tilt | 9.7647 degrees |
| Maximum gimbal command | 7.0393 degrees |
| Maximum altitude deviation | 0.0149 m |

This isolated benchmark remains useful for diagnosing horizontal behavior independently of the integrated landing controller.

## Vertical-velocity PID benchmark

The vertical-velocity controller follows an altitude-dependent descent profile with throttle PID control. Gravity and the profile's nominal deceleration provide feed-forward throttle; conditional integration and an integral bound prevent windup when the actuator reaches its 0 to 1 limits.

The selected gains are `Kp=0.08`, `Ki=0.001`, and `Kd=0.01`. A fixed batch of 100 initial conditions was sampled from 80 to 120 m altitude, -25 to -15 m/s vertical speed, and 950 to 1000 kg mass. Horizontal position, horizontal velocity, attitude, and angular velocity were held at zero for this vertical-only benchmark.

| Metric | Result |
|---|---:|
| Safe landings | 100 / 100 |
| Success rate | 100% |
| Mean touchdown speed | 1.1010 m/s |
| 95th-percentile touchdown speed | 1.1938 m/s |
| Mean propellant use | 37.4923 kg |

Every sampled touchdown stayed below the configured 2 m/s limit.

## Suicide-burn benchmark

The suicide-burn controller calculates when a descending rocket must ignite to meet the configured 2 m/s touchdown-speed limit. The roadmap's constant-mass closed-form estimate is included as the reference calculation. The executable controller also accounts for propellant loss and compensates when ignition falls inside a 0.02-second control interval.

| Initial altitude | Initial vertical speed | Initial mass | Ignition altitude | Touchdown speed | Outcome |
|---:|---:|---:|---:|---:|---|
| 80 m | -15 m/s | 950 kg | 42.184892 m | -1.998971 m/s | success |
| 100 m | -20 m/s | 1000 kg | 58.397961 m | -1.996813 m/s | success |
| 120 m | -25 m/s | 1000 kg | 73.606685 m | -1.995567 m/s | success |

These results validate a vertical, wind-free baseline. They do not establish structural safety or performance with horizontal error, tilt, wind, sensor noise, or actuator delay.

## System flow

~~~text
Initial condition
      |
      v
Gymnasium environment <---- Controller
      |                         ^
      v                         |
Variable-mass dynamics ---------+
      |
      +----> Contact and fuel events
      |
      +----> JSON or NPZ episode record
                    |
                    +----> Plots and animation
~~~

The state and action conventions are shared by simulation, controllers, future datasets, and learned policies. Saved episodes can be replayed without rerunning physics.

## Installation

Use Python 3.12.7 from the repository root.

~~~powershell
python --version
python -m pip install -e ".[dev]"
~~~

## Run the landing controllers

Run the integrated controller from a state with horizontal and vertical error:

~~~powershell
python scripts/run_episode.py --initial-state 10 100 0 -20 0 0 1000 --controller integrated-pid
~~~

Run the velocity-profile PID controller:

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller velocity-pid
~~~

Run the suicide-burn controller:

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller suicide-burn
~~~

The initial-state order is x z vx vz theta omega mass. Values use SI units; theta and omega use rad and rad/s.

Save an integrated landing episode, diagnostic plot, and animation:

~~~powershell
python scripts/run_episode.py --initial-state 10 100 0 -20 0 0 1000 --controller integrated-pid --output artifacts/integrated.json --plot artifacts/integrated.png --animation artifacts/integrated.gif
~~~

Replay the saved record without simulation:

~~~powershell
python scripts/run_episode.py --replay artifacts/velocity-pid.json --animation artifacts/replay.gif
~~~

Generated artifacts are excluded from Git and are never overwritten automatically.

## Verification

Run the complete regression and style checks:

~~~powershell
python -m pytest -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
~~~

Run the Week 1 numerical-stability gate:

~~~powershell
python scripts/verify_week1.py --episodes 100 --seed 20260907
~~~

This gate checks coordinate and torque signs, fuel flow, five termination modes, finite states, monotonic mass, ground penetration, event time, and log consistency.

Compare PID gains on identical sampled initial conditions:

~~~powershell
python scripts/sweep_velocity_gains.py --episodes 100 --seed 20260910
~~~

The sweep reports success rate, touchdown-speed statistics, and mean fuel use. Pass explicit `--kp`, `--ki`, and `--kd` lists to test a different grid.

Run the reproducible horizontal-attitude benchmark:

~~~powershell
python scripts/evaluate_horizontal_control.py --episodes 100 --seed 20260914
~~~

This check reports pad-direction convergence, position-error reduction, tilt and gimbal peaks, and altitude deviation while compensating for vertical thrust loss.

Run the 100-case integrated landing benchmark:

~~~powershell
python scripts/evaluate_integrated_control.py --episodes 100 --seed 20260914
~~~

The report includes every touchdown state component, fuel use, phase update counts, and observed throttle and gimbal slew rates.

Tune the integrated baseline and save a complete report and best configuration:

~~~powershell
python scripts/tune_integrated_controller.py --report artifacts/tuning.json --best-config artifacts/best-integrated.yaml
~~~

Pass `--method random` to run the seeded random search. Neither output is overwritten automatically. The selected configuration can be used directly with other entry points:

~~~powershell
python scripts/run_episode.py --config artifacts/best-integrated.yaml --initial-state 10 100 0 -20 0 0 1000 --controller integrated-pid --output artifacts/tuned-landing.json --plot artifacts/tuned-landing.png --animation artifacts/tuned-landing.gif --fps 20 --max-frames 240
~~~

GIF rendering is completed after the simulation and can take several seconds. `--max-frames` controls rendering time and output size by downsampling the recorded trajectory; it does not alter the simulated physics.

Reproduce the frozen 1,000-case Week 2 benchmark and its success/failure GIFs:

~~~powershell
python scripts/evaluate_frozen_baseline.py --output-dir artifacts/week2-baseline
~~~

The command verifies the controller and initial-condition SHA-256 values before simulation, applies the 70% acceptance gate, and refuses to overwrite any of its five outputs. The two GIFs use the same initial condition: the success is nominal and the failure adds a documented 40 m/s constant-wind stress case.

Generate the isolated-disturbance report and performance curves:

~~~powershell
python scripts/evaluate_pid_disturbances.py --report artifacts/disturbances.json --plot artifacts/disturbances.png
~~~

Use `--episodes` to override the configured 30 trials per strength. Existing outputs are not overwritten.

## Suicide-burn calculation

For the first reference estimate, mass is held constant during the burn:

~~~text
a_net = T_max / mass - gravity
h_stop = (downward_speed^2 - target_speed^2) / (2 * a_net)
~~~

An ignition above the required height is classified as early_burn; ignition below it is late_burn. The executable baseline refines this estimate with the analytic variable-mass solution and a fractional transition command when the ideal ignition occurs between control updates.

## Vertical-velocity profile

The velocity reference decreases the permitted downward speed as altitude falls:

~~~text
target_vz = -min(max_descent_speed, sqrt(touchdown_speed^2 + 2 * deceleration * altitude))
throttle = saturate(feed_forward + Kp * error + Ki * integral + Kd * error_rate)
~~~

The integral term is clamped and stops accumulating when its error would push an already saturated command farther outside the actuator range.

## Horizontal-position and attitude loops

The outer loop maps horizontal position and velocity error to a target horizontal acceleration and bounded target attitude. The inner loop uses attitude error and angular rate to request angular acceleration, then inverts the thrust-torque equation to obtain gimbal angle.

~~~text
target_ax = clip(Kp_x * (target_x - x) - Kd_x * vx)
target_theta = clip(atan2(target_ax, gravity), max_tilt)
target_alpha = Kp_theta * (target_theta - theta) - Kd_theta * omega
gimbal = clip(asin(-target_alpha * inertia / (lever_arm * thrust)))
~~~

When enabled, coupling compensation divides the requested base throttle by `cos(theta + gimbal)`. Throttle and gimbal remain subject to actuator limits. A zero-thrust command safely returns zero gimbal because no attitude torque is available.

## Repository structure

~~~text
README.md / README.ko.md        English default and Korean project overview
configs/                        Shared experiment configuration
docs/spec.md / docs/spec.ko.md  English default and Korean technical specification
docs/dataset-card*.md           English default and Korean dataset documentation
scripts/                        Reproducible experiment entry points
src/powered_landing_guidance/   Physics, environment, controllers, and visualization
tests/                          Regression, boundary, and controller tests
.local/experiment_logs/         Local daily notes excluded from Git
~~~

The tracked repository contains only source code, reproducible configuration, technical documentation, and tests. Daily work notes stay on the development machine under .local/experiment_logs/ and are intentionally absent from GitHub.

## Roadmap

- Week 1: simulator, event handling, logging, replay, and numerical verification - complete
- Week 2: suicide-burn and frozen PID baseline - complete
- Week 3: constrained optimal-control teacher - full planar solver, numerical tuning, 100-case pipeline, validation, comparison, and protocol freeze complete
- Week 4: final 1,007/100/191 train/validation/hard-test offline dataset complete; Behavior Cloning next
- Week 5: DAgger closed-loop improvement
- Week 6: disturbances and Monte Carlo evaluation
- Week 7: report, comparison visuals, and interactive demo

## Current limitations

- Terminal powered descent only; launch, ascent, boost-back, and reentry are out of scope.
- Disturbances are evaluated one at a time; coupled wind, sensing, and actuator failures are not covered yet.
- Aerodynamic drag is a point-force model without aerodynamic torque, lift, altitude-varying density, or turbulence.
- Sensor errors are independent zero-mean Gaussian samples, and engine lag currently affects throttle only.
- Throttle and gimbal rates are penalized but not yet enforced as hard actuator-rate constraints.
- Ground contact uses a point model without landing-leg or structural-impact dynamics.
- The 2 m/s success threshold is a simulation criterion, not a hardware safety guarantee.

See [the model and environment specification](docs/spec.md) for coordinate, unit, event, and episode-log rules. A [Korean specification](docs/spec.ko.md) is also available.
