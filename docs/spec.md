# Model and Environment Specification

[Korean specification](spec.ko.md)

This document defines the state, control, coordinate, unit, event, and episode-record contracts shared by the simulator, controllers, datasets, and future learned policies. Any change to a sign or unit must update both the implementation and this specification.

## 1. Model scope

- 2D planar rigid body
- Translational degrees of freedom: `x`, `z`
- Rotational degree of freedom: `theta`
- Variable vehicle mass: `mass`
- Control inputs: throttle and thrust-vector gimbal angle
- Implemented effects: gravity, thrust, propellant consumption, point-position ground contact, and optional quadratic aerodynamic drag with wind
- Evaluation-only effects: Gaussian sensor noise, actual thrust-scale error, and first-order throttle lag
- Fixed parameters: moment of inertia and engine lever arm

## 2. Inertial frame

~~~text
                       +z (up)
                        ^
                        |       body axis
                        |      /   theta > 0
                        |     /
                        |    /
                  COM   o----------------> +x (right)
                        |
                        |
                   ground: z = 0
~~~

- Origin: landing target on the ground
- `+x`: right on screen
- `+z`: upward
- Gravity vector: `(0, -g)`
- Ground: `z = ground_z_m`, with a default of `0 m`

## 3. Angles and signs

- `theta = 0`: the vehicle longitudinal axis is aligned with `+z`
- `theta > 0`: the nose tilts clockwise toward `+x`
- `omega = d(theta)/dt`: positive in the same direction as `theta`
- `gimbal_angle = 0`: thrust is aligned with the vehicle axis
- `gimbal_angle > 0`: the thrust vector rotates from the vehicle axis toward `+theta`
- `gimbal_angle` describes the force-vector direction, not mechanical nozzle deflection

The inertial thrust-vector angle is `theta + gimbal_angle`:

~~~text
thrust_x = thrust * sin(theta + gimbal_angle)
thrust_z = thrust * cos(theta + gimbal_angle)
~~~

The equations of motion are:

~~~text
x_dot     = vx
z_dot     = vz
vx_dot    = thrust_x / mass
vz_dot    = thrust_z / mass - g
theta_dot = omega
omega_dot = torque / moment_of_inertia
mass_dot  = -thrust / (specific_impulse * standard_gravity)
~~~

Applied thrust, torque, and mass flow become zero when the vehicle reaches dry mass:

~~~text
if mass <= dry_mass:
    applied_thrust = 0
    torque = 0
    mass_dot = 0
~~~

`standard_gravity` converts specific impulse in seconds to effective exhaust velocity in meters per second. It is distinct from the environmental `gravity` value.

The engine application point lies below the center of mass on the vehicle axis. For engine lever arm `l`, thrust `T`, and gimbal angle `delta`, the torque convention is:

~~~text
torque    = -l * T * sin(delta)
omega_dot = torque / moment_of_inertia
~~~

A positive gimbal therefore produces negative angular acceleration, and a negative gimbal produces positive angular acceleration.

## 4. State vector

The array order is part of the public interface and must not change silently.

| Index | Name | Meaning | Internal unit |
|---:|---|---|---|
| 0 | `x` | Horizontal position | m |
| 1 | `z` | Vertical position | m |
| 2 | `vx` | Horizontal velocity | m/s |
| 3 | `vz` | Vertical velocity | m/s |
| 4 | `theta` | Attitude from the vertical axis | rad |
| 5 | `omega` | Angular velocity | rad/s |
| 6 | `mass` | Current vehicle mass | kg |

~~~text
state = [x, z, vx, vz, theta, omega, mass]
~~~

## 5. Control vector

| Index | Name | Meaning | Internal unit or range |
|---:|---|---|---|
| 0 | `throttle` | Fraction of maximum thrust | dimensionless, `[0, 1]` |
| 1 | `gimbal_angle` | Thrust-vector angle from the vehicle axis | rad |

~~~text
control = [throttle, gimbal_angle]
~~~

Human-readable angle limits and initial values in YAML use the `_deg` suffix. Conversion to radians happens once at the simulator boundary; internal dynamics, logs, and future datasets use radians.

Controller commands are limited before entering the dynamics:

~~~text
applied_throttle = clip(commanded_throttle, throttle_min, throttle_max)
applied_gimbal   = clip(commanded_gimbal, -gimbal_limit, +gimbal_limit)
~~~

## 6. Invariants

- `mass >= dry_mass`
- `0 <= throttle <= 1`
- `abs(gimbal_angle) <= gimbal_limit`
- Every state and control value is finite
- Time step, maximum time, gravity, mass, and thrust parameters are positive
- Initial mass is at or above dry mass

## 7. Landing-success criteria

The pre-contact state must satisfy every configured limit:

| Criterion | Default limit |
|---|---:|
| Horizontal position error | `abs(x) <= 1.0 m` |
| Horizontal speed | `abs(vx) <= 1.0 m/s` |
| Vertical speed | `abs(vz) <= 2.0 m/s` |
| Attitude | `abs(theta) <= 5 deg` |
| Angular speed | `abs(omega) <= 5 deg/s` |
| Mass | `mass >= dry_mass` |

These values are initial project criteria stored in `configs/default.yaml`. Controller comparisons use the same seeds and initial conditions.

## 8. Environment API

Create `RocketLandingEnv(config)` with a dictionary returned by `load_config`.

- `observation, info = reset(seed=None, options=None)`
- `next_state, reward, terminated, truncated, info = step(action)`
- `episode_log`: isolated, JSON-compatible copy of the current episode record
- `save_episode(path)`: write a new record without overwriting an existing file

Calling `step` before `reset` or after termination raises an error. Calling `reset` starts a new episode, so save any required record first. The environment rejects non-finite values, invalid action shapes, below-ground initial states, and initial mass below dry mass.

Observations are seven-element `float64` state arrays. Actions are `float64` `[throttle, gimbal_angle]` arrays in SI-compatible units. Position and velocity deliberately have no arbitrary finite bounds, so Gymnasium's checker can emit advisory warnings about infinite observation-space limits. Non-finite observations are never allowed.

## 9. Initial conditions

By default, `reset` independently samples configured fields from `initial_state_sampling`. YAML stores the same units as `initial_state`, except that attitude and angular velocity are expressed in degrees before conversion to radians.

- `options={"randomize": False}`: use the nominal initial state
- `options={"initial_state": [...]}`: use an exact seven-element SI/radian state
- `reset(seed=n)`: initialize the environment random generator
- `reset(seed=None)`: continue its existing random-generator state

The same seed and action sequence produce the same record. Callers that use `action_space.sample()` must seed the action space separately. An exact initial state and recorded action sequence can reproduce the physics without the original reset seed.

## 10. Termination

Ground contact uses point-position `z`. The returned contact observation is never below ground. Attitude is wrapped with `atan2(sin(theta), cos(theta))` only for contact evaluation; internal states and records retain the unwrapped angle.

| Outcome | Rule |
|---|---|
| `running` | No terminal event |
| `success` | Position, velocity, attitude, and angular-speed limits all pass |
| `hard_landing` | Position, attitude, and angular speed pass, but `vx` or `vz` fails |
| `crash` | Position, attitude, or angular speed fails |
| `fuel_depletion` | Dry mass is reached in flight |
| `timeout` | Maximum simulation time is reached |

A state that begins on the ground or at dry mass is evaluated at `t=0` on the first step. Simultaneous events use ground contact, fuel depletion, then timeout priority. Sequential events stop at the earliest event time. Fuel depletion terminates the environment immediately; the dynamics function itself can still model a later ballistic segment when called directly.

`success`, `hard_landing`, `crash`, and `fuel_depletion` set `terminated=True`. Only `timeout` sets `truncated=True`. Success reward is `+1`, physical failure reward is `-1`, and running or timeout reward is `0`. A final learning reward has not been designed yet.

## 11. Events inside a step

One action remains constant for `simulation.dt_s`. Internal integration intervals are at most 0.02 seconds, and the final interval is shortened to hit the maximum simulation time or fuel-depletion time exactly.

Each internal interval checks for ground crossing. Brent root search locates the contact time. Because a powered trajectory can descend and rise while both interval endpoints remain above ground, a conservative acceleration bound identifies intervals that also need a minimum-altitude search.

The root-search time tolerance is `1e-12 s`; final physical accuracy still depends on the selected Euler or RK4 integrator and its integration interval. A descending endpoint within floating-point tolerance of the ground is treated as contact to prevent a coincident event from being misclassified as timeout.

## 12. Episode record schema version 1

| Field | Contents |
|---|---|
| `config` | Complete configuration used for the episode |
| `seed_argument` | Seed passed to reset, or `null` |
| `rng_state_after_reset` | Generator state immediately after initial-state creation |
| `state_names`, `state_units` | State order and units |
| `action_names`, `action_units` | Action order and units |
| `initial_state` | Seven-element SI state at `t=0` |
| `steps` | Ordered step results |
| `outcome` | Latest outcome |

Each step contains `index`, `time_s`, `state`, `commanded_action`, `clipped_action`, `thrust_start_n`, `thrust_end_n`, `fuel_used_kg`, `reward`, `terminated`, `truncated`, `outcome`, and `is_success`.

`fuel_used_kg` is the difference between initial and current mass. `clipped_action` contains actuator-limited commands; `thrust_start_n` and `thrust_end_n` show whether dry-mass cutoff actually removed thrust. Returned states, info dictionaries, and logs are copied so external modification cannot mutate environment state.

## 13. Replay and visualization

`save_episode_data` writes the same episode record as JSON or compressed NPZ. NPZ stores the source JSON plus validation arrays for `times_s`, `states`, `actions`, and `thrust_n`, without object pickle. `load_episode_data` checks those arrays against the source record. Both formats replay stored states directly rather than rerunning physics.

Time-series plots show position, velocity, attitude, angular speed, mass, throttle, and gimbal. The 2D animation shows the x-z trajectory, vehicle attitude, thrust vector, ground, and the `x=0` landing target. Vehicle geometry is illustrative and does not model real dimensions or landing legs. GIF is supported directly; MP4 requires FFmpeg. Output functions never overwrite an existing file.

## 14. Vertical suicide-burn baseline

`SuicideBurnController` handles vertical descent with zero horizontal and attitude error. The first ignition-height estimate holds mass constant during the burn:

~~~text
net_deceleration = max_thrust / mass - gravity
stopping_distance = (downward_speed^2 - target_speed^2) / (2 * net_deceleration)
~~~

`estimate_suicide_burn` exposes that reference calculation. The executable controller refines ignition height with the analytic variable-mass vertical solution. If available propellant is insufficient or maximum thrust cannot overcome gravity, the estimate is `infeasible` and the controller commands immediate maximum thrust.

When ideal ignition lies within one 0.02-second control interval, the first burn command uses fractional throttle to reduce timing quantization. Maximum thrust remains latched after ignition. Ignition above the requirement is `early_burn`, below it is `late_burn`, and within tolerance it is `on_time`.

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller suicide-burn
~~~

This baseline does not compensate for wind, horizontal error, tilt, sensor delay, or engine delay. The 2 m/s contact limit is a simulation criterion, not a hardware structural-safety guarantee.

## 15. Vertical-velocity PID baseline

`VerticalVelocityPIDController` generates an altitude-dependent vertical-speed reference and tracks it with throttle PID. Its present evaluation scope holds horizontal position, horizontal velocity, attitude, and angular velocity at zero.

~~~text
speed_limit = sqrt(touchdown_speed^2 + 2 * profile_deceleration * altitude)
target_vz = -min(max_descent_speed, speed_limit)
error = target_vz - current_vz
~~~

The default profile uses a 1 m/s touchdown target, 30 m/s maximum descent speed, and 3 m/s² nominal deceleration. Feed-forward throttle compensates for gravity and reference acceleration before PID correction:

~~~text
feed_forward = mass * (gravity + target_acceleration) / max_thrust
raw_throttle = feed_forward + Kp * error + Ki * integral + Kd * error_rate
throttle = clip(raw_throttle, throttle_min, throttle_max)
~~~

Default gains are `Kp=0.08`, `Ki=0.001`, and `Kd=0.01`. Integrated error is limited to ±10 m. Integration is cancelled for a step when positive error would push an upper-saturated command higher or negative error would push a lower-saturated command lower. `reset` clears the integral, previous error, and last-command telemetry.

The configured evaluation range independently samples altitude from 80 to 120 m, vertical speed from -25 to -15 m/s, and mass from 950 to 1000 kg. With seed 20260910, all 100 sampled cases landed successfully. Mean touchdown speed was 1.1010 m/s, the 95th percentile was 1.1938 m/s, and mean propellant use was 37.4923 kg.

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller velocity-pid
python scripts/sweep_velocity_gains.py --episodes 100 --seed 20260910
~~~

These results apply only to the current deterministic, vertical, wind-free simulation range. Horizontal and attitude control, wind, model error, sensor noise, and actuator delay remain outside this baseline.

## 16. Horizontal-position and attitude controller

`HorizontalAttitudeController` is a cascaded, non-learning controller. It accepts the current state and a base throttle from a separate vertical controller. This interface keeps horizontal-attitude logic independent until the integrated-controller stage.

The outer loop calculates horizontal acceleration from position and velocity error:

~~~text
raw_ax = Kp_x * (target_x - x) - Kd_x * vx
target_ax = clip(raw_ax, -max_ax, max_ax)
target_theta = clip(atan2(target_ax, gravity), -max_tilt, max_tilt)
~~~

The default outer-loop gains are `Kp_x=0.05 s^-2` and `Kd_x=0.45 s^-1`. Horizontal acceleration is limited to 2.5 m/s² and target tilt to 15 degrees.

The inner loop calculates desired angular acceleration and inverts the documented torque convention. Positive gimbal produces negative angular acceleration:

~~~text
target_alpha = Kp_theta * (target_theta - theta) - Kd_theta * omega
sin(gimbal) = -target_alpha * inertia / (lever_arm * thrust)
~~~

The default inner-loop gains are `Kp_theta=4.0 s^-2` and `Kd_theta=3.0 s^-1`. The inverse is clipped to its mathematical domain and then to the 15-degree vehicle gimbal limit. If base throttle is zero, gimbal is zero because thrust vectoring cannot produce torque.

Tilting reduces vertical thrust by `cos(theta + gimbal)`. With coupling compensation enabled, the output throttle is iteratively adjusted so its vertical component matches the requested base throttle, subject to the 0 to 1 actuator range.

The reproducible evaluation uses 100 wind-free, fixed-duration hover trials at 100 m. Initial horizontal error is at least 5 m within -20 to 20 m; horizontal speed, attitude, angular rate, and mass are sampled from the configured ranges. With seed 20260914, all 100 trials completed and reduced absolute horizontal error. Mean error fell from 12.4712 m to 5.4894 m. Maximum target tilt was 9.7647 degrees, maximum gimbal was 7.0393 degrees, and maximum altitude deviation was 0.0149 m.

~~~powershell
python scripts/evaluate_horizontal_control.py --episodes 100 --seed 20260914
~~~

This isolated result establishes wind-free convergence toward the pad without claiming touchdown success. The integrated controller below combines these commands and evaluates complete landings.

## 17. Integrated landing controller

`IntegratedLandingController` combines the vertical-velocity PID throttle with horizontal-position and attitude gimbal control. The simulator advances at 0.02-second intervals, while the controller updates every 0.1 seconds. The latest command is held for the four intermediate simulation steps. The controller rejects a control interval that is shorter than, or not an integer multiple of, the simulation interval.

Altitude selects one of two independently stateful control phases. Above 30 m, the approach phase uses a 1.0 m/s touchdown target, 1.5 m/s² profile deceleration, vertical PID gains `(0.08, 0.001, 0.01)`, horizontal gains `(0.5, 1.7)`, and a 15-degree target-tilt limit. At or below 30 m, the terminal phase uses a 0.8 m/s touchdown target, 1.0 m/s² profile deceleration, vertical PID gains `(0.1, 0.001, 0.015)`, horizontal gains `(0.3, 1.35)`, and a 5-degree target-tilt limit. The terminal limits prioritize an upright, low-rate contact.

The vertical controller first produces base throttle. The selected horizontal-attitude controller then calculates gimbal and compensates the tilted thrust's vertical component. The combined command is finally limited relative to the previous control update:

~~~text
abs(throttle[k] - throttle[k-1]) <= 2.0 * control_interval
abs(gimbal[k] - gimbal[k-1]) <= deg2rad(60) * control_interval
~~~

The first calculated command initializes the held action because no previous controller output exists. Later updates expose separate throttle and gimbal slew-limit flags. Time must be finite, nonnegative, and nondecreasing. `reset` clears both phases, held commands, update counters, and scheduling state.

The medium-difficulty evaluation samples 100 wind-free cases with horizontal error from 3 to 15 m on either side of the pad, altitude from 80 to 120 m, horizontal speed from -2 to 2 m/s, vertical speed from -25 to -15 m/s, attitude from -5 to 5 degrees, angular rate from -2 to 2 deg/s, and mass from 950 to 1000 kg. With seed 20260914, all 100 cases satisfied every touchdown limit.

Mean and maximum absolute touchdown position were 0.4516 m and 0.9440 m. Mean and maximum horizontal speed were 0.1476 m/s and 0.3526 m/s. Mean and maximum vertical speed were 0.7819 m/s and 0.7827 m/s. Mean and maximum attitude were 0.7559 degrees and 1.4386 degrees, while angular-rate values were 1.1004 deg/s and 2.4399 deg/s. Mean propellant use was 53.0159 kg. Maximum observed slew was 1.7784 throttle units per second and 60.0000 deg/s gimbal.

~~~powershell
python scripts/evaluate_integrated_control.py --episodes 100 --seed 20260914
python scripts/run_episode.py --initial-state 10 100 0 -20 0 0 1000 --controller integrated-pid
~~~

This controller is a deterministic wind-free baseline. Disturbance rejection, uncertainty, sensor noise, actuator delay, and hardware safety remain outside this result.

## 18. Baseline tuning automation

`tune_integrated_controller.py` searches three scale factors applied to both integrated-controller phases: horizontal position gain, horizontal velocity gain, and descent-profile deceleration. It supports a Cartesian `grid` search and a uniform seeded `random` search. Both methods always include the unscaled `(1, 1, 1)` reference. Scaling is performed on an isolated configuration copy, so the input configuration is not mutated.

Each candidate is evaluated on the same training batch. Selection minimizes the following scalar objective:

~~~text
failure_rate = 1 - success_rate
fuel_fraction = mean_fuel_used / (initial_mass - dry_mass)
normalized_landing_error = mean(
    mean_abs_x / x_limit,
    mean_abs_vx / vx_limit,
    mean_abs_vz / vz_limit,
    mean_abs_theta / theta_limit,
    mean_abs_omega / omega_limit,
)
score = 1000 * failure_rate + 5 * fuel_fraction + 10 * normalized_landing_error
~~~

The failure term dominates the two secondary terms, while normalized fuel and touchdown error make their scales explicit. Objective weights, search ranges, candidate count, episode counts, and seeds are configuration values. Invalid ranges, nonpositive counts, equal train and validation seeds, and an objective with neither secondary term are rejected during configuration loading.

The default Cartesian grid contains 12 scale combinations and adds the unscaled reference for 13 total candidates. Training uses 16 initial conditions from seed 20260912. Only the training objective selects the winner. The winner is then evaluated once on 100 independently sampled initial conditions from seed 20260913.

The selected scales were `(1.10, 0.95, 1.10)`. All 16 training cases landed safely; the objective was 3.716630 and mean propellant use was 51.1532 kg. The unscaled candidate also landed all 16 cases but scored 4.179125 and used 52.9862 kg. Independent validation produced 100 safe landings in 100 cases, 51.9394 kg mean propellant use, 0.3942 m mean absolute touchdown position, and 0.7802 m/s mean absolute vertical touchdown speed.

~~~powershell
python scripts/tune_integrated_controller.py --report artifacts/tuning.json --best-config artifacts/best-integrated.yaml
~~~

The JSON report contains all candidate evaluations, objective components, seeds, and the independent validation result. The YAML output contains the complete selected configuration plus selection metadata and passes the same configuration validation as the tracked default. Existing output files are never overwritten.

## 19. Isolated PID disturbance evaluation

`evaluate_pid_disturbances.py` applies the selected Day 12 gain scales `(1.10, 0.95, 1.10)` and changes one disturbance model at a time. Every strength within a curve reuses the same 30 initial conditions from seed 20260915. Random wind direction, gust start time, and sensor-noise stream are also paired by episode across strength levels. Zero-strength results are required to match across all five curves.

### 19.1 Wind and aerodynamic drag

Wind is expressed as a two-dimensional inertial air-velocity vector. With vehicle velocity `v` and air velocity `w`, the relative velocity and quadratic drag force are:

~~~text
v_relative = v - w
F_drag = -0.5 * air_density * drag_coefficient * reference_area
         * norm(v_relative) * v_relative
~~~

The force acts through the modeled center of mass and therefore produces no aerodynamic torque. It is evaluated at each RK4 stage. Passing no wind model preserves the earlier ideal dynamics exactly and applies no aerodynamic force.

A constant-wind episode holds its sampled positive or negative horizontal direction for the entire flight. A gust is a horizontal half-sine pulse with a two-second duration, a start time sampled uniformly from 2 to 6 seconds, and a direction sampled independently for each episode. The same direction and start time are reused when comparing amplitudes.

### 19.2 Sensor noise

Sensor noise is added only to the state observed by the controller; contact detection and physical propagation use the true state. Independent zero-mean Gaussian samples use the following one-times standard deviations: x and z `0.5 m`, vx and vz `0.1 m/s`, attitude and angular rate `0.25 deg` and `0.25 deg/s`, and mass `0.5 kg`. A dimensionless strength multiplies all seven values. Noisy altitude and mass are clipped at ground and dry mass so the observation remains inside the controller's physical domain.

### 19.3 Thrust error and engine lag

Thrust scale changes the simulator's actual maximum thrust while the controller retains its nominal model. Propellant flow uses the actual thrust and the unchanged specific impulse. A scale of `0.95` therefore means that every throttle setting produces 95% of the thrust expected by the controller.

Engine lag is a first-order throttle response integrated exactly over each 0.02-second simulation interval:

~~~text
response = 1 - exp(-dt / time_constant)
actual_throttle += response * (commanded_throttle - actual_throttle)
~~~

The initial actual throttle is zero. Gimbal remains immediate because gimbal-servo dynamics are not part of this disturbance. A zero time constant reproduces the command exactly.

### 19.4 Performance curves and collapse rule

The first tested nonzero disturbance strength with a safe-landing rate below 95% is reported as the collapse point. With 30 episodes per level, each case changes the observed success rate by 3.33 percentage points. Results at the measured collapse points were:

| Disturbance | Last passing strength | Collapse strength | Success rate | Outcomes at collapse |
|---|---:|---:|---:|---|
| Constant wind | 10 m/s | 20 m/s | 86.7% | 26 success, 4 crash |
| Gust amplitude | 40 m/s | 60 m/s | 86.7% | 26 success, 4 crash |
| Sensor-noise scale | 1x | 2x | 86.7% | 26 success, 4 crash |
| Actual thrust scale | 1.00x | 0.95x | 93.3% | 28 success, 2 crash |
| Throttle-lag time constant | 0.2 s | 0.5 s | 90.0% | 27 success, 3 crash |

The shared zero-disturbance result was 29 safe landings and one crash. This new seed exposes a boundary case that was absent from the separate Day 12 validation batch. The values above are tested grid thresholds, not exact physical limits or hardware certifications.

~~~powershell
python scripts/evaluate_pid_disturbances.py --report artifacts/disturbances.json --plot artifacts/disturbances.png
~~~

The JSON contains every level's outcome counts, terminal-state errors, normalized error, and propellant use. The PNG contains five safe-landing-rate curves, the 95% threshold, and the detected collapse strength. Existing files are never overwritten.

## 20. Frozen Week 2 baseline protocol

`configs/pid-baseline-v1.yaml` is the comparison baseline for all later methods. It materializes the Day 12 selected scales in the integrated-controller values. The approach-phase position gain, velocity gain, and profile deceleration are `0.55`, `1.615`, and `1.65 m/s^2`; the terminal values are `0.33`, `1.2825`, and `1.1 m/s^2`. The controller mapping has the frozen digest `80d4cbedbc545457072dc08f308ee408fd25996e3df500520135ec0452818a8a`.

The fixed evaluation set uses sampler `integrated_uniform_v1`, 1,000 episodes, and seed 20260914. The sampler draws the sign and magnitude of horizontal position separately so every state begins at least 3 m from the target. It then draws the remaining six state components independently from the ranges in the integrated-controller evaluation configuration. Angles are converted to radians before hashing or simulation.

`initial_condition_sha256` hashes a schema prefix, the two matrix dimensions as little-endian unsigned 64-bit integers, and the contiguous state matrix as little-endian IEEE 754 float64 values. The expected digest is `a9593a383acc0017738ed383a123ca743a32bce46c64d249b08ca577a448a952`. `frozen_baseline_initial_states` regenerates the matrix and verifies both the controller and state digests. It returns a read-only array and rejects controller edits, seed or range drift, a changed sampler, or a changed episode count.

The nominal evaluation produced the following result:

| Metric | Value |
|---|---:|
| Safe landings | 999 / 1,000 (99.9%) |
| Crash | 1 / 1,000 |
| Mean / maximum absolute touchdown x | 0.4098 / 1.3494 m |
| Mean / maximum absolute touchdown vx | 0.1656 / 0.5915 m/s |
| Mean / p95 / maximum absolute touchdown vz | 0.7802 / 0.7810 / 0.7814 m/s |
| Mean / maximum absolute touchdown attitude | 0.5991 / 3.2498 deg |
| Mean / maximum absolute touchdown angular rate | 1.3178 / 3.6500 deg/s |
| Mean propellant use | 51.4460 kg |

The 99.9% result passes the configured 70% Week 2 acceptance gate. The single crash exceeded the 1 m horizontal position limit; the reported maxima for horizontal speed, vertical speed, attitude, and angular rate remained inside their respective contact limits. This finite simulation result is not a universal robustness claim.

~~~powershell
python scripts/evaluate_frozen_baseline.py --output-dir artifacts/week2-baseline
~~~

The command writes a JSON benchmark report, two replayable episode logs, and two GIFs. The representative success is initial-condition index 0 under nominal wind. The representative failure reuses index 0 with a 40 m/s constant horizontal wind, which produces a 2.5911 m touchdown position error and a crash. This stress case demonstrates a known failure mode and is not counted in the nominal success rate. The command checks all destination names first and never overwrites an existing output.

## 21. Day 15 optimal-control teacher formulation

The seven-state teacher problem is defined in `LandingOptimalControlProblem` but the full planar problem is **not solved yet**. Day 16 adds a solver for its vertical subset only. The Day 15 model is a nominal, wind-free, sensor-perfect, instantaneous-actuator model. It uses the same planar state ordering, units, force direction, torque sign, variable-mass law, and landing thresholds as the simulator. The simulator's optional aerodynamic drag is absent because no wind model is supplied in nominal rollouts. The nonlinear-program equations do not clip actions or switch thrust off at dry mass; hard actuator bounds and a 1 kg propellant reserve keep feasible trajectories strictly inside that smooth region.

The state is `X = [x, z, vx, vz, theta, omega, m]` and the control is `U = [q, delta]`, where `q` is dimensionless throttle and `delta` is the gimbal angle in radians. The optimization uses `N = 100` piecewise-constant control intervals. Final time `T` is a decision variable in `[5, 25] s`, with mesh step `h = T/N`. The initial state is fixed to the supplied state. Units are SI throughout; `theta` and `delta` are positive clockwise toward `+x` under the project convention.

| Mathematical quantity | Equation or bound | Code input and output |
|---|---|---|
| Thrust | `F = T_max q` | `dynamics(X, U)` receives throttle `q`; returns a 7-vector derivative |
| Position | `dx/dt = vx`, `dz/dt = vz` | derivative elements 0-1 |
| Translation | `dvx/dt = F sin(theta + delta)/m`; `dvz/dt = F cos(theta + delta)/m - g` | derivative elements 2-3; matches `state_derivative` in the smooth feasible region |
| Rotation | `dtheta/dt = omega`; `domega/dt = -L F sin(delta)/I` | derivative elements 4-5; negative torque sign matches the simulator |
| Propellant | `dm/dt = -F/(Isp g0)` | derivative element 6 |
| Multiple-shooting defect | `X[k+1] - RK4(X[k], U[k], T/N) = 0` | `rk4_defects(states, controls, T)` returns an `N x 7` matrix |
| Path margins | `z-ground >= 0`; `m-dry_mass-1 kg >= 0`; `q_min <= q <= q_max`; `|delta| <= delta_max` | `path_margins(X, U)` returns six nonnegative-required values; state bounds also apply at the final node |
| Terminal landing | `z(T)=ground`; `|x(T)-target_x|<=1 m`; `|vx(T)|<=1 m/s`; `-2<=vz(T)<=0 m/s`; `|theta(T)|<=5 deg`; `|omega(T)|<=5 deg/s` | `terminal_violations(X_T)` returns six normalized nonnegative values; zero means each terminal bound holds |

The Stage A violation of the terminal-height equality is normalized by a fixed 1 m scale. The terminal descent constraint intentionally excludes upward ground contact. The simulator uses point contact and has no landing-leg or impact-load model. The path constraints apply at shooting nodes; a solver result must still be rolled out in the event-driven simulator because node feasibility alone cannot rule out between-node ground penetration or other transcription error.

The objective is staged rather than relying on a single large weighted penalty. In **Stage A**, initial state, dynamics, duration, actuator, altitude, and propellant constraints are hard; terminal constraints are temporarily relaxed. The solver minimizes the sum of squares of the six normalized positive terminal violations. Stage A passes only when the maximum normalized terminal violation is at most `0.001`. This gives a measurable feasibility target and prevents low fuel use from compensating for a missed landing. Stage B starts from that solution, imposes the terminal constraints as hard bounds, and minimizes:

~~~text
fuel_fraction = (m_initial - m_final) / (m_initial - dry_mass)
touchdown_error = mean([
    ((x_final - target_x) / x_limit)^2,
    (vx_final / vx_limit)^2,
    ((vz_final - target_vz) / vz_limit)^2,
    (theta_final / theta_limit)^2,
    (omega_final / omega_limit)^2,
])
smoothness = mean over k=1..N-1 of [
    ((q[k] - q[k-1]) / (q_max - q_min))^2
    + ((delta[k] - delta[k-1]) / (2 delta_max))^2
]
J = 1.0 * fuel_fraction + 0.1 * touchdown_error + 0.01 * smoothness
~~~

The target terminal vertical velocity is `-0.8 m/s`, within the simulator's `[-2, 0] m/s` safe-descending interval. Every cost term is dimensionless. Stage B's fuel fraction uses the propellant available in the particular initial state, not the vehicle's nominal 250 kg capacity. The weights are explicit initial modeling choices, not tuned or validated performance claims. A candidate is not a teacher trajectory until a solver reports convergence, constraint residuals are checked, and independent simulator rollout agrees.

## 22. Day 16 vertical CasADi/IPOPT teacher

Day 16 solves only the vertical subset of the Day 15 problem. Its state is `Y=[z,vz,m]`, control is throttle `q`, and dynamics are `dz/dt=vz`, `dvz/dt=T_max*q/m-g`, and `dm/dt=-T_max*q/(Isp*g0)`. The initial horizontal and attitude state must be zero. The 100 constant-control intervals, free final time in `[5,25] s`, dry-mass-plus-1-kg reserve, actuator bounds, and landing vertical-speed limit come from the same configuration as the planar formulation. Each interval has a CasADi RK4 continuity equality, making this direct multiple shooting rather than single shooting.

A constant-acceleration estimate supplies the initial duration and throttle profile; a numerical rollout supplies the initial state nodes. Stage A keeps the initial state, RK4 dynamics, duration, altitude, fuel, and throttle bounds hard. Two nonnegative terminal slacks permit temporary height and vertical-speed violations, and the solver minimizes their normalized squares. Stage B starts from the Stage A solution, fixes final height to ground, constrains final speed to `[-2,0] m/s`, and minimizes the Day 15 vertical restriction of fuel fraction, touchdown-speed error, and throttle change. A stage must meet the configured `0.001` normalized feasibility threshold before its output is accepted.

The JSON report records each IPOPT return status, iteration count, objective, maximum normalized hard and terminal violation, and each RK4 state defect in physical units. The optimized controls are separately integrated with the existing planar simulator at `0.02 s` steps. The report checks ground height, terminal descent speed, fuel reserve, minimum altitude, and disagreement at every optimization node; the PNG overlays this rollout and the NLP nodes. Solver or feasibility failures write a diagnostic JSON report without a plot. This independent nominal rollout does not establish performance under wind, sensor error, engine lag, or a population of initial states.

~~~bash
python -m pip install -e ".[optimization]"
python scripts/solve_vertical_landing.py --output-dir artifacts/day16-vertical
python scripts/solve_vertical_landing.py --initial-z 110 --initial-vz -18 --initial-mass 980 --output-dir artifacts/day16-custom
~~~

The command refuses to overwrite existing files. The nominal case converges with a `5.000 s` duration, about `-0.832 m/s` simulator terminal velocity, and about `27.435 kg` of fuel use. These are one-case outputs, not a general success-rate estimate. Day 17 will add horizontal translation; Day 18 will add rotation and gimbal torque.
