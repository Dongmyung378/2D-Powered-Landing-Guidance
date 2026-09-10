# Model and Environment Specification

[Korean specification](spec.ko.md)

This document defines the state, control, coordinate, unit, event, and episode-record contracts shared by the simulator, controllers, datasets, and future learned policies. Any change to a sign or unit must update both the implementation and this specification.

## 1. Model scope

- 2D planar rigid body
- Translational degrees of freedom: `x`, `z`
- Rotational degree of freedom: `theta`
- Variable vehicle mass: `mass`
- Control inputs: throttle and thrust-vector gimbal angle
- Implemented effects: gravity, thrust, propellant consumption, and point-position ground contact
- Deferred effects: aerodynamic drag and wind
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
