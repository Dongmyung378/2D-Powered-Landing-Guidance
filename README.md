# Powered Landing Guidance

[Korean README](README.ko.md)

A reproducible 2D reusable-rocket powered-landing project that progresses from rigid-body simulation to classical control, optimal-control teaching, imitation learning, and robustness evaluation.

## Project snapshot

| Item | Current status |
|---|---|
| Roadmap | Week 2 classical-control stage |
| Model | Planar 3-DoF with variable mass |
| State | x, z, vx, vz, theta, omega, mass |
| Action | throttle, gimbal angle |
| Current controllers | Suicide-burn, vertical PID, horizontal-attitude, and integrated landing |
| Runtime | Python 3.12.7 |

The repository currently provides a tested simulation environment, two non-learning vertical landing baselines, cascaded horizontal-position and attitude control, and an integrated PID landing controller. Optimal control, Behavior Cloning, DAgger, disturbances, and Monte Carlo evaluation remain later roadmap stages.

## Integrated landing benchmark

The integrated controller combines vertical throttle with horizontal-attitude gimbal control and runs at 10 Hz while the simulator runs at 50 Hz. Commands are held between controller updates. Approach and terminal phases use separate velocity profiles and gains, switching at 30 m altitude. Throttle slew is limited to 2.0 per second and gimbal slew to 60 degrees per second.

A fixed batch of 100 medium-difficulty, wind-free initial conditions covered 3 to 15 m horizontal error, 80 to 120 m altitude, -2 to 2 m/s horizontal speed, -25 to -15 m/s vertical speed, -5 to 5 degrees attitude, -2 to 2 deg/s angular rate, and 950 to 1000 kg mass.

| Metric | Result |
|---|---:|
| Safe landings | 100 / 100 |
| Mean / maximum absolute touchdown position | 0.4516 / 0.9440 m |
| Mean / maximum horizontal touchdown speed | 0.1476 / 0.3526 m/s |
| Mean / maximum vertical touchdown speed | 0.7819 / 0.7827 m/s |
| Mean / maximum touchdown attitude | 0.7559 / 1.4386 degrees |
| Mean / maximum touchdown angular rate | 1.1004 / 2.4399 deg/s |
| Mean propellant use | 53.0159 kg |
| Maximum observed throttle / gimbal slew | 1.7784 per second / 60.0000 deg/s |

Every trial satisfied all configured landing limits. This is a deterministic wind-free benchmark, not a robustness or hardware-safety result.

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
scripts/                        Reproducible experiment entry points
src/powered_landing_guidance/   Physics, environment, controllers, and visualization
tests/                          Regression, boundary, and controller tests
.local/experiment_logs/         Local daily notes excluded from Git
~~~

The tracked repository contains only source code, reproducible configuration, technical documentation, and tests. Daily work notes stay on the development machine under .local/experiment_logs/ and are intentionally absent from GitHub.

## Roadmap

- Week 1: simulator, event handling, logging, replay, and numerical verification - complete
- Week 2: suicide-burn and PID baselines - in progress
- Week 3: constrained optimal-control teacher
- Week 4: dataset generation and Behavior Cloning
- Week 5: DAgger closed-loop improvement
- Week 6: disturbances and Monte Carlo evaluation
- Week 7: report, comparison visuals, and interactive demo

## Current limitations

- Terminal powered descent only; launch, ascent, boost-back, and reentry are out of scope.
- The integrated controller is validated only in deterministic, wind-free simulation.
- Aerodynamic drag, wind, sensor noise, thrust error, and engine delay are not active yet.
- Ground contact uses a point model without landing-leg or structural-impact dynamics.
- The 2 m/s success threshold is a simulation criterion, not a hardware safety guarantee.

See [the model and environment specification](docs/spec.md) for coordinate, unit, event, and episode-log rules. A [Korean specification](docs/spec.ko.md) is also available.
