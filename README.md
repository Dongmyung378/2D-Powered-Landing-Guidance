# Powered Landing Guidance

[한국어 README](README.ko.md)

A reproducible 2D reusable-rocket powered-landing project that progresses from rigid-body simulation to classical control, optimal-control teaching, imitation learning, and robustness evaluation.

## Project snapshot

| Item | Current status |
|---|---|
| Roadmap | Week 2 classical-control stage |
| Model | Planar 3-DoF with variable mass |
| State | x, z, vx, vz, theta, omega, mass |
| Action | throttle, gimbal angle |
| Current controllers | Suicide-burn and vertical-velocity PID |
| Runtime | Python 3.12.7 |

The repository currently provides a tested simulation environment and two non-learning vertical landing baselines. Horizontal and attitude control, optimal control, Behavior Cloning, DAgger, disturbances, and Monte Carlo evaluation are scheduled for later roadmap stages.

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

## Run the vertical landing controllers

Run the velocity-profile PID controller:

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller velocity-pid
~~~

Run the suicide-burn controller:

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller suicide-burn
~~~

The initial-state order is x z vx vz theta omega mass. Values use SI units; theta and omega use rad and rad/s.

Save the episode, diagnostic plot, and animation:

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller velocity-pid --output artifacts/velocity-pid.json --plot artifacts/velocity-pid.png --animation artifacts/velocity-pid.gif
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

## Repository structure

~~~text
configs/                        Shared experiment configuration
docs/                           Model and interface specification
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
- The current controllers are vertical-only and do not correct horizontal position or attitude.
- Aerodynamic drag, wind, sensor noise, thrust error, and engine delay are not active yet.
- Ground contact uses a point model without landing-leg or structural-impact dynamics.
- The 2 m/s success threshold is a simulation criterion, not a hardware safety guarantee.

See [the model and environment specification](docs/spec.md) for coordinate, unit, event, and episode-log rules.
