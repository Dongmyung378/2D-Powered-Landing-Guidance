"""Solve and validate one Day 18 full planar 3-DoF landing trajectory."""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict
from pathlib import Path

import casadi as ca
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from powered_landing_guidance import load_config  # noqa: E402
from powered_landing_guidance.optimal_control import LandingOptimalControlProblem  # noqa: E402
from powered_landing_guidance.planar_optimal_control import (  # noqa: E402
    PlanarLandingResult,
    PlanarSolveError,
    solve_planar_landing,
)


def save_plot(
    result: PlanarLandingResult,
    problem: LandingOptimalControlProblem,
    destination: Path,
) -> None:
    """Render the path, complete state, and physical controls."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
    try:
        path = axes[0, 0]
        path.plot(result.rollout_states[:, 0], result.rollout_states[:, 1], label="Simulator")
        path.plot(result.states[:, 0], result.states[:, 1], ".", ms=2, label="NLP nodes")
        path.scatter(
            [problem.target_x_m],
            [problem.ground_z_m],
            marker="x",
            color="black",
            label="Target",
        )
        path.set(xlabel="Horizontal position (m)", ylabel="Altitude (m)", title="Flight path")
        path.legend(loc="best")

        axes[0, 1].plot(result.rollout_times_s, result.rollout_states[:, 2], label="vx")
        axes[0, 1].plot(result.rollout_times_s, result.rollout_states[:, 3], label="vz")
        axes[0, 1].set(xlabel="Time (s)", ylabel="Velocity (m/s)", title="Velocity")
        axes[0, 1].legend(loc="best")

        axes[0, 2].plot(
            result.rollout_times_s,
            np.rad2deg(result.rollout_states[:, 4]),
        )
        axes[0, 2].axhline(np.rad2deg(problem.max_abs_tilt_rad), color="black", ls="--", lw=0.8)
        axes[0, 2].axhline(-np.rad2deg(problem.max_abs_tilt_rad), color="black", ls="--", lw=0.8)
        axes[0, 2].set(xlabel="Time (s)", ylabel="Attitude (deg)", title="Body attitude")

        axes[0, 3].plot(
            result.rollout_times_s,
            np.rad2deg(result.rollout_states[:, 5]),
        )
        axes[0, 3].axhline(
            np.rad2deg(problem.max_abs_angular_rate_rad_s), color="black", ls="--", lw=0.8
        )
        axes[0, 3].axhline(
            -np.rad2deg(problem.max_abs_angular_rate_rad_s), color="black", ls="--", lw=0.8
        )
        axes[0, 3].set(
            xlabel="Time (s)",
            ylabel="Angular rate (deg/s)",
            title="Angular rate",
        )

        control_times = result.times_s
        axes[1, 0].step(
            control_times,
            np.r_[result.controls[:, 0], result.controls[-1, 0]],
            where="post",
            color="tab:orange",
        )
        axes[1, 0].set(
            xlabel="Time (s)", ylabel="Throttle (0-1)", title="Throttle", ylim=(-0.05, 1.05)
        )

        axes[1, 1].step(
            control_times,
            np.rad2deg(np.r_[result.controls[:, 1], result.controls[-1, 1]]),
            where="post",
            color="tab:green",
        )
        limit_deg = np.rad2deg(problem.parameters.gimbal_limit_rad)
        axes[1, 1].axhline(limit_deg, color="black", ls="--", lw=0.8)
        axes[1, 1].axhline(-limit_deg, color="black", ls="--", lw=0.8)
        axes[1, 1].set(xlabel="Time (s)", ylabel="Gimbal (deg)", title="Gimbal")

        axes[1, 2].plot(result.rollout_times_s, result.rollout_states[:, 6])
        axes[1, 2].axhline(
            problem.parameters.dry_mass_kg + problem.min_propellant_reserve_kg,
            color="black",
            ls="--",
            lw=0.8,
        )
        axes[1, 2].set(xlabel="Time (s)", ylabel="Mass (kg)", title="Vehicle mass")

        axes[1, 3].plot(result.rollout_times_s, result.rollout_states[:, 0], label="x")
        axes[1, 3].plot(result.rollout_times_s, result.rollout_states[:, 1], label="z")
        axes[1, 3].set(xlabel="Time (s)", ylabel="Position (m)", title="Position")
        axes[1, 3].legend(loc="best")

        for axis in axes.flat:
            axis.grid(alpha=0.25)
        fig.suptitle("Day 18 full planar 3-DoF landing")
        image = io.BytesIO()
        fig.savefig(image, format="png", dpi=160)
        with destination.open("xb") as stream:
            stream.write(image.getvalue())
    finally:
        plt.close(fig)


def build_report(
    result: PlanarLandingResult,
    problem: LandingOptimalControlProblem,
) -> dict[str, object]:
    """Build the machine-readable constraint and simulator validation report."""
    final = result.rollout_states[-1]
    p = problem.parameters
    interpolated_nodes = np.column_stack(
        [
            np.interp(result.times_s, result.rollout_times_s, result.rollout_states[:, index])
            for index in range(7)
        ]
    )
    disagreement_by_state = np.max(np.abs(interpolated_nodes - result.states), axis=0)
    terminal_violations = problem.terminal_violations(final)
    height_error = abs(final[1] - problem.ground_z_m)
    max_tilt = float(np.max(np.abs(result.rollout_states[:, 4])))
    max_rate = float(np.max(np.abs(result.rollout_states[:, 5])))
    max_gimbal = float(np.max(np.abs(result.controls[:, 1])))
    verified = bool(
        np.max(terminal_violations) <= problem.feasibility_tolerance
        and height_error <= 1e-3
        and np.min(result.rollout_states[:, 1]) >= problem.ground_z_m - 1e-3
        and np.min(result.rollout_states[:, 6])
        >= p.dry_mass_kg + problem.min_propellant_reserve_kg - 1e-3
        and max_tilt <= problem.max_abs_tilt_rad + 1e-6
        and max_rate <= problem.max_abs_angular_rate_rad_s + 1e-6
        and max_gimbal <= p.gimbal_limit_rad + 1e-6
        and np.max(disagreement_by_state) <= 1e-3
    )
    return {
        "schema_version": 1,
        "problem": "day18-full-planar-3dof",
        "solver": "CasADi/IPOPT",
        "status": "validated" if verified else "rollout_failed",
        "casadi_version": ca.__version__,
        "initial_state_x_z_vx_vz_theta_omega_mass": problem.initial_state.as_array().tolist(),
        "intervals": problem.intervals,
        "duration_s": result.duration_s,
        "guess_source": result.guess_source,
        "source_outcome": result.source_outcome,
        "limits": {
            "max_abs_tilt_deg": float(np.rad2deg(problem.max_abs_tilt_rad)),
            "max_abs_gimbal_deg": float(np.rad2deg(p.gimbal_limit_rad)),
            "max_abs_angular_rate_deg_s": float(np.rad2deg(problem.max_abs_angular_rate_rad_s)),
        },
        "stage_a": asdict(result.stage_a),
        "stage_b": asdict(result.stage_b),
        "simulator_validation": {
            "passed": verified,
            "model": "full_planar_rigid_body_variable_mass",
            "final_state_x_z_vx_vz_theta_omega_mass": final.tolist(),
            "terminal_violations": dict(
                zip(
                    ("x", "z", "vx", "vz", "theta", "omega"),
                    terminal_violations.tolist(),
                    strict=True,
                )
            ),
            "fuel_used_kg": problem.initial_state.mass - final[6],
            "height_error_m": height_error,
            "min_altitude_m": float(np.min(result.rollout_states[:, 1])),
            "remaining_propellant_kg": float(final[6] - p.dry_mass_kg),
            "max_abs_tilt_deg": float(np.rad2deg(max_tilt)),
            "max_abs_angular_rate_deg_s": float(np.rad2deg(max_rate)),
            "max_abs_gimbal_deg": float(np.rad2deg(max_gimbal)),
            "max_node_disagreement_by_state": dict(
                zip(
                    ("x", "z", "vx", "vz", "theta", "omega", "mass"),
                    disagreement_by_state.tolist(),
                    strict=True,
                )
            ),
        },
        "trajectory": {
            "times_s": result.times_s.tolist(),
            "states_x_z_vx_vz_theta_omega_mass": result.states.tolist(),
            "controls_throttle_gimbal_rad": result.controls.tolist(),
            "rollout_times_s": result.rollout_times_s.tolist(),
            "rollout_states_x_z_vx_vz_theta_omega_mass": result.rollout_states.tolist(),
        },
    }


def _initial_state(args: argparse.Namespace, config: dict[str, object]) -> tuple[float, ...]:
    initial = config["initial_state"]
    return (
        args.initial_x,
        float(initial["z_m"] if args.initial_z is None else args.initial_z),
        float(initial["vx_m_s"] if args.initial_vx is None else args.initial_vx),
        float(initial["vz_m_s"] if args.initial_vz is None else args.initial_vz),
        float(np.deg2rad(args.initial_theta_deg)),
        float(np.deg2rad(args.initial_omega_deg_s)),
        float(initial["mass_kg"] if args.initial_mass is None else args.initial_mass),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/pid-baseline-v1.yaml",
    )
    parser.add_argument("--initial-x", type=float, default=10.0)
    parser.add_argument("--initial-z", type=float)
    parser.add_argument("--initial-vx", type=float)
    parser.add_argument("--initial-vz", type=float)
    parser.add_argument("--initial-theta-deg", type=float, default=3.0)
    parser.add_argument("--initial-omega-deg-s", type=float, default=-1.0)
    parser.add_argument("--initial-mass", type=float)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/day18-planar-3dof",
    )
    args = parser.parse_args()
    report_path = args.output_dir / "planar-3dof-teacher.json"
    plot_path = args.output_dir / "planar-3dof-teacher.png"
    for destination in (report_path, plot_path):
        if destination.exists():
            parser.error(f"refusing to overwrite existing output: {destination}")

    config = load_config(args.config)
    problem = LandingOptimalControlProblem.from_config(config, _initial_state(args, config))
    try:
        result = solve_planar_landing(
            problem,
            config,
            integration_step_s=config["simulation"]["dt_s"],
        )
    except PlanarSolveError as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema_version": 1,
                    "problem": "day18-full-planar-3dof",
                    "solver": "CasADi/IPOPT",
                    "status": "failed",
                    "failure": {
                        "stage": error.stage,
                        "solver_status": error.status,
                        "diagnostics": (
                            asdict(error.diagnostics) if error.diagnostics is not None else None
                        ),
                    },
                },
                stream,
                indent=2,
                allow_nan=False,
            )
        parser.exit(1, f"{error}; diagnostics saved to {report_path}\n")

    report = build_report(result, problem)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    save_plot(result, problem, plot_path)
    validation = report["simulator_validation"]
    final = result.rollout_states[-1]
    print(
        f"stage A={result.stage_a.status}, stage B={result.stage_b.status}, "
        f"duration={result.duration_s:.3f} s, final x={final[0]:.3f} m, "
        f"final vx={final[2]:.3f} m/s, final vz={final[3]:.3f} m/s, "
        f"final theta={np.rad2deg(final[4]):.3f} deg, "
        f"final omega={np.rad2deg(final[5]):.3f} deg/s, "
        f"full-model replay={validation['passed']}"
    )
    print(f"report={report_path}")
    print(f"plot={plot_path}")
    if not validation["passed"]:
        raise SystemExit("full-model simulator validation failed")


if __name__ == "__main__":
    main()
