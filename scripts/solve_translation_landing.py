"""Solve and plot one Day 17 horizontal-correction landing trajectory."""

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
from powered_landing_guidance.translation_optimal_control import (  # noqa: E402
    TranslationLandingResult,
    TranslationSolveError,
    solve_translation_landing,
)


def save_plot(
    result: TranslationLandingResult,
    problem: LandingOptimalControlProblem,
    destination: Path,
) -> None:
    """Render the 2D flight path and state/control histories."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    try:
        path = axes[0, 0]
        path.plot(result.rollout_states[:, 0], result.rollout_states[:, 1], label="Simulator")
        path.plot(result.states[:, 0], result.states[:, 1], ".", ms=2, label="NLP nodes")
        path.scatter(
            [problem.target_x_m], [problem.ground_z_m], marker="x", color="black", label="Target"
        )
        path.set(xlabel="Horizontal position (m)", ylabel="Altitude (m)", title="Flight path")
        path.legend(loc="best")

        for axis, index, label in (
            (axes[0, 1], 0, "Horizontal position (m)"),
            (axes[0, 2], 2, "Horizontal velocity (m/s)"),
            (axes[1, 0], 3, "Vertical velocity (m/s)"),
        ):
            axis.plot(result.rollout_times_s, result.rollout_states[:, index])
            axis.plot(result.times_s, result.states[:, index], ".", ms=2)
            axis.set(xlabel="Time (s)", ylabel=label)

        axes[1, 1].step(
            result.times_s,
            np.r_[result.controls[:, 0], result.controls[-1, 0]],
            where="post",
            color="tab:orange",
        )
        axes[1, 1].set(xlabel="Time (s)", ylabel="Throttle (0-1)", ylim=(-0.05, 1.05))
        axes[1, 2].step(
            result.times_s,
            np.rad2deg(np.r_[result.controls[:, 1], result.controls[-1, 1]]),
            where="post",
            color="tab:green",
        )
        axes[1, 2].set(xlabel="Time (s)", ylabel="Thrust angle (deg)")
        for axis in axes.flat:
            axis.grid(alpha=0.25)
        fig.suptitle(f"Day 17 ideal-vector landing - {result.guess_source} guess")
        image = io.BytesIO()
        fig.savefig(image, format="png", dpi=160)
        with destination.open("xb") as stream:
            stream.write(image.getvalue())
    finally:
        plt.close(fig)


def build_report(
    result: TranslationLandingResult,
    problem: LandingOptimalControlProblem,
) -> dict[str, object]:
    """Record solver residuals and the ideal-vector simulator replay check."""
    final = result.rollout_states[-1]
    p = problem.parameters
    x_limit, vx_limit, vz_limit = problem.terminal_limits[:3]
    interpolated_nodes = np.column_stack(
        [
            np.interp(result.times_s, result.rollout_times_s, result.rollout_states[:, index])
            for index in range(5)
        ]
    )
    disagreement = float(np.max(np.abs(interpolated_nodes - result.states)))
    height_error = abs(final[1] - problem.ground_z_m)
    verified = bool(
        height_error <= 1e-3
        and abs(final[0] - problem.target_x_m) <= x_limit + 1e-3
        and abs(final[2]) <= vx_limit + 1e-3
        and -vz_limit - 1e-3 <= final[3] <= 1e-3
        and np.min(result.rollout_states[:, 1]) >= problem.ground_z_m - 1e-3
        and np.min(result.rollout_states[:, 4])
        >= p.dry_mass_kg + problem.min_propellant_reserve_kg - 1e-3
        and disagreement <= 1e-3
    )
    return {
        "schema_version": 1,
        "problem": "day17-ideal-vector-planar-translation",
        "solver": "CasADi/IPOPT",
        "status": "validated" if verified else "rollout_failed",
        "casadi_version": ca.__version__,
        "initial_state_x_z_vx_vz_mass": [
            problem.initial_state.x,
            problem.initial_state.z,
            problem.initial_state.vx,
            problem.initial_state.vz,
            problem.initial_state.mass,
        ],
        "intervals": problem.intervals,
        "duration_s": result.duration_s,
        "guess_source": result.guess_source,
        "pid_outcome": result.pid_outcome,
        "max_abs_thrust_angle_deg": float(np.rad2deg(problem.max_abs_thrust_angle_rad)),
        "stage_a": asdict(result.stage_a),
        "stage_b": asdict(result.stage_b),
        "simulator_validation": {
            "passed": verified,
            "model": "ideal_vector_heading_reset_each_control_interval",
            "final_state_x_z_vx_vz_mass": final.tolist(),
            "horizontal_error_reduction_m": (
                abs(problem.initial_state.x - problem.target_x_m)
                - abs(final[0] - problem.target_x_m)
            ),
            "fuel_used_kg": problem.initial_state.mass - final[4],
            "height_error_m": height_error,
            "min_altitude_m": float(np.min(result.rollout_states[:, 1])),
            "remaining_propellant_kg": float(final[4] - p.dry_mass_kg),
            "max_node_disagreement": disagreement,
        },
        "trajectory": {
            "times_s": result.times_s.tolist(),
            "states_x_z_vx_vz_mass": result.states.tolist(),
            "controls_throttle_angle_rad": result.controls.tolist(),
            "rollout_times_s": result.rollout_times_s.tolist(),
            "rollout_states_x_z_vx_vz_mass": result.rollout_states.tolist(),
        },
    }


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
    parser.add_argument("--initial-mass", type=float)
    parser.add_argument("--guess", choices=("analytic", "pid"), default="analytic")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/day17-translation",
    )
    args = parser.parse_args()
    report_path = args.output_dir / "translation-teacher.json"
    plot_path = args.output_dir / "translation-teacher.png"
    for destination in (report_path, plot_path):
        if destination.exists():
            parser.error(f"refusing to overwrite existing output: {destination}")

    config = load_config(args.config)
    initial = config["initial_state"]
    state = (
        args.initial_x,
        float(initial["z_m"] if args.initial_z is None else args.initial_z),
        float(initial["vx_m_s"] if args.initial_vx is None else args.initial_vx),
        float(initial["vz_m_s"] if args.initial_vz is None else args.initial_vz),
        0.0,
        0.0,
        float(initial["mass_kg"] if args.initial_mass is None else args.initial_mass),
    )
    problem = LandingOptimalControlProblem.from_config(config, state)
    try:
        result = solve_translation_landing(
            problem,
            guess_source=args.guess,
            pid_config=config if args.guess == "pid" else None,
            integration_step_s=config["simulation"]["dt_s"],
        )
    except TranslationSolveError as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema_version": 1,
                    "problem": "day17-ideal-vector-planar-translation",
                    "solver": "CasADi/IPOPT",
                    "status": "failed",
                    "guess_source": args.guess,
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
    print(
        f"stage A={result.stage_a.status}, stage B={result.stage_b.status}, "
        f"duration={result.duration_s:.3f} s, final x={result.rollout_states[-1, 0]:.3f} m, "
        f"final vx={result.rollout_states[-1, 2]:.3f} m/s, "
        f"final vz={result.rollout_states[-1, 3]:.3f} m/s, "
        f"ideal-vector replay={validation['passed']}"
    )
    print(f"report={report_path}")
    print(f"plot={plot_path}")
    if not validation["passed"]:
        raise SystemExit("ideal-vector simulator validation failed")


if __name__ == "__main__":
    main()
