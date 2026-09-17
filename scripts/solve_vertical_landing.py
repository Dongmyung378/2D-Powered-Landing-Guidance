"""Solve and plot one Day 16 vertical landing trajectory."""

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
from powered_landing_guidance.vertical_optimal_control import (  # noqa: E402
    VerticalLandingResult,
    VerticalSolveError,
    solve_vertical_landing,
)


def save_plot(result: VerticalLandingResult, destination: Path) -> None:
    """Compare optimization nodes with the independent simulator rollout."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, layout="constrained")
    labels = (("Altitude", "m"), ("Vertical velocity", "m/s"), ("Mass", "kg"))
    try:
        for axis, index, (name, unit) in zip(axes.flat[:3], range(3), labels, strict=True):
            axis.plot(result.rollout_times_s, result.rollout_states[:, index], label="Simulator")
            axis.plot(result.times_s, result.states[:, index], ".", ms=2, label="NLP nodes")
            axis.set_ylabel(f"{name} ({unit})")
            axis.grid(alpha=0.25)
        axes[0, 0].legend(loc="best")
        axes[1, 1].step(
            result.times_s,
            np.r_[result.throttle, result.throttle[-1]],
            where="post",
            color="tab:orange",
        )
        axes[1, 1].set_ylabel("Throttle (0-1)")
        axes[1, 1].set_ylim(-0.05, 1.05)
        axes[1, 1].grid(alpha=0.25)
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 1].set_xlabel("Time (s)")
        fig.suptitle("Day 16 vertical optimal landing")
        image = io.BytesIO()
        fig.savefig(image, format="png", dpi=160)
        with destination.open("xb") as stream:
            stream.write(image.getvalue())
    finally:
        plt.close(fig)


def build_report(
    result: VerticalLandingResult,
    problem: LandingOptimalControlProblem,
) -> dict[str, object]:
    """Capture solver diagnostics, controls, and independent rollout checks."""
    final = result.rollout_states[-1]
    p = problem.parameters
    height_error = abs(final[0] - problem.ground_z_m)
    vz_limit = float(problem.terminal_limits[2])
    interpolated_nodes = np.column_stack(
        [
            np.interp(result.times_s, result.rollout_times_s, result.rollout_states[:, index])
            for index in range(3)
        ]
    )
    max_node_disagreement = float(np.max(np.abs(interpolated_nodes - result.states)))
    verified = bool(
        height_error <= 1e-3
        and -vz_limit - 1e-3 <= final[1] <= 1e-3
        and np.min(result.rollout_states[:, 0]) >= problem.ground_z_m - 1e-3
        and np.min(result.rollout_states[:, 2])
        >= p.dry_mass_kg + problem.min_propellant_reserve_kg - 1e-3
        and max_node_disagreement <= 1e-3
    )
    return {
        "schema_version": 1,
        "problem": "day16-vertical-multiple-shooting",
        "solver": "CasADi/IPOPT",
        "status": "validated" if verified else "rollout_failed",
        "casadi_version": ca.__version__,
        "initial_state": [
            problem.initial_state.z,
            problem.initial_state.vz,
            problem.initial_state.mass,
        ],
        "intervals": problem.intervals,
        "duration_s": result.duration_s,
        "stage_a": asdict(result.stage_a),
        "stage_b": asdict(result.stage_b),
        "simulator_validation": {
            "passed": verified,
            "final_state": final.tolist(),
            "fuel_used_kg": problem.initial_state.mass - final[2],
            "height_error_m": height_error,
            "min_altitude_m": float(np.min(result.rollout_states[:, 0])),
            "min_propellant_reserve_kg": float(np.min(result.rollout_states[:, 2] - p.dry_mass_kg)),
            "max_node_disagreement": max_node_disagreement,
        },
        "trajectory": {
            "times_s": result.times_s.tolist(),
            "states_z_vz_mass": result.states.tolist(),
            "throttle": result.throttle.tolist(),
            "rollout_times_s": result.rollout_times_s.tolist(),
            "rollout_states_z_vz_mass": result.rollout_states.tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/default.yaml",
    )
    parser.add_argument("--initial-z", type=float, help="initial altitude in metres")
    parser.add_argument("--initial-vz", type=float, help="initial vertical velocity in m/s")
    parser.add_argument("--initial-mass", type=float, help="initial mass in kilograms")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/day16-vertical",
    )
    args = parser.parse_args()
    report_path = args.output_dir / "vertical-teacher.json"
    plot_path = args.output_dir / "vertical-teacher.png"
    for destination in (report_path, plot_path):
        if destination.exists():
            parser.error(f"refusing to overwrite existing output: {destination}")

    config = load_config(args.config)
    initial = config["initial_state"]
    state = (
        0.0,
        float(initial["z_m"] if args.initial_z is None else args.initial_z),
        0.0,
        float(initial["vz_m_s"] if args.initial_vz is None else args.initial_vz),
        0.0,
        0.0,
        float(initial["mass_kg"] if args.initial_mass is None else args.initial_mass),
    )
    problem = LandingOptimalControlProblem.from_config(config, state)
    try:
        result = solve_vertical_landing(problem, integration_step_s=config["simulation"]["dt_s"])
    except VerticalSolveError as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema_version": 1,
                    "problem": "day16-vertical-multiple-shooting",
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
    save_plot(result, plot_path)
    validation = report["simulator_validation"]
    print(
        f"stage A={result.stage_a.status}, stage B={result.stage_b.status}, "
        f"duration={result.duration_s:.3f} s, final vz={result.rollout_states[-1, 1]:.3f} m/s, "
        f"fuel={validation['fuel_used_kg']:.3f} kg, simulator check={validation['passed']}"
    )
    print(f"report={report_path}")
    print(f"plot={plot_path}")
    if not validation["passed"]:
        raise SystemExit("independent simulator validation failed")


if __name__ == "__main__":
    main()
