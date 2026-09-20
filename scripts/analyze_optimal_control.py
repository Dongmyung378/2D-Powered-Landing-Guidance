"""Run the Day 19 objective trade-off and mesh-accuracy study."""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from powered_landing_guidance import load_config  # noqa: E402
from powered_landing_guidance.optimal_control_study import (  # noqa: E402
    OptimalControlStudyResult,
    run_optimal_control_study,
)
from powered_landing_guidance.planar_optimal_control import PlanarSolveError  # noqa: E402


def save_plot(result: OptimalControlStudyResult, destination: Path) -> None:
    """Plot normalized objective trade-offs and mesh timing/error trends."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    try:
        tradeoff = result.tradeoff_cases
        profile_names = [case.profile for case in tradeoff]
        fuel = [case.objective.fuel for case in tradeoff]
        smoothness = [case.objective.smoothness for case in tradeoff]
        axes[0, 0].plot(fuel, smoothness, "o-")
        for name, x_value, y_value in zip(profile_names, fuel, smoothness, strict=True):
            axes[0, 0].annotate(name, (x_value, y_value), xytext=(4, 4), textcoords="offset points")
        axes[0, 0].set(
            xlabel="Normalized fuel term",
            ylabel="Normalized smoothness term",
            title="Fuel and control smoothness",
        )

        positions = np.arange(len(tradeoff))
        width = 0.36
        axes[0, 1].bar(
            positions - width / 2,
            [
                case.max_throttle_rate_per_s / result.objective_scales.throttle_rate_per_s
                for case in tradeoff
            ],
            width,
            label="Throttle rate / 2 s^-1",
        )
        axes[0, 1].bar(
            positions + width / 2,
            [
                np.deg2rad(case.max_gimbal_rate_deg_s) / result.objective_scales.gimbal_rate_rad_s
                for case in tradeoff
            ],
            width,
            label="Gimbal rate / 60 deg s^-1",
        )
        axes[0, 1].set_xticks(positions, profile_names, rotation=15)
        axes[0, 1].set(title="Peak control rates", ylabel="Rate / reference rate")
        axes[0, 1].legend(loc="best")

        mesh = result.mesh_cases
        intervals = [case.intervals for case in mesh]
        axes[1, 0].plot(intervals, [case.solve_time_s for case in mesh], "o-")
        axes[1, 0].set(
            xlabel="Control intervals",
            ylabel="Wall time (s)",
            title="Mesh size and solve time",
        )

        axes[1, 1].semilogy(
            intervals,
            [case.max_final_error_ratio for case in mesh],
            "o-",
            label="Final-state error / tolerance",
        )
        axes[1, 1].semilogy(
            intervals,
            [case.max_node_error_ratio for case in mesh],
            "s-",
            label="Maximum node error / tolerance",
        )
        axes[1, 1].axhline(1.0, color="black", ls="--", lw=0.8, label="Acceptance limit")
        axes[1, 1].set(
            xlabel="Control intervals",
            ylabel="Normalized replay error",
            title="Mesh size and transcription error",
        )
        axes[1, 1].legend(loc="best")

        for axis in axes.flat:
            axis.grid(alpha=0.25)
        fig.suptitle("Day 19 objective and mesh study")
        image = io.BytesIO()
        fig.savefig(image, format="png", dpi=160)
        with destination.open("xb") as stream:
            stream.write(image.getvalue())
    finally:
        plt.close(fig)


def build_report(
    result: OptimalControlStudyResult,
    initial_state: tuple[float, ...],
) -> dict[str, object]:
    """Return a JSON-ready report with the Day 19 completion gate."""
    report = asdict(result)
    report.update(
        {
            "schema_version": 1,
            "problem": "day19-objective-and-mesh-study",
            "solver": "CasADi/IPOPT",
            "status": "validated" if result.all_replays_passed else "replay_failed",
            "initial_state_x_z_vx_vz_theta_omega_mass": list(initial_state),
            "completion_gate": {
                "criterion": "all final state differences are within configured tolerances",
                "passed": result.all_replays_passed,
            },
        }
    )
    return report


def _initial_state(args: argparse.Namespace, config: dict[str, Any]) -> tuple[float, ...]:
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
        default=Path(__file__).resolve().parents[1] / "artifacts/day19-study",
    )
    args = parser.parse_args()
    report_path = args.output_dir / "day19-study.json"
    plot_path = args.output_dir / "day19-study.png"
    for destination in (report_path, plot_path):
        if destination.exists():
            parser.error(f"refusing to overwrite existing output: {destination}")

    config = load_config(args.config)
    initial_state = _initial_state(args, config)
    try:
        result = run_optimal_control_study(config, initial_state)
    except PlanarSolveError as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema_version": 1,
                    "problem": "day19-objective-and-mesh-study",
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

    report = build_report(result, initial_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    save_plot(result, plot_path)

    print("objective trade-off:")
    for case in result.tradeoff_cases:
        print(
            f"  {case.profile}: fuel={case.fuel_used_kg:.3f} kg, "
            f"smoothness={case.objective.smoothness:.5f}, "
            f"throttle_rate={case.max_throttle_rate_per_s:.3f}/s, "
            f"gimbal_rate={case.max_gimbal_rate_deg_s:.3f} deg/s"
        )
    print("mesh study:")
    for case in result.mesh_cases:
        print(
            f"  N={case.intervals}: time={case.solve_time_s:.3f} s, "
            f"final_error_ratio={case.max_final_error_ratio:.3e}, "
            f"node_error_ratio={case.max_node_error_ratio:.3e}, "
            f"replay={case.replay_passed}"
        )
    print(f"completion gate={result.all_replays_passed}")
    print(f"report={report_path}")
    print(f"plot={plot_path}")
    if not result.all_replays_passed:
        raise SystemExit("one or more simulator replays exceeded the configured tolerance")


if __name__ == "__main__":
    main()
