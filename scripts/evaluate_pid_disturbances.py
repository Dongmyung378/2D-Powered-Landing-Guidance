"""Measure integrated PID landing performance under isolated disturbances."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import matplotlib
import numpy as np
from numpy.typing import ArrayLike, NDArray

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

from powered_landing_guidance import IntegratedLandingController, load_config, validate_config
from powered_landing_guidance.envs import RocketLandingEnv

if __package__:
    from scripts.tune_integrated_controller import ScaleCandidate, apply_candidate
else:
    from tune_integrated_controller import ScaleCandidate, apply_candidate

from powered_landing_guidance.evaluation import sample_integrated_initial_states

type DisturbanceKind = Literal[
    "constant_wind",
    "gust",
    "sensor_noise",
    "thrust_scale",
    "engine_lag",
]

DISTURBANCE_ORDER: tuple[DisturbanceKind, ...] = (
    "constant_wind",
    "gust",
    "sensor_noise",
    "thrust_scale",
    "engine_lag",
)

DISTURBANCE_LABELS = {
    "constant_wind": ("Constant wind", "m/s"),
    "gust": ("Gust amplitude", "m/s"),
    "sensor_noise": ("Sensor noise scale", "x"),
    "thrust_scale": ("Actual thrust scale", "x"),
    "engine_lag": ("Throttle lag time constant", "s"),
}


@dataclass(frozen=True, slots=True)
class DisturbanceScenario:
    kind: DisturbanceKind
    strength: float


@dataclass(frozen=True, slots=True)
class DisturbanceLevelResult:
    strength: float
    episodes: int
    successes: int
    success_rate: float
    outcome_counts: dict[str, int]
    mean_abs_touchdown_x_m: float
    max_abs_touchdown_x_m: float
    mean_abs_touchdown_vx_m_s: float
    mean_abs_touchdown_vz_m_s: float
    max_abs_touchdown_vz_m_s: float
    mean_abs_touchdown_theta_deg: float
    mean_abs_touchdown_omega_deg_s: float
    mean_normalized_terminal_error: float
    mean_fuel_used_kg: float


@dataclass(frozen=True, slots=True)
class DisturbanceCurveResult:
    kind: DisturbanceKind
    label: str
    unit: str
    collapse_strength: float | None
    last_passing_strength: float
    levels: tuple[DisturbanceLevelResult, ...]


@dataclass(frozen=True, slots=True)
class DisturbanceEvaluationResult:
    seed: int
    episodes_per_level: int
    collapse_success_rate: float
    controller_scales: dict[str, float]
    curves: tuple[DisturbanceCurveResult, ...]


@dataclass(slots=True)
class FirstOrderThrottleLag:
    """First-order throttle actuator with an immediate gimbal channel."""

    time_constant_s: float
    throttle: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_constant_s) or self.time_constant_s < 0.0:
            raise ValueError("time_constant_s must be nonnegative and finite")

    def apply(self, command: ArrayLike, dt_s: float) -> NDArray[np.float64]:
        values = np.asarray(command, dtype=np.float64)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("command must be a finite two-element vector")
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        if self.time_constant_s == 0.0:
            self.throttle = float(values[0])
        else:
            response = -np.expm1(-dt_s / self.time_constant_s)
            self.throttle += float(response * (values[0] - self.throttle))
        return np.asarray((self.throttle, values[1]), dtype=np.float64)


def sensor_noise_standard_deviations(config: dict) -> NDArray[np.float64]:
    """Return configured sensor standard deviations in canonical SI state order."""
    values = config["disturbance_evaluation"]["sensor_noise"]["standard_deviations"]
    return np.asarray(
        (
            values["x_m"],
            values["z_m"],
            values["vx_m_s"],
            values["vz_m_s"],
            np.deg2rad(values["theta_deg"]),
            np.deg2rad(values["omega_deg_s"]),
            values["mass_kg"],
        ),
        dtype=np.float64,
    )


def add_sensor_noise(
    state: ArrayLike,
    standard_deviations: ArrayLike,
    scale: float,
    rng: np.random.Generator,
    *,
    ground_z_m: float,
    dry_mass_kg: float,
) -> NDArray[np.float64]:
    """Perturb a controller observation without changing the physical state."""
    values = np.asarray(state, dtype=np.float64)
    deviations = np.asarray(standard_deviations, dtype=np.float64)
    if values.shape != (7,) or deviations.shape != (7,):
        raise ValueError("state and standard deviations must have shape (7,)")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(deviations)):
        raise ValueError("state and standard deviations must be finite")
    if np.any(deviations < 0.0) or not np.isfinite(scale) or scale < 0.0:
        raise ValueError("noise deviations and scale must be nonnegative")
    if scale == 0.0:
        return values.copy()
    observed = values + rng.normal(0.0, deviations * scale)
    observed[1] = max(observed[1], ground_z_m)
    observed[6] = max(observed[6], dry_mass_kg)
    return observed


def half_sine_gust(
    amplitude_m_s: float,
    start_time_s: float,
    duration_s: float,
    direction: float,
) -> Callable[[float], NDArray[np.float64]]:
    """Return a finite-duration horizontal half-sine gust profile."""
    if amplitude_m_s < 0.0 or start_time_s < 0.0 or duration_s <= 0.0:
        raise ValueError("gust amplitude and start must be nonnegative; duration must be positive")
    if direction not in (-1.0, 1.0):
        raise ValueError("gust direction must be -1 or 1")

    def velocity(time_s: float) -> NDArray[np.float64]:
        phase = (time_s - start_time_s) / duration_s
        horizontal = (
            direction * amplitude_m_s * np.sin(np.pi * phase) if 0.0 <= phase <= 1.0 else 0.0
        )
        return np.asarray((horizontal, 0.0), dtype=np.float64)

    return velocity


def _controller_config(config: dict) -> tuple[dict, dict[str, float]]:
    selected = config["baseline_tuning"].get("selected")
    if selected is not None:
        scales = {key: float(value) for key, value in selected["candidate"].items()}
        return deepcopy(config), scales
    scales = {
        key: float(value)
        for key, value in config["disturbance_evaluation"]["controller_scales"].items()
    }
    candidate = ScaleCandidate(
        position_gain_scale=scales["position_gain_scale"],
        velocity_gain_scale=scales["velocity_gain_scale"],
        descent_profile_scale=scales["descent_profile_scale"],
    )
    return apply_candidate(config, candidate), scales


def disturbance_levels(config: dict, kind: DisturbanceKind) -> tuple[float, ...]:
    """Read ordered strengths for one disturbance curve."""
    settings = config["disturbance_evaluation"]
    if kind == "constant_wind":
        values = settings["constant_wind_m_s"]
    elif kind == "gust":
        values = settings["gust"]["amplitude_m_s"]
    elif kind == "sensor_noise":
        values = settings["sensor_noise"]["scales"]
    elif kind == "thrust_scale":
        values = settings["thrust_scale"]
    elif kind == "engine_lag":
        values = settings["engine_lag_s"]
    else:
        raise ValueError(f"unsupported disturbance kind: {kind}")
    return tuple(float(value) for value in values)


def evaluate_disturbance_level(
    config: dict,
    initial_states: NDArray[np.float64],
    scenario: DisturbanceScenario,
    *,
    seed: int,
) -> DisturbanceLevelResult:
    """Evaluate one isolated disturbance strength on a paired initial-state batch."""
    if initial_states.ndim != 2 or initial_states.shape[1] != 7 or len(initial_states) == 0:
        raise ValueError("initial_states must have shape (episodes, 7)")
    if scenario.kind not in DISTURBANCE_ORDER:
        raise ValueError(f"unsupported disturbance kind: {scenario.kind}")
    if not np.isfinite(scenario.strength) or scenario.strength < 0.0:
        raise ValueError("disturbance strength must be nonnegative and finite")
    if scenario.kind == "thrust_scale" and scenario.strength <= 0.0:
        raise ValueError("thrust scale must be positive")

    actual_config = deepcopy(config)
    if scenario.kind == "thrust_scale":
        actual_config["vehicle"]["max_thrust_n"] *= scenario.strength
        validate_config(actual_config)

    settings = config["disturbance_evaluation"]
    gust_settings = settings["gust"]
    sensor_deviations = sensor_noise_standard_deviations(config)
    rng = np.random.default_rng(seed)
    wind_directions = rng.choice(np.asarray((-1.0, 1.0)), size=len(initial_states))
    gust_starts = rng.uniform(*gust_settings["start_time_s"], size=len(initial_states))
    noise_seeds = rng.integers(
        0, np.iinfo(np.uint32).max, size=len(initial_states), dtype=np.uint32
    )

    outcomes: list[str] = []
    final_states: list[NDArray[np.float64]] = []
    fuel_used: list[float] = []
    dt_s = float(config["simulation"]["dt_s"])
    ground = float(config["simulation"]["ground_z_m"])
    dry_mass = float(config["vehicle"]["dry_mass_kg"])

    for index, initial_state in enumerate(initial_states):
        wind_velocity = None
        if scenario.kind == "constant_wind" and scenario.strength > 0.0:
            wind_velocity = np.asarray(
                (wind_directions[index] * scenario.strength, 0.0),
                dtype=np.float64,
            )
        elif scenario.kind == "gust" and scenario.strength > 0.0:
            wind_velocity = half_sine_gust(
                scenario.strength,
                float(gust_starts[index]),
                float(gust_settings["duration_s"]),
                float(wind_directions[index]),
            )

        env = RocketLandingEnv(actual_config, wind_velocity_m_s=wind_velocity)
        controller = IntegratedLandingController.from_config(config)
        throttle_lag = FirstOrderThrottleLag(
            scenario.strength if scenario.kind == "engine_lag" else 0.0
        )
        noise_rng = np.random.default_rng(noise_seeds[index])
        try:
            state, info = env.reset(options={"initial_state": initial_state})
            while True:
                observation = state
                if scenario.kind == "sensor_noise":
                    observation = add_sensor_noise(
                        state,
                        sensor_deviations,
                        scenario.strength,
                        noise_rng,
                        ground_z_m=ground,
                        dry_mass_kg=dry_mass,
                    )
                commanded = controller.command(observation, time_s=float(info["time_s"]))
                applied = throttle_lag.apply(commanded, dt_s)
                state, _, terminated, truncated, info = env.step(applied)
                if terminated or truncated:
                    break
        finally:
            env.close()

        outcomes.append(str(info["outcome"]))
        final_states.append(np.asarray(state, dtype=np.float64))
        fuel_used.append(float(info["fuel_used_kg"]))

    states = np.asarray(final_states)
    absolute = np.abs(states)
    counts = dict(sorted(Counter(outcomes).items()))
    successes = counts.get("success", 0)
    limits = config["landing_success"]
    limit_vector = np.asarray(
        (
            limits["max_abs_x_m"],
            limits["max_abs_vx_m_s"],
            limits["max_abs_vz_m_s"],
            np.deg2rad(limits["max_abs_theta_deg"]),
            np.deg2rad(limits["max_abs_omega_deg_s"]),
        ),
        dtype=np.float64,
    )
    terminal_values = absolute[:, (0, 2, 3, 4, 5)]
    return DisturbanceLevelResult(
        strength=scenario.strength,
        episodes=len(initial_states),
        successes=successes,
        success_rate=successes / len(initial_states),
        outcome_counts=counts,
        mean_abs_touchdown_x_m=float(np.mean(absolute[:, 0])),
        max_abs_touchdown_x_m=float(np.max(absolute[:, 0])),
        mean_abs_touchdown_vx_m_s=float(np.mean(absolute[:, 2])),
        mean_abs_touchdown_vz_m_s=float(np.mean(absolute[:, 3])),
        max_abs_touchdown_vz_m_s=float(np.max(absolute[:, 3])),
        mean_abs_touchdown_theta_deg=float(np.rad2deg(np.mean(absolute[:, 4]))),
        mean_abs_touchdown_omega_deg_s=float(np.rad2deg(np.mean(absolute[:, 5]))),
        mean_normalized_terminal_error=float(np.mean(terminal_values / limit_vector)),
        mean_fuel_used_kg=float(np.mean(fuel_used)),
    )


def evaluate_disturbances(
    config: dict,
    *,
    episodes_per_level: int | None = None,
) -> DisturbanceEvaluationResult:
    """Build all isolated performance curves with paired initial conditions."""
    settings = config.get("disturbance_evaluation")
    if not isinstance(settings, dict):
        raise ValueError("config must contain disturbance_evaluation settings")
    count = int(
        settings["episodes_per_level"] if episodes_per_level is None else episodes_per_level
    )
    if count <= 0:
        raise ValueError("episodes_per_level must be positive")
    seed = int(settings["seed"])
    collapse_rate = float(settings["collapse_success_rate"])
    controller_config, controller_scales = _controller_config(config)
    initial_states = sample_integrated_initial_states(controller_config, count, seed)

    curves: list[DisturbanceCurveResult] = []
    for kind in DISTURBANCE_ORDER:
        results: list[DisturbanceLevelResult] = []
        for strength in disturbance_levels(config, kind):
            result = evaluate_disturbance_level(
                controller_config,
                initial_states,
                DisturbanceScenario(kind, strength),
                seed=seed,
            )
            results.append(result)
            print(
                f"{kind}={strength:g}: {result.successes}/{result.episodes} "
                f"({result.success_rate:.1%}), error={result.mean_normalized_terminal_error:.4f}, "
                f"fuel={result.mean_fuel_used_kg:.4f} kg",
                flush=True,
            )

        baseline = results[0]
        if curves and (
            baseline.successes != curves[0].levels[0].successes
            or not np.isclose(
                baseline.mean_normalized_terminal_error,
                curves[0].levels[0].mean_normalized_terminal_error,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise RuntimeError("zero-strength baselines must match across isolated curves")
        collapse = next(
            (result.strength for result in results[1:] if result.success_rate < collapse_rate),
            None,
        )
        passing = [result.strength for result in results if result.success_rate >= collapse_rate]
        label, unit = DISTURBANCE_LABELS[kind]
        curves.append(
            DisturbanceCurveResult(
                kind=kind,
                label=label,
                unit=unit,
                collapse_strength=collapse,
                last_passing_strength=passing[-1],
                levels=tuple(results),
            )
        )

    return DisturbanceEvaluationResult(
        seed=seed,
        episodes_per_level=count,
        collapse_success_rate=collapse_rate,
        controller_scales=controller_scales,
        curves=tuple(curves),
    )


def save_disturbance_outputs(
    result: DisturbanceEvaluationResult,
    *,
    report_path: Path,
    plot_path: Path,
) -> None:
    """Save a complete JSON report and success-rate curves without overwrite."""
    if report_path.resolve() == plot_path.resolve():
        raise ValueError("report and plot paths must differ")
    if plot_path.suffix.lower() != ".png":
        raise ValueError("plot output must end in .png")
    for path in (report_path, plot_path):
        if path.exists():
            raise FileExistsError(f"output file already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(asdict(result), stream, ensure_ascii=False, indent=2, allow_nan=False)

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    flat_axes = axes.ravel()
    try:
        for axis, curve in zip(flat_axes, result.curves, strict=False):
            strengths = [level.strength for level in curve.levels]
            success_rates = [100.0 * level.success_rate for level in curve.levels]
            axis.plot(strengths, success_rates, marker="o", linewidth=2, color="tab:blue")
            axis.axhline(
                100.0 * result.collapse_success_rate,
                color="tab:red",
                linestyle="--",
                linewidth=1,
                label="collapse threshold",
            )
            if curve.collapse_strength is not None:
                axis.axvline(curve.collapse_strength, color="0.35", linestyle=":", linewidth=1)
            if curve.kind == "thrust_scale":
                axis.invert_xaxis()
            axis.set_title(curve.label)
            axis.set_xlabel(f"Strength ({curve.unit})")
            axis.set_ylabel("Safe landings (%)")
            axis.set_ylim(-2.0, 102.0)
            axis.grid(alpha=0.25)

        summary_axis = flat_axes[-1]
        summary_axis.axis("off")
        summary_lines = [
            f"Seed: {result.seed}",
            f"Episodes per level: {result.episodes_per_level}",
            f"Collapse threshold: {result.collapse_success_rate:.0%}",
            "",
        ]
        for curve in result.curves:
            collapse = (
                "not observed"
                if curve.collapse_strength is None
                else f"{curve.collapse_strength:g}"
            )
            summary_lines.append(f"{curve.label}: {collapse} {curve.unit}")
        summary_axis.text(0.0, 1.0, "\n".join(summary_lines), va="top", family="monospace")
        figure.suptitle("Integrated PID isolated-disturbance performance")

        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                prefix=f".{plot_path.stem}-",
                suffix=".png",
                dir=plot_path.parent,
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
            figure.savefig(temporary, dpi=150)
            with temporary.open("rb") as source, plot_path.open("xb") as target:
                shutil.copyfileobj(source, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    finally:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/default.yaml",
    )
    parser.add_argument("--episodes", type=int, help="override episodes per strength level")
    parser.add_argument("--report", type=Path, help="save the complete result as JSON")
    parser.add_argument("--plot", type=Path, help="save the success-rate curves as PNG")
    args = parser.parse_args()
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be positive")
    if (args.report is None) != (args.plot is None):
        parser.error("--report and --plot must be provided together")

    config = load_config(args.config)
    result = evaluate_disturbances(config, episodes_per_level=args.episodes)
    print("collapse points:")
    for curve in result.curves:
        collapse = (
            "not observed" if curve.collapse_strength is None else f"{curve.collapse_strength:g}"
        )
        print(
            f"- {curve.kind}: last passing={curve.last_passing_strength:g} {curve.unit}, "
            f"collapse={collapse} {curve.unit}"
        )

    if args.report is not None and args.plot is not None:
        save_disturbance_outputs(result, report_path=args.report, plot_path=args.plot)
        print(f"saved report: {args.report}")
        print(f"saved plot: {args.plot}")


if __name__ == "__main__":
    main()
