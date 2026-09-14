"""Tune integrated-controller scales with separate train and validation batches."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from powered_landing_guidance import load_config, validate_config

if __package__:
    from scripts.evaluate_integrated_control import (
        IntegratedControlResult,
        evaluate_integrated_control,
        sample_integrated_initial_states,
    )
else:
    from evaluate_integrated_control import (
        IntegratedControlResult,
        evaluate_integrated_control,
        sample_integrated_initial_states,
    )

type SearchMethod = Literal["grid", "random"]


@dataclass(frozen=True, slots=True)
class ScaleCandidate:
    position_gain_scale: float
    velocity_gain_scale: float
    descent_profile_scale: float


@dataclass(frozen=True, slots=True)
class ObjectiveBreakdown:
    score: float
    failure_term: float
    fuel_term: float
    landing_error_term: float
    normalized_landing_error: float


@dataclass(frozen=True, slots=True)
class CandidateResult:
    candidate: ScaleCandidate
    objective: ObjectiveBreakdown
    evaluation: IntegratedControlResult


@dataclass(frozen=True, slots=True)
class TuningResult:
    method: SearchMethod
    train_seed: int
    validation_seed: int
    train_episodes: int
    validation_episodes: int
    candidates_evaluated: int
    train_results: tuple[CandidateResult, ...]
    best_train_result: CandidateResult
    validation_result: IntegratedControlResult


def _baseline_candidate() -> ScaleCandidate:
    return ScaleCandidate(1.0, 1.0, 1.0)


def grid_candidates(config: dict) -> list[ScaleCandidate]:
    """Return the unique Cartesian grid plus the unscaled baseline."""
    grid = config["baseline_tuning"]["search"]["grid"]
    candidates = {
        ScaleCandidate(float(position), float(velocity), float(profile))
        for position, velocity, profile in product(
            grid["position_gain_scale"],
            grid["velocity_gain_scale"],
            grid["descent_profile_scale"],
        )
    }
    candidates.add(_baseline_candidate())
    return sorted(
        candidates,
        key=lambda item: (
            item.position_gain_scale,
            item.velocity_gain_scale,
            item.descent_profile_scale,
        ),
    )


def random_candidates(config: dict, seed: int) -> list[ScaleCandidate]:
    """Return the baseline followed by reproducible uniform random candidates."""
    search = config["baseline_tuning"]["search"]
    ranges = search["random"]
    rng = np.random.default_rng(seed)
    candidates = [_baseline_candidate()]
    for _ in range(int(search["random_candidates"])):
        candidates.append(
            ScaleCandidate(
                position_gain_scale=float(rng.uniform(*ranges["position_gain_scale"])),
                velocity_gain_scale=float(rng.uniform(*ranges["velocity_gain_scale"])),
                descent_profile_scale=float(rng.uniform(*ranges["descent_profile_scale"])),
            )
        )
    return candidates


def apply_candidate(config: dict, candidate: ScaleCandidate) -> dict:
    """Return an isolated configuration with one candidate applied to both phases."""
    tuned = deepcopy(config)
    phases = tuned["integrated_landing_controller"]["phases"]
    for phase in phases.values():
        outer = phase["horizontal_outer_loop"]
        outer["position_kp_s2"] = round(
            outer["position_kp_s2"] * candidate.position_gain_scale,
            12,
        )
        outer["velocity_kd_s"] = round(
            outer["velocity_kd_s"] * candidate.velocity_gain_scale,
            12,
        )
        profile = phase["vertical_profile"]
        profile["deceleration_m_s2"] = round(
            profile["deceleration_m_s2"] * candidate.descent_profile_scale,
            12,
        )
    validate_config(tuned)
    return tuned


def objective_breakdown(config: dict, result: IntegratedControlResult) -> ObjectiveBreakdown:
    """Combine failure rate, propellant use, and normalized touchdown error."""
    weights = config["baseline_tuning"]["objective"]
    limits = config["landing_success"]
    normalized_landing_error = float(
        np.mean(
            (
                result.mean_abs_touchdown_x_m / limits["max_abs_x_m"],
                result.mean_abs_touchdown_vx_m_s / limits["max_abs_vx_m_s"],
                result.mean_abs_touchdown_vz_m_s / limits["max_abs_vz_m_s"],
                result.mean_abs_touchdown_theta_deg / limits["max_abs_theta_deg"],
                result.mean_abs_touchdown_omega_deg_s / limits["max_abs_omega_deg_s"],
            )
        )
    )
    propellant_capacity = float(config["vehicle"]["initial_mass_kg"]) - float(
        config["vehicle"]["dry_mass_kg"]
    )
    failure_term = float(weights["failure_penalty"]) * (1.0 - result.success_rate)
    fuel_term = float(weights["fuel_weight"]) * result.mean_fuel_used_kg / propellant_capacity
    landing_error_term = float(weights["landing_error_weight"]) * normalized_landing_error
    return ObjectiveBreakdown(
        score=failure_term + fuel_term + landing_error_term,
        failure_term=failure_term,
        fuel_term=fuel_term,
        landing_error_term=landing_error_term,
        normalized_landing_error=normalized_landing_error,
    )


def tune_controller(
    config: dict,
    *,
    method: SearchMethod | None = None,
    train_episodes: int | None = None,
    validation_episodes: int | None = None,
) -> tuple[TuningResult, dict]:
    """Select on the train batch and evaluate the winner once on validation."""
    search = config["baseline_tuning"]["search"]
    selected_method: SearchMethod = search["method"] if method is None else method
    train_count = int(search["train_episodes"] if train_episodes is None else train_episodes)
    validation_count = int(
        search["validation_episodes"] if validation_episodes is None else validation_episodes
    )
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("train and validation episode counts must be positive")
    train_seed = int(search["train_seed"])
    validation_seed = int(search["validation_seed"])
    train_states = sample_integrated_initial_states(config, train_count, train_seed)
    validation_states = sample_integrated_initial_states(
        config,
        validation_count,
        validation_seed,
    )
    if np.array_equal(train_states, validation_states):
        raise RuntimeError("train and validation initial states must differ")

    if selected_method == "grid":
        candidates = grid_candidates(config)
    elif selected_method == "random":
        candidates = random_candidates(config, train_seed)
    else:
        raise ValueError(f"unsupported search method: {selected_method}")

    trial_results: list[CandidateResult] = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_config = apply_candidate(config, candidate)
        evaluation = evaluate_integrated_control(candidate_config, train_states)
        objective = objective_breakdown(config, evaluation)
        trial_results.append(CandidateResult(candidate, objective, evaluation))
        print(
            f"candidate {index}/{len(candidates)}: score={objective.score:.6f}, "
            f"success={evaluation.successes}/{evaluation.episodes}, "
            f"fuel={evaluation.mean_fuel_used_kg:.4f} kg, "
            f"scales=({candidate.position_gain_scale:.4f}, "
            f"{candidate.velocity_gain_scale:.4f}, "
            f"{candidate.descent_profile_scale:.4f})",
            flush=True,
        )

    best_train_result = min(
        trial_results,
        key=lambda trial: (
            trial.objective.score,
            trial.candidate.position_gain_scale,
            trial.candidate.velocity_gain_scale,
            trial.candidate.descent_profile_scale,
        ),
    )
    best_config = apply_candidate(config, best_train_result.candidate)
    validation_result = evaluate_integrated_control(best_config, validation_states)
    result = TuningResult(
        method=selected_method,
        train_seed=train_seed,
        validation_seed=validation_seed,
        train_episodes=train_count,
        validation_episodes=validation_count,
        candidates_evaluated=len(candidates),
        train_results=tuple(trial_results),
        best_train_result=best_train_result,
        validation_result=validation_result,
    )
    best_config["baseline_tuning"]["selected"] = {
        "method": selected_method,
        "train_seed": train_seed,
        "validation_seed": validation_seed,
        "candidate": asdict(best_train_result.candidate),
        "train_objective": asdict(best_train_result.objective),
        "validation_success_rate": validation_result.success_rate,
    }
    return result, best_config


def save_tuning_outputs(
    result: TuningResult,
    best_config: dict,
    *,
    report_path: Path,
    config_path: Path,
) -> None:
    """Save a machine-readable report and loadable best configuration without overwrite."""
    if report_path.resolve() == config_path.resolve():
        raise ValueError("report and config paths must differ")
    for path in (report_path, config_path):
        if path.exists():
            raise FileExistsError(f"output file already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(asdict(result), stream, ensure_ascii=False, indent=2, allow_nan=False)
    with config_path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(best_config, stream, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/default.yaml",
    )
    parser.add_argument("--method", choices=("grid", "random"))
    parser.add_argument("--train-episodes", type=int)
    parser.add_argument("--validation-episodes", type=int)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--best-config", type=Path)
    args = parser.parse_args()
    if (args.report is None) != (args.best_config is None):
        parser.error("--report and --best-config must be provided together")

    config = load_config(args.config)
    result, best_config = tune_controller(
        config,
        method=args.method,
        train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
    )
    candidate = result.best_train_result.candidate
    validation = result.validation_result
    print(
        "best scales: "
        f"position={candidate.position_gain_scale:.6f}, "
        f"velocity={candidate.velocity_gain_scale:.6f}, "
        f"profile={candidate.descent_profile_scale:.6f}"
    )
    print(f"train objective: {result.best_train_result.objective.score:.6f}")
    print(
        f"validation: {validation.successes}/{validation.episodes} "
        f"({validation.success_rate:.1%}), fuel={validation.mean_fuel_used_kg:.4f} kg, "
        f"mean |x|={validation.mean_abs_touchdown_x_m:.4f} m"
    )
    if args.report is not None and args.best_config is not None:
        save_tuning_outputs(
            result,
            best_config,
            report_path=args.report,
            config_path=args.best_config,
        )
        print(f"saved report: {args.report}")
        print(f"saved best config: {args.best_config}")


if __name__ == "__main__":
    main()
