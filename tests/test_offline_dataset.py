"""Day 22 dataset schema, split, leakage, normalization, and CLI checks."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from powered_landing_guidance import load_config, validate_config
from powered_landing_guidance.offline_dataset import (
    DATASET_SPLITS,
    build_dataset_design,
    compute_training_normalization_statistics,
    dataset_configuration_sha256,
    harder_test_analysis,
    sample_split_initial_states,
    split_initial_condition_ids,
    validate_split_integrity,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/pid-baseline-v1.yaml")
EXPECTED_CONFIGURATION_SHA256 = "7a45d09cc8d21bc795615538e6cf9c69a6a08eee58ffd1561228ecbc62f929f9"
EXPECTED_INITIAL_STATE_SHA256 = {
    "train": "06e0203d6ffb9f26687ce6d734fb9010c969b9cb7d139bbef22a1231311937d1",
    "validation": "ca45656bf8785808716bb522f201bfa72827f45af21d67e98c1485e2652070eb",
    "test": "b244481859f99fd19c3ba44235fbd8f6afb4d31b04e59fa37e9d67c173420725",
}


def test_split_plan_is_reproducible_disjoint_and_read_only() -> None:
    first_report, first_arrays = build_dataset_design(CONFIG)
    second_report, second_arrays = build_dataset_design(CONFIG)

    assert first_report == second_report
    assert first_report["dataset_configuration_sha256"] == dataset_configuration_sha256(CONFIG)
    assert first_report["dataset_configuration_sha256"] == EXPECTED_CONFIGURATION_SHA256
    assert {
        split: first_report["split_plan"][split]["initial_state_sha256"] for split in DATASET_SPLITS
    } == EXPECTED_INITIAL_STATE_SHA256
    assert first_report["planned_trajectories"] == 1100
    assert first_report["completion_gate"]["passed"] is True
    assert first_report["split_integrity"]["passed"] is True
    schema = first_report["storage_schema"]
    assert schema["arrays"]["states"] == {
        "shape": ["sum(state_count)", 7],
        "dtype": "float64",
    }
    assert schema["arrays"]["actions"] == {
        "shape": ["sum(action_count)", 2],
        "dtype": "float64",
    }
    assert set(schema["solver_quality"]["required_fields"]) <= set(schema["arrays"])
    assert set(first_arrays) == {
        f"{split}_{suffix}"
        for split in DATASET_SPLITS
        for suffix in ("initial_states", "initial_condition_ids")
    }
    for name, values in first_arrays.items():
        np.testing.assert_array_equal(values, second_arrays[name])
        assert values.flags.writeable is False
    assert first_arrays["train_initial_states"].shape == (800, 7)
    assert first_arrays["validation_initial_states"].shape == (100, 7)
    assert first_arrays["test_initial_states"].shape == (200, 7)


def test_sampled_states_follow_each_declared_range() -> None:
    for split in DATASET_SPLITS:
        states = sample_split_initial_states(CONFIG, split)
        ranges = CONFIG["offline_dataset"]["splits"][split]["ranges"]

        assert np.all(np.abs(states[:, 0]) >= ranges["min_abs_x_m"])
        assert np.all((states[:, 0] >= ranges["x_m"][0]) & (states[:, 0] <= ranges["x_m"][1]))
        assert np.all((states[:, 1] >= ranges["z_m"][0]) & (states[:, 1] <= ranges["z_m"][1]))
        assert np.all((states[:, 2] >= ranges["vx_m_s"][0]) & (states[:, 2] <= ranges["vx_m_s"][1]))
        assert np.all((states[:, 3] >= ranges["vz_m_s"][0]) & (states[:, 3] <= ranges["vz_m_s"][1]))
        assert np.all(
            (np.rad2deg(states[:, 4]) >= ranges["theta_deg"][0])
            & (np.rad2deg(states[:, 4]) <= ranges["theta_deg"][1])
        )
        assert np.all(
            (np.rad2deg(states[:, 5]) >= ranges["omega_deg_s"][0])
            & (np.rad2deg(states[:, 5]) <= ranges["omega_deg_s"][1])
        )
        assert np.all(
            (states[:, 6] >= ranges["mass_kg"][0]) & (states[:, 6] <= ranges["mass_kg"][1])
        )


def test_test_envelope_is_explicitly_harder_than_train() -> None:
    analysis = harder_test_analysis(CONFIG)

    assert analysis["train_difficulty"] == "easy_nominal"
    assert analysis["test_difficulty"] == "hard_ood"
    assert analysis["passed"] is True
    assert all(analysis["checks"].values())


def test_split_integrity_rejects_initial_condition_leakage() -> None:
    states = sample_split_initial_states(CONFIG, "validation")
    identifiers = split_initial_condition_ids(states)

    with pytest.raises(ValueError, match="leakage"):
        validate_split_integrity(
            {
                "train": identifiers[:2],
                "validation": identifiers[1:3],
                "test": identifiers[3:5],
            }
        )


def test_normalization_uses_aligned_train_transitions_and_scale_floor() -> None:
    states = np.asarray(
        (
            (1.0, 10.0, 0.0, -2.0, 0.1, 0.0, 900.0),
            (3.0, 14.0, 0.0, -4.0, 0.3, 0.0, 920.0),
        )
    )
    actions = np.asarray(((0.2, -0.1), (0.6, 0.1)))

    statistics = compute_training_normalization_statistics(CONFIG, states, actions)

    assert statistics["fit_split"] == "train"
    assert statistics["terminal_states_included"] is False
    np.testing.assert_allclose(statistics["state"]["mean"], np.mean(states, axis=0))
    np.testing.assert_allclose(statistics["action"]["mean"], [0.4, 0.0])
    assert statistics["state"]["scale"][2] == pytest.approx(1.0e-6)
    assert statistics["state"]["scale"][5] == pytest.approx(1.0e-6)
    with pytest.raises(ValueError, match="aligned"):
        compute_training_normalization_statistics(CONFIG, states, actions[:1])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("schema_version",), "schema_version"),
        (("storage", "split_unit"), "split_unit"),
        (("normalization", "fit_split"), "fit_split"),
        (("splits", "validation", "seed"), "seeds"),
        (("splits", "validation", "ranges", "z_m"), "validation ranges"),
        (("splits", "test", "ranges", "min_abs_x_m"), "strictly harder"),
    ],
)
def test_config_rejects_invalid_dataset_contract(mutation, message) -> None:
    config = deepcopy(CONFIG)
    target = config["offline_dataset"]
    for key in mutation[:-1]:
        target = target[key]
    key = mutation[-1]
    replacements = {
        "schema_version": 2,
        "split_unit": "state",
        "fit_split": "validation",
        "seed": config["offline_dataset"]["splits"]["train"]["seed"],
        "z_m": [71.0, 105.0],
        "min_abs_x_m": 9.0,
    }
    target[key] = replacements[key]

    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_cli_writes_machine_readable_plan_and_refuses_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "day22"
    command = [
        sys.executable,
        str(ROOT / "scripts/design_offline_dataset.py"),
        "--output-dir",
        str(output_dir),
    ]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)

    assert first.returncode == 0, first.stderr
    report = json.loads((output_dir / "dataset-design.json").read_text(encoding="utf-8"))
    assert report["completion_gate"]["passed"] is True
    assert report["planned_trajectories"] == 1100
    with np.load(output_dir / "initial-condition-plan.npz", allow_pickle=False) as plan:
        assert set(plan.files) == {
            f"{split}_{suffix}"
            for split in DATASET_SPLITS
            for suffix in ("initial_states", "initial_condition_ids")
        }
        assert plan["train_initial_condition_ids"].dtype.kind == "U"

    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
