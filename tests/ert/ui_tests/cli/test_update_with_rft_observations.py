from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from resfo_utilities import RFTReader

from ert.mode_definitions import ENSEMBLE_EXPERIMENT_MODE, ENSEMBLE_SMOOTHER_MODE
from ert.storage import open_storage

from .run_cli import run_cli


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("flow"), reason="flow not available")
def test_that_rft_response_for_inactive_cell_is_interpolated_from_nearest_active_cells(
    use_tmpdir, source_root
):
    """
    Runs ensemble experiment on a 3x1x1 grid where cell (2,1,1) is inactive.
    Well OP1 has connections in all 3 cells. Observations are placed at:
     - EAST=25 (active cell 1,1,1) -> should have a response value
     - EAST=75 (inactive cell 2,1,1) -> interpolated from neighbors
     - EAST=125 (active cell 3,1,1) -> should have a response value
    """
    shutil.copytree(
        os.path.join(source_root, "test-data", "ert", "rft_interpolate_example"),
        "test-data",
    )
    run_cli(
        ENSEMBLE_EXPERIMENT_MODE,
        "--disable-monitoring",
        "test-data/rft.ert",
    )

    with open_storage(Path("storage"), mode="r") as storage:
        experiment = next(iter(storage.experiments))
        ensemble = experiment.get_ensemble_by_name("default")
        obs_and_resp = ensemble.get_observations_and_responses(
            experiment.observation_keys,
            np.array([0]),
        )

    assert obs_and_resp.columns == [
        "response_key",
        "index",
        "observation_key",
        "observations",
        "std",
        "east",
        "north",
        "radius",
        "0",
    ]
    assert sorted(obs_and_resp["observation_key"].to_list()) == [
        "rft_obs_1",
        "rft_obs_2",
        "rft_obs_3",
    ]

    obs_by_key = {
        row["observation_key"]: row for row in obs_and_resp.iter_rows(named=True)
    }

    # Active cell (1,1,1) at EAST=25 — should have a real response
    response_1 = obs_by_key["rft_obs_1"]["0"]
    assert response_1 is not None
    assert not np.isnan(response_1)

    # Active cell (3,1,1) at EAST=125 — should have a real response
    response_3 = obs_by_key["rft_obs_3"]["0"]
    assert response_3 is not None
    assert not np.isnan(response_3)

    # Inactive cell (2,1,1) at EAST=75 — Flow drops connection, but the
    # response is linearly interpolated from the two nearest active cells.
    # The missing cell is exactly between the two active cells, so the
    # interpolated value should be the average of the two responses.
    response_2 = obs_by_key["rft_obs_2"]["0"]
    assert response_2 is not None
    assert not np.isnan(response_2)
    assert response_2 == pytest.approx((response_1 + response_3) / 2, rel=1e-4)


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("flow"), reason="flow not available")
def test_that_rft_example_with_rft_observation_keyword_yelds_same_result_as_gendata_rft(
    use_tmpdir, source_root, snapshot, request
):
    """
    Created snapshot of rft pressures by using the GENDATA_RFT in combination
    with GENERAL_OBSERVATION. This test runs the same experiment using RFT_OBSERVATION
    to verify that the resulting RFT pressure data is the same.
    """
    shutil.copytree(
        os.path.join(source_root, "test-data", "ert", "rft_example"), "test-data"
    )
    run_cli(ENSEMBLE_SMOOTHER_MODE, "--disable-monitoring", "test-data/rft.ert")
    pressure = {}
    for file in sorted(Path("spe1_out").rglob("*.RFT")):
        rft = RFTReader.open(file)
        for entry in rft:
            key = "/".join(file.parts[-3:-1]) + f", {entry.date}, {entry.well}"
            pressure[key] = entry["PRESSURE"].tolist()

    FILE_NAME = "rft_pressures.json"

    if bool(request.config.getoption("--snapshot-update")):
        snapshot.assert_match(json.dumps(pressure, indent=2) + "\n", FILE_NAME)

    with Path(snapshot.snapshot_dir, FILE_NAME).open(encoding="utf-8") as file:
        pressure_snapshot = json.load(file)

    assert isinstance(pressure, dict)
    for key, snapshot_value in pressure_snapshot.items():
        assert pressure.get(key) == pytest.approx(snapshot_value, abs=0.05), (
            f"RFT pressure snapshot test failed for {key}"
        )
