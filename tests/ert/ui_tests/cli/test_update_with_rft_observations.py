from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from resfo_utilities import RFTReader

from ert.mode_definitions import ES_MDA_MODE

from .run_cli import run_cli


@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_rft_example(use_tmpdir, source_root, snapshot):
    shutil.copytree(
        os.path.join(source_root, "test-data", "ert", "rft_example"), "test-data"
    )
    run_cli(ES_MDA_MODE, "--disable-monitoring", "test-data/rft.ert")
    pressure = {}
    for file in sorted(Path("spe1_out").rglob("*.RFT")):
        rft = RFTReader.open(file)
        pressure["/".join(file.parts[-3:-1])] = {
            f"{entry.date}, {entry.well}": entry["PRESSURE"].tolist() for entry in rft
        }

    snapshot.assert_match(json.dumps(pressure), "rft_pressures")
