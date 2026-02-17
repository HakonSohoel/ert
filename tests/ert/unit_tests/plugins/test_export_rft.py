from unittest.mock import MagicMock

import polars as pl

from ert.config import RFTConfig
from ert.config._observations import RFTObservation
from ert.plugins import ErtPluginManager
from ert.plugins.hook_implementations.workflows.export_rft import (
    ExportRFTJob,
)
from ert.storage import open_storage


def _create_rft_observation(
    well: str = "WELL1",
    date: str = "2020-01-01",
    prop: str = "PRESSURE",
    obs_name: str = "obs1",
    east: float = 100.0,
    north: float = 200.0,
    tvd: float = 25.0,
    md: float | None = 50.0,
    zone: str | None = None,
    value: float = 150.0,
    error: float = 5.0,
) -> RFTObservation:
    return RFTObservation(
        name=obs_name,
        well=well,
        date=date,
        property=prop,
        value=value,
        error=error,
        north=north,
        east=east,
        tvd=tvd,
        md=md,
        zone=zone,
    )


def _create_rft_response_df(
    well: str = "WELL1",
    date: str = "2020-01-01",
    prop: str = "PRESSURE",
    value: float = 148.0,
    east: float = 100.0,
    north: float = 200.0,
    tvd: float = 25.0,
    zone: str | None = None,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "response_key": [f"{well}:{date}:{prop}"],
            "well_and_date": [f"{well}:{date}"],
            "property": [prop],
            "values": pl.Series([value], dtype=pl.Float32),
            "east": pl.Series([east], dtype=pl.Float32),
            "north": pl.Series([north], dtype=pl.Float32),
            "tvd": pl.Series([tvd], dtype=pl.Float32),
            "zone": pl.Series([zone], dtype=pl.String),
            "i": pl.Series([0], dtype=pl.Int32),
            "j": pl.Series([0], dtype=pl.Int32),
            "k": pl.Series([0], dtype=pl.Int32),
        }
    )


def test_that_export_rft_job_is_registered_in_plugin_manager():
    pm = ErtPluginManager()
    assert "EXPORT_RFT" in pm.get_ertscript_workflows().get_workflows()


def test_that_export_rft_writes_csv_files_to_runpaths(tmp_path):
    rft_config = RFTConfig(input_files=["DUMMY"])

    observations = [_create_rft_observation()]

    responses_real0 = _create_rft_response_df()
    responses_real1 = _create_rft_response_df(value=152.0)

    runpath0 = tmp_path / "real0"
    runpath1 = tmp_path / "real1"
    runpath0.mkdir()
    runpath1.mkdir()

    with open_storage(tmp_path / "storage", mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [o.model_dump(mode="json") for o in observations],
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=2, name="test")
        ensemble.save_response("rft", responses_real0, 0)
        ensemble.save_response("rft", responses_real1, 1)

        run_paths = MagicMock()
        run_paths.get_paths.return_value = [str(runpath0), str(runpath1)]

        job = ExportRFTJob()
        job.run(run_paths, ensemble, [])

        output_file0 = runpath0 / "share/results/tables/rft_ert.csv"
        output_file1 = runpath1 / "share/results/tables/rft_ert.csv"

        assert output_file0.exists()
        assert output_file1.exists()

        df0 = pl.read_csv(output_file0)
        df1 = pl.read_csv(output_file1)

        assert "realization" not in df0.columns
        assert "realization" not in df1.columns

        assert df0["pressure"][0] == responses_real0["values"][0]
        assert df1["pressure"][0] == responses_real1["values"][0]


def test_that_export_rft_uses_custom_filename(tmp_path):
    rft_config = RFTConfig(input_files=["DUMMY"])

    observations = [_create_rft_observation()]

    responses_real0 = _create_rft_response_df()

    runpath0 = tmp_path / "real0"
    runpath0.mkdir()

    with open_storage(tmp_path / "storage", mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [o.model_dump(mode="json") for o in observations],
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        ensemble.save_response("rft", responses_real0, 0)

        run_paths = MagicMock()
        run_paths.get_paths.return_value = [str(runpath0)]

        job = ExportRFTJob()
        job.run(run_paths, ensemble, ["custom_rft.csv"])

        output_file = runpath0 / "custom_rft.csv"
        assert output_file.exists()

        df = pl.read_csv(output_file)
        assert df["pressure"][0] == responses_real0["values"][0]
