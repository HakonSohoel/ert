from datetime import datetime

import numpy as np
import polars as pl
import pytest

from ert.config import GenKwConfig, RFTConfig, SummaryConfig
from ert.config._observations import RFTObservation
from ert.exceptions import StorageError
from ert.storage import open_storage


def test_that_load_scalar_keys_loads_all_parameters(tmp_path):
    """Test that load_scalar_keys loads all scalar parameters when keys=None."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                    GenKwConfig(
                        name="param2",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                    GenKwConfig(
                        name="param3",
                        group="group2",
                        distribution={"name": "normal", "mean": 0, "std": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        # Save parameters
        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2],
                    "param1": [1.0, 2.0, 3.0],
                    "param2": [4.0, 5.0, 6.0],
                    "param3": [7.0, 8.0, 9.0],
                }
            )
        )

        # Load all parameters
        df = ensemble.load_scalar_keys()
        assert df.shape == (3, 4)
        assert "realization" in df.columns
        assert "param1" in df.columns
        assert "param2" in df.columns
        assert "param3" in df.columns
        assert df["param1"].to_list() == [1.0, 2.0, 3.0]


def test_that_load_scalar_keys_loads_specific_parameters(tmp_path):
    """Test that load_scalar_keys loads only specified parameters."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                    GenKwConfig(
                        name="param2",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                    GenKwConfig(
                        name="param3",
                        group="group2",
                        distribution={"name": "normal", "mean": 0, "std": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2],
                    "param1": [1.0, 2.0, 3.0],
                    "param2": [4.0, 5.0, 6.0],
                    "param3": [7.0, 8.0, 9.0],
                }
            )
        )

        # Load only param1 and param3
        df = ensemble.load_scalar_keys(keys=["param1", "param3"])
        assert df.shape == (3, 3)
        assert "realization" in df.columns
        assert "param1" in df.columns
        assert "param3" in df.columns
        assert "param2" not in df.columns


def test_that_load_scalar_keys_filters_by_realizations(tmp_path):
    """Test that load_scalar_keys filters by specified realizations."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2, 3, 4],
                    "param1": [1.0, 2.0, 3.0, 4.0, 5.0],
                }
            )
        )

        # Load only realizations 1 and 3
        df = ensemble.load_scalar_keys(keys=["param1"], realizations=np.array([1, 3]))
        assert df.shape == (2, 2)
        assert df["realization"].to_list() == [1, 3]
        assert df["param1"].to_list() == [2.0, 4.0]


def test_that_load_scalar_keys_raises_key_error_for_missing_parameters(tmp_path):
    """Test that load_scalar_keys raises KeyError for non-existent parameters."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        with pytest.raises(KeyError, match="No SCALAR dataset in storage"):
            ensemble.load_scalar_keys(keys=["param1"])


def test_that_load_scalar_keys_raises_key_error_for_unregistered_parameters(tmp_path):
    """Test that load_scalar_keys raises KeyError for parameters not in experiment."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2],
                    "param1": [1.0, 2.0, 3.0],
                }
            )
        )

        with pytest.raises(
            KeyError,
            match="Parameters not registered to the experiment: \\{'param2'\\}",
        ):
            ensemble.load_scalar_keys(keys=["param1", "param2"])


def test_that_load_scalar_keys_raises_index_error_for_missing_realizations(tmp_path):
    """Test that load_scalar_keys raises IndexError when no matching realizations."""
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    GenKwConfig(
                        name="param1",
                        group="group1",
                        distribution={"name": "uniform", "min": 0, "max": 1},
                    ).model_dump(mode="json"),
                ]
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=5, name="test")

        ensemble.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0, 1, 2],
                    "param1": [1.0, 2.0, 3.0],
                }
            )
        )

        with pytest.raises(
            IndexError,
            match="No matching realizations \\[5 6\\] found for \\['param1'\\]",
        ):
            ensemble.load_scalar_keys(keys=["param1"], realizations=np.array([5, 6]))


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
    well: str,
    date: str,
    prop: str,
    value: float,
    east: float,
    north: float,
    tvd: float,
    zone: str | None,
    i: int = 0,
    j: int = 0,
    k: int = 0,
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
            "i": pl.Series([i], dtype=pl.Int32),
            "j": pl.Series([j], dtype=pl.Int32),
            "k": pl.Series([k], dtype=pl.Int32),
        }
    )


def test_that_get_rft_observations_and_responses_returns_joined_data(tmp_path):
    zonemap_file = tmp_path / "zonemap.txt"
    zonemap_file.write_text("1 Z1")
    rft_config = RFTConfig(input_files=["DUMMY"], zonemap=zonemap_file)

    observations = [
        _create_rft_observation(zone="Z1"),
    ]

    responses_real0 = pl.concat(
        [
            _create_rft_response_df(
                "WELL1",
                "2020-01-01",
                "PRESSURE",
                148.0,
                100.0,
                200.0,
                25.0,
                "Z1",
                1,
                2,
                3,
            ),
            _create_rft_response_df(
                "WELL1", "2020-01-01", "SGAS", 0.1, 100.0, 200.0, 25.0, "Z1", 1, 2, 3
            ),
            _create_rft_response_df(
                "WELL1", "2020-01-01", "SWAT", 0.2, 100.0, 200.0, 25.0, "Z1", 1, 2, 3
            ),
        ]
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [o.model_dump(mode="json") for o in observations],
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        ensemble.save_response("rft", responses_real0, 0)

        result = ensemble.get_rft_observations_and_responses()

        assert {k: v.to_list() for k, v in result.to_dict().items()} == {
            "order": [0],
            "utm_x": [100.0],
            "utm_y": [200.0],
            "measured_depth": [50.0],
            "true_vertical_depth": [25.0],
            "zone": ["Z1"],
            "pressure": [148.0],
            "swat": [pytest.approx(0.2)],
            "sgas": [pytest.approx(0.1)],
            "soil": [pytest.approx(0.7)],  # soil is computed as 1 - sgas - swat
            "valid_zone": [True],
            "is_active": [True],
            "i": [1],
            "j": [2],
            "k": [3],
            "well": ["WELL1"],
            "time": ["2020-01-01"],
            "realization": [0],
            "report_step": [0],
            "observed": [150.0],
            "error": [5.0],
        }


def test_that_get_rft_observations_is_active_based_on_matching_pressure_response(
    tmp_path,
):
    """Test that is_active is True when there is a matching PRESSURE response,
    and False if not"""
    rft_config = RFTConfig(input_files=["DUMMY"])

    observations = [
        _create_rft_observation(),
        _create_rft_observation(
            obs_name="obs2",
            tvd=30.0,
            md=60.0,
            value=160.0,
        ),
    ]

    responses_real0 = pl.concat(
        [
            _create_rft_response_df(
                "WELL1", "2020-01-01", "PRESSURE", 148.0, 100.0, 200.0, 25.0, None
            ),
        ]
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [o.model_dump(mode="json") for o in observations],
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        ensemble.save_response("rft", responses_real0, 0)

        result = ensemble.get_rft_observations_and_responses().sort(
            "true_vertical_depth"
        )

        assert result["is_active"][0] is True  # tvd=50 has pressure response
        assert result["is_active"][1] is False  # tvd=60 has no pressure response


def test_that_get_rft_observations_and_responses_sets_valid_zone_with_null_equality(
    tmp_path,
):
    """Test that valid_zone is True when zone equals response_zone,
    including None == None."""
    zonemap_file = tmp_path / "zonemap.txt"
    zonemap_file.write_text("1 Z1\n2 Z2\n")
    rft_config = RFTConfig(input_files=["DUMMY"], zonemap=zonemap_file)

    observations = [
        _create_rft_observation(zone="Z1"),
        _create_rft_observation(
            obs_name="obs2",
            tvd=30.0,
            md=60.0,
            value=160.0,
        ),
        _create_rft_observation(
            obs_name="obs3",
            tvd=35.0,
            md=70.0,
            zone="Z2",
            value=170.0,
        ),
    ]

    responses_real0 = pl.concat(
        [
            _create_rft_response_df(
                "WELL1", "2020-01-01", "PRESSURE", 148.0, 100.0, 200.0, 25.0, "Z1"
            ),
            _create_rft_response_df(
                "WELL1", "2020-01-01", "PRESSURE", 158.0, 100.0, 200.0, 30.0, None
            ),
            _create_rft_response_df(
                "WELL1", "2020-01-01", "PRESSURE", 168.0, 100.0, 200.0, 35.0, "Z1"
            ),
        ]
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [o.model_dump(mode="json") for o in observations],
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        ensemble.save_response("rft", responses_real0, 0)

        result = ensemble.get_rft_observations_and_responses().sort(
            "true_vertical_depth"
        )

        assert result["valid_zone"][0] is True  # Z1 == Z1
        assert result["valid_zone"][1] is True  # None == None
        assert result["valid_zone"][2] is False  # Z2 != Z1


def test_that_get_rft_observations_and_responses_order_is_row_index_within_well(
    tmp_path,
):
    """Test that order column is 0-based row index within each well."""
    rft_config = RFTConfig(input_files=["DUMMY"])

    observations = [
        _create_rft_observation(obs_name="obs1", tvd=25.0, md=50.0),
        _create_rft_observation(obs_name="obs2", tvd=30.0, md=60.0),
        _create_rft_observation(obs_name="obs3", tvd=35.0, md=70.0),
        _create_rft_observation(well="WELL2", obs_name="obs4", tvd=40.0, md=80.0),
        _create_rft_observation(well="WELL2", obs_name="obs5", tvd=45.0, md=90.0),
    ]

    responses_real0 = pl.concat(
        [
            _create_rft_response_df(
                "WELL1", "2020-01-01", "PRESSURE", 148.0, 100.0, 200.0, 25.0, None
            )
        ]
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [o.model_dump(mode="json") for o in observations],
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        ensemble.save_response("rft", responses_real0, 0)

        result = ensemble.get_rft_observations_and_responses().sort(
            ["well", "true_vertical_depth"]
        )

        assert result["well"].to_list() == ["WELL1", "WELL1", "WELL1", "WELL2", "WELL2"]
        assert result["order"].to_list() == [0, 1, 2, 0, 1]


def test_that_get_rft_observations_and_responses_handles_multiple_realizations(
    tmp_path,
):
    """Test that responses from multiple realizations are concatenated correctly."""
    rft_config = RFTConfig(input_files=["DUMMY"])

    observations = [_create_rft_observation()]

    responses_real0 = _create_rft_response_df(
        "WELL1", "2020-01-01", "PRESSURE", 148.0, 100.0, 200.0, 25.0, None
    )
    responses_real1 = _create_rft_response_df(
        "WELL1", "2020-01-01", "PRESSURE", 152.0, 100.0, 200.0, 25.0, None
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [o.model_dump(mode="json") for o in observations],
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=2, name="test")
        ensemble.save_response("rft", responses_real0, 0)
        ensemble.save_response("rft", responses_real1, 1)

        result = ensemble.get_rft_observations_and_responses().sort("realization")

        assert result.shape[0] == 2
        assert result["realization"].to_list() == [0, 1]
        assert result["pressure"][0] == pytest.approx(148.0)
        assert result["pressure"][1] == pytest.approx(152.0)


def test_that_get_rft_observations_and_responses_raises_error_for_no_observations(
    tmp_path,
):
    """Test that StorageError is raised when no RFT observations exist."""
    rft_config = RFTConfig(input_files=["DUMMY"])

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": {},
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")

        with pytest.raises(StorageError, match="No RFT observations found"):
            ensemble.get_rft_observations_and_responses()


def test_that_get_rft_observations_and_responses_raises_error_when_response_not_saved(
    tmp_path,
):
    rft_config = RFTConfig(input_files=["DUMMY"])

    observations = [_create_rft_observation()]

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [o.model_dump(mode="json") for o in observations],
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")

        with pytest.raises(KeyError, match="No response for key rft"):
            ensemble.get_rft_observations_and_responses()


def test_that_get_rft_observations_and_responses_adds_missing_saturation_columns(
    tmp_path,
):
    rft_config = RFTConfig(input_files=["DUMMY"])

    observations = [_create_rft_observation()]

    responses_real0 = _create_rft_response_df(
        "WELL1", "2020-01-01", "PRESSURE", 148.0, 100.0, 200.0, 25.0, None
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [o.model_dump(mode="json") for o in observations],
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        ensemble.save_response("rft", responses_real0, 0)

        result = ensemble.get_rft_observations_and_responses()

        assert "sgas" in result.columns
        assert "swat" in result.columns
        assert "soil" in result.columns
        assert result["sgas"][0] is None
        assert result["swat"][0] is None
        assert result["soil"][0] is None


def test_that_get_rft_observations_and_responses_maps_report_step_from_summary_times(
    tmp_path,
):
    """Test that report_step is mapped from summary response times."""
    rft_config = RFTConfig(input_files=["DUMMY"])
    summary_config = SummaryConfig(keys=["FOPR"], input_files=["DUMMY"])

    observations = [
        _create_rft_observation(date="2020-01-15"),
        _create_rft_observation(
            date="2020-02-15",
            obs_name="obs2",
            tvd=30.0,
            md=60.0,
            value=160.0,
        ),
    ]

    rft_responses = pl.concat(
        [
            _create_rft_response_df(
                "WELL1", "2020-01-15", "PRESSURE", 148.0, 100.0, 200.0, 25.0, None
            ),
            _create_rft_response_df(
                "WELL1", "2020-02-15", "PRESSURE", 158.0, 100.0, 200.0, 30.0, None
            ),
        ]
    )

    summary_responses = pl.DataFrame(
        {
            "response_key": ["FOPR"] * 3,
            "time": pl.Series(
                [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 15),
                    datetime(2020, 2, 15),
                ]
            ).dt.cast_time_unit("ms"),
            "values": pl.Series([100.0, 200.0, 300.0], dtype=pl.Float32),
        }
    )

    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [
                    rft_config.model_dump(mode="json"),
                    summary_config.model_dump(mode="json"),
                ],
                "observations": [o.model_dump(mode="json") for o in observations],
            }
        )
        ensemble = storage.create_ensemble(experiment.id, ensemble_size=1, name="test")
        ensemble.save_response("rft", rft_responses, 0)
        ensemble.save_response("summary", summary_responses, 0)

        result = ensemble.get_rft_observations_and_responses().sort("time")

        assert "report_step" in result.columns
        assert result["report_step"][0] == 1
        assert result["report_step"][1] == 2
