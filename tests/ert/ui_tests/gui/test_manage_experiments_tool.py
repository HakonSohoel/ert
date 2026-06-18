import datetime
import shutil
import time
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import polars as pl
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QPushButton,
    QTableWidget,
    QTextEdit,
)

from ert.config import ErtConfig, SummaryConfig
from ert.config.rft_config import RFTConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.tools.manage_experiments import ManageExperimentsPanel
from ert.gui.tools.manage_experiments.rft_qc_widget import (
    FilterPanel,
    RftQcWidget,
    _add_status_col_to_df,
    _PointStatus,
    _unique_points_per_coordinate,
)
from ert.gui.tools.manage_experiments.storage_info_widget import (
    ExportDialog,
    _EnsembleWidget,
    _EnsembleWidgetTabs,
    _ExperimentWidget,
    _RealizationWidget,
    _WidgetType,
)
from ert.gui.tools.manage_experiments.storage_widget import StorageWidget
from ert.storage import (
    RealizationStorageState,
    Storage,
    open_storage,
)
from tests.ert.rft_generator import cell_start, create_egrid
from tests.ert.ui_tests.cli.analysis.test_adaptive_localization import (
    run_cli_ES_with_case,
)

from .conftest import add_experiment_in_manage_experiment_dialog


def test_design_matrix_in_manage_experiments_panel(
    copy_poly_case_with_design_matrix, qtbot, use_tmpdir
):
    num_realizations = 10
    a_values = list(range(num_realizations))
    design_dict = {
        "REAL": list(range(num_realizations)),
        "a": a_values,
    }
    default_list = [["b", 1], ["c", 2]]
    copy_poly_case_with_design_matrix(design_dict, default_list)
    config = ErtConfig.from_file("poly.ert")
    notifier = ErtNotifier()
    notifier.set_storage(str(config.ens_path))
    assert config.ensemble_config.parameter_configuration == []
    assert config.analysis_config.design_matrix is not None

    with notifier.write_storage() as storage:
        storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    pc.model_dump(mode="json")
                    for pc in (
                        config.analysis_config.design_matrix.parameter_configurations
                    )
                ],
                "response_configuration": [
                    rc.model_dump(mode="json")
                    for rc in config.ensemble_config.response_configuration
                ],
            },
            name="my-experiment",
        ).create_ensemble(
            ensemble_size=config.runpath_config.num_realizations,
            name="my-design",
        )

    # Notifier storage is persistent, read-storage is not,
    # hence we get the ensemble from the read storage
    ensemble = notifier.storage.get_experiment_by_name(
        "my-experiment"
    ).get_ensemble_by_name("my-design")
    notifier.set_current_ensemble_id(ensemble.id)
    assert all(
        RealizationStorageState.UNDEFINED in s for s in ensemble.get_ensemble_state()
    )

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
    )
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_from_scratch_button"),
        Qt.MouseButton.LeftButton,
    )
    assert (
        RealizationStorageState.PARAMETERS_LOADED in s
        for s in ensemble.get_ensemble_state()
    )

    params = ensemble.load_parameters("DESIGN_MATRIX").drop("realization")
    np.testing.assert_array_equal(params["a"].to_list(), a_values)
    np.testing.assert_array_equal(params["b"].to_list(), np.ones(num_realizations))
    np.testing.assert_array_equal(params["c"].to_list(), 2 * np.ones(num_realizations))

    add_experiment_in_manage_experiment_dialog(
        qtbot, tool, experiment_name="my-experiment-2", ensemble_name="my-design-2"
    )

    experiments = list(notifier.storage.experiments)
    assert len(experiments) == 2

    # The write-storage writes the experiments,
    # and the read-storage refreshes itself.
    # There is no guarantee that the experiment UUIDs are in order-of-creation
    # hence, we do not assert the order here
    assert {e.name for e in experiments} == {"my-experiment", "my-experiment-2"}
    exp2 = notifier.storage.get_experiment_by_name("my-experiment-2")
    ensemble = exp2.get_ensemble_by_name("my-design-2")
    for param in exp2.parameter_configuration.values():
        assert param.group_name == "DESIGN_MATRIX"
    assert {p.name for p in exp2.parameter_configuration.values()} == {"a", "b", "c"}
    assert all(
        RealizationStorageState.UNDEFINED in s for s in ensemble.get_ensemble_state()
    )


@pytest.mark.usefixtures("copy_poly_case")
def test_init_prior(qtbot):
    config = ErtConfig.from_file("poly.ert")
    config.random_seed = 1234
    notifier = ErtNotifier()
    notifier.set_storage(config.ens_path)

    with notifier.write_storage() as storage:
        ensemble = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    pc.model_dump(mode="json")
                    for pc in config.ensemble_config.parameter_configuration
                ],
                "response_configuration": [
                    rc.model_dump(mode="json")
                    for rc in config.ensemble_config.response_configuration
                ],
            },
            name="my-experiment",
        ).create_ensemble(
            ensemble_size=config.runpath_config.num_realizations,
            name="prior",
        )

        assert all(
            RealizationStorageState.UNDEFINED in s
            for s in ensemble.get_ensemble_state()
        )
    notifier.set_current_ensemble_id(ensemble.id)

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
    )
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_from_scratch_button"),
        Qt.MouseButton.LeftButton,
    )
    assert (
        RealizationStorageState.PARAMETERS_LOADED in s
        for s in notifier.current_ensemble.get_ensemble_state()
    )
    assert notifier.current_ensemble.load_parameters_numpy(
        "COEFFS", np.arange(ensemble.ensemble_size)
    ).mean() == pytest.approx(0.0458710649708845)


@pytest.mark.usefixtures("copy_poly_case")
def test_that_init_updates_the_info_tab(qtbot):
    config = ErtConfig.from_file("poly.ert")
    notifier = ErtNotifier()
    notifier.set_storage(config.ens_path)
    ensemble_config = config.ensemble_config

    with notifier.write_storage() as storage:
        ensemble = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    pc.model_dump(mode="json")
                    for pc in ensemble_config.parameter_configuration
                ],
                "response_configuration": [
                    rc.model_dump(mode="json")
                    for rc in ensemble_config.response_configuration
                ],
                "observations": [
                    od.model_dump(mode="json") for od in config.observation_declarations
                ],
                "ert_templates": config.ert_templates,
            },
            name="my-experiment",
        ).create_ensemble(
            ensemble_size=config.runpath_config.num_realizations, name="default"
        )
    notifier.set_current_ensemble_id(ensemble.id)

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
    )

    html_edit = tool.findChild(QTextEdit, name="ensemble_state_text")
    assert not html_edit.toPlainText()

    # select the created ensemble
    storage_widget = tool.findChild(StorageWidget)
    storage_widget._tree_view.expandAll()
    model_index = storage_widget._tree_view.model().index(
        0, 0, storage_widget._tree_view.model().index(0, 0)
    )
    storage_widget._tree_view.setCurrentIndex(model_index)

    # select the correct tab
    ensemble_widget = tool.findChild(_EnsembleWidget)
    ensemble_widget._currentTabChanged(1)

    assert "UNDEFINED" in html_edit.toPlainText()
    assert "RealizationStorageState.UNDEFINED" not in html_edit.toPlainText()

    # Change to the "initialize from scratch" tab
    tool.setCurrentIndex(1)
    qtbot.mouseClick(
        tool.findChild(QPushButton, name="initialize_from_scratch_button"),
        Qt.MouseButton.LeftButton,
    )

    # Change back to first tab
    tool.setCurrentIndex(0)
    ensemble_widget._currentTabChanged(1)
    assert "PARAMETERS_LOADED" in html_edit.toPlainText()
    assert "RealizationStorageState.PARAMETERS_LOADED" not in html_edit.toPlainText()

    # select the observation
    storage_info_widget = tool._storage_info_widget
    storage_info_widget._ensemble_widget._tab_widget.setCurrentIndex(
        _EnsembleWidgetTabs.OBSERVATIONS_TAB
    )
    observation_tree = storage_info_widget._ensemble_widget._observations_tree_widget
    model_index = observation_tree.model().index(
        0, 0, observation_tree.model().index(0, 0)
    )
    observation_tree.setCurrentIndex(model_index)
    assert (
        storage_info_widget._ensemble_widget._figure.axes[0].title.get_text()
        == "POLY_OBS"
    )


def test_experiment_view(
    qtbot, snake_oil_case_storage: ErtConfig, snake_oil_storage: Storage
):
    config = snake_oil_case_storage
    storage = snake_oil_storage

    notifier = ErtNotifier()
    notifier.set_storage(str(storage.path))

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
    )

    # select the experiment
    storage_widget = tool.findChild(StorageWidget)
    storage_widget._tree_view.expandAll()
    model_index = storage_widget._tree_view.model().index(0, 0)
    storage_widget._tree_view.setCurrentIndex(model_index)
    assert (
        tool._storage_info_widget._content_layout.currentIndex()
        == _WidgetType.EXPERIMENT_WIDGET
    )

    experiment_widget = tool._storage_info_widget._content_layout.currentWidget()
    assert isinstance(experiment_widget, _ExperimentWidget)
    assert experiment_widget._name_label.text()
    assert experiment_widget._uuid_label.text()
    assert experiment_widget._parameters_text_edit.toPlainText()
    assert experiment_widget._responses_text_edit.toPlainText()
    assert experiment_widget._observations_text_edit.toPlainText()


def test_ensemble_view(
    qtbot, snake_oil_case_storage: ErtConfig, snake_oil_storage: Storage
):
    config = snake_oil_case_storage
    storage = snake_oil_storage

    notifier = ErtNotifier()
    notifier.set_storage(str(storage.path))

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
    )

    # select the ensemble
    storage_widget = tool.findChild(StorageWidget)
    storage_widget._tree_view.expandAll()
    model_index = storage_widget._tree_view.model().index(
        0, 0, storage_widget._tree_view.model().index(0, 0)
    )
    storage_widget._tree_view.setCurrentIndex(model_index)
    assert (
        tool._storage_info_widget._content_layout.currentIndex()
        == _WidgetType.ENSEMBLE_WIDGET
    )

    ensemble_widget = tool._storage_info_widget._content_layout.currentWidget()
    assert isinstance(ensemble_widget, _EnsembleWidget)
    assert ensemble_widget._name_label.text()
    assert ensemble_widget._uuid_label.text()
    assert not ensemble_widget._state_text_edit.toPlainText()

    ensemble_widget._tab_widget.setCurrentIndex(_EnsembleWidgetTabs.STATE_TAB)
    assert ensemble_widget._state_text_edit.toPlainText()

    ensemble_widget._tab_widget.setCurrentIndex(_EnsembleWidgetTabs.OBSERVATIONS_TAB)
    ensemble_widget._observations_tree_widget.expandAll()
    assert ensemble_widget._observations_tree_widget.topLevelItemCount() == 2
    assert ensemble_widget._observations_tree_widget.topLevelItem(0).childCount() == 4
    assert ensemble_widget._observations_tree_widget.topLevelItem(1).childCount() == 6

    # simulate clicking some different entries in observation list
    ensemble_widget._observations_tree_widget.currentItemChanged.emit(
        ensemble_widget._observations_tree_widget.topLevelItem(0).child(10), None
    )
    assert ensemble_widget._figure.get_axes()[0].get_title() == "WPR_DIFF_1"

    ensemble_widget._observations_tree_widget.currentItemChanged.emit(
        ensemble_widget._observations_tree_widget.topLevelItem(1).child(2), None
    )
    assert ensemble_widget._figure.get_axes()[0].get_title() == "WOPR_OP1_72"

    ensemble_widget._observations_tree_widget.currentItemChanged.emit(
        ensemble_widget._observations_tree_widget.topLevelItem(1).child(3), None
    )
    assert ensemble_widget._figure.get_axes()[0].get_title() == "WOPR_OP1_108"


@pytest.mark.usefixtures("copy_poly_case")
def test_ensemble_observations_view(qtbot):
    Path("observations").write_text(
        """GENERAL_OBSERVATION POLY_OBS {
        DATA       = POLY_RES;
        INDEX_LIST = 0,1,2,3,4;
        OBS_FILE   = poly_obs_data.txt;
    };
    GENERAL_OBSERVATION POLY_OBS1_1 {
        DATA       = POLY_RES1;
        INDEX_LIST = 0,1,2,3,4;
        OBS_FILE   = poly_obs_data1.txt;
    };
    GENERAL_OBSERVATION POLY_OBS1_2 {
        DATA       = POLY_RES2;
        INDEX_LIST = 0,1,2,3,4;
        OBS_FILE   = poly_obs_data2.txt;
    };
    """,
        encoding="utf-8",
    )

    Path("poly_eval.py").write_text(
        """#!/usr/bin/env python3
import json


def _load_coeffs(filename):
    with open(filename, encoding="utf-8") as f:
        return json.load(f)


def _evaluate(coeffs, x):
    return coeffs["a"]["value"] * x**2 + coeffs["b"]["value"] * x + coeffs["c"]["value"]


if __name__ == "__main__":
    coeffs = _load_coeffs("parameters.json")
    output = [_evaluate(coeffs, x) for x in range(10)]
    with open("poly.out", "w", encoding="utf-8") as f:
        f.write("\\n".join(map(str, output)))

    with open("poly.out1", "w", encoding="utf-8") as f:
        f.write("\\n".join(map(str, [x*2 for x in output])))

    with open("poly.out2", "w", encoding="utf-8") as f:
        f.write("\\n".join(map(str, [x*3 for x in output])))
""",
        encoding="utf-8",
    )

    shutil.copy("poly_obs_data.txt", "poly_obs_data1.txt")
    shutil.copy("poly_obs_data.txt", "poly_obs_data2.txt")

    Path("poly_localization_0.ert").write_text(
        """
        QUEUE_SYSTEM LOCAL
QUEUE_OPTION LOCAL MAX_RUNNING 2

RUNPATH poly_out/realization-<IENS>/iter-<ITER>

OBS_CONFIG observations
REALIZATION_MEMORY 50mb

NUM_REALIZATIONS 100
MIN_REALIZATIONS 1

GEN_KW COEFFS coeff_priors
GEN_DATA POLY_RES RESULT_FILE:poly.out
GEN_DATA POLY_RES1 RESULT_FILE:poly.out1
GEN_DATA POLY_RES2 RESULT_FILE:poly.out2

INSTALL_JOB poly_eval POLY_EVAL
FORWARD_MODEL poly_eval

ANALYSIS_SET_VAR STD_ENKF LOCALIZATION True
ANALYSIS_SET_VAR STD_ENKF LOCALIZATION_CORRELATION_THRESHOLD 0.0

ANALYSIS_SET_VAR OBSERVATIONS AUTO_SCALE *
ANALYSIS_SET_VAR OBSERVATIONS AUTO_SCALE POLY_OBS1_*
""",
        encoding="utf-8",
    )

    prior_ens_id, _, _ = run_cli_ES_with_case(
        "poly_localization_0.ert", "test_experiment"
    )
    config = ErtConfig.from_file("poly_localization_0.ert")

    notifier = ErtNotifier()
    with open_storage(config.ens_path, mode="r") as storage:
        notifier.set_storage(str(storage.path))

        tool = ManageExperimentsPanel(
            config, notifier, config.runpath_config.num_realizations
        )

        assert storage.get_ensemble(prior_ens_id).name

        # select the ensemble
        storage_widget = tool.findChild(StorageWidget)
        storage_widget._tree_view.expandAll()

        model = storage_widget._tree_view.model()
        experiment_index = model.index(0, 0)

        target_index = None
        for r in range(model.rowCount(experiment_index)):
            idx = model.index(r, 0, experiment_index)
            if model.data(idx, Qt.ItemDataRole.DisplayRole) == "iter-1":
                target_index = idx
                break
        assert target_index is not None
        storage_widget._tree_view.setCurrentIndex(target_index)
        assert (
            tool._storage_info_widget._content_layout.currentIndex()
            == _WidgetType.ENSEMBLE_WIDGET
        )

        ensemble_widget = tool._storage_info_widget._content_layout.currentWidget()
        assert isinstance(ensemble_widget, _EnsembleWidget)
        assert ensemble_widget._name_label.text()
        assert ensemble_widget._uuid_label.text()
        assert not ensemble_widget._state_text_edit.toPlainText()

        ensemble_widget._tab_widget.setCurrentIndex(_EnsembleWidgetTabs.STATE_TAB)
        assert ensemble_widget._state_text_edit.toPlainText()

        ensemble_widget._tab_widget.setCurrentIndex(
            _EnsembleWidgetTabs.OBSERVATIONS_TAB
        )

        # Check that a scaled observation is plotted
        assert any(
            line
            for line in ensemble_widget._figure.get_axes()[0].get_lines()
            if "Scaled observation" in line.get_xdata()
        )


@pytest.mark.usefixtures("copy_poly_case")
def test_ensemble_observations_view_on_empty_ensemble(qtbot):
    config = ErtConfig.from_file("poly.ert")
    notifier = ErtNotifier()
    notifier.set_storage(config.ens_path)

    with notifier.write_storage() as storage:
        notifier.set_storage(str(storage.path))
        exp = storage.create_experiment(
            experiment_config={
                "response_configuration": [
                    SummaryConfig(keys=["*"]).model_dump(mode="json")
                ],
                "observations": [
                    {
                        "type": "summary_observation",
                        "name": "O4",
                        "key": "FOPR",
                        "date": "2000-01-01",
                        "value": 10.2,
                        "error": 0.1,
                    }
                ],
            }
        )

        exp.create_ensemble(
            name="test", ensemble_size=config.runpath_config.num_realizations
        )

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
    )

    # select the ensemble
    storage_widget = tool.findChild(StorageWidget)
    storage_widget._tree_view.expandAll()
    model_index = storage_widget._tree_view.model().index(
        0, 0, storage_widget._tree_view.model().index(0, 0)
    )
    storage_widget._tree_view.setCurrentIndex(model_index)
    assert (
        tool._storage_info_widget._content_layout.currentIndex()
        == _WidgetType.ENSEMBLE_WIDGET
    )

    ensemble_widget = tool._storage_info_widget._content_layout.currentWidget()
    assert isinstance(ensemble_widget, _EnsembleWidget)
    assert ensemble_widget._name_label.text()
    assert ensemble_widget._uuid_label.text()
    assert not ensemble_widget._state_text_edit.toPlainText()

    ensemble_widget._tab_widget.setCurrentIndex(_EnsembleWidgetTabs.STATE_TAB)
    assert ensemble_widget._state_text_edit.toPlainText()

    ensemble_widget._tab_widget.setCurrentIndex(_EnsembleWidgetTabs.OBSERVATIONS_TAB)

    # Expect only one figure, the one for the observation
    assert len(ensemble_widget._figure.get_axes()) == 1


def test_realization_view(
    qtbot, snake_oil_case_storage: ErtConfig, snake_oil_storage: Storage
):
    config = snake_oil_case_storage
    storage = snake_oil_storage

    notifier = ErtNotifier()
    notifier.set_storage(str(storage.path))

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
    )

    # select the realization
    storage_widget = tool.findChild(StorageWidget)
    storage_widget._tree_view.expandAll()
    model_index = storage_widget._tree_view.model().index(
        0,
        0,
        storage_widget._tree_view.model().index(
            0, 0, storage_widget._tree_view.model().index(0, 0)
        ),
    )
    storage_widget._tree_view.setCurrentIndex(model_index)
    assert (
        tool._storage_info_widget._content_layout.currentIndex()
        == _WidgetType.REALIZATION_WIDGET
    )

    realization_widget = tool._storage_info_widget._content_layout.currentWidget()
    assert type(realization_widget) is _RealizationWidget

    assert (
        realization_widget._state_label.text()
        == "Realization state: PARAMETERS_LOADED, RESPONSES_LOADED"
    )
    assert {"gen_data - RESPONSES_LOADED", "summary - RESPONSES_LOADED"}.issubset(
        set(realization_widget._response_text_edit.toPlainText().splitlines())
    )

    assert {
        "OP1_PERSISTENCE - PARAMETERS_LOADED",
        "OP1_OCTAVES - PARAMETERS_LOADED",
        "OP1_DIVERGENCE_SCALE - PARAMETERS_LOADED",
        "OP1_OFFSET - PARAMETERS_LOADED",
        "OP2_PERSISTENCE - PARAMETERS_LOADED",
        "OP2_OCTAVES - PARAMETERS_LOADED",
        "OP2_DIVERGENCE_SCALE - PARAMETERS_LOADED",
        "OP2_OFFSET - PARAMETERS_LOADED",
        "BPR_555_PERSISTENCE - PARAMETERS_LOADED",
        "BPR_138_PERSISTENCE - PARAMETERS_LOADED",
    } == set(realization_widget._parameter_text_edit.toPlainText().strip().splitlines())


def test_that_parameters_pane_is_populated_correctly(
    qtbot, snake_oil_case_storage: ErtConfig, snake_oil_storage: Storage
):
    config = snake_oil_case_storage
    storage = snake_oil_storage

    notifier = ErtNotifier()
    notifier.set_storage(str(storage.path))

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
    )

    storage_widget = tool.findChild(StorageWidget)
    storage_widget._tree_view.expandAll()
    model_index = storage_widget._tree_view.model().index(
        0, 0, storage_widget._tree_view.model().index(0, 0)
    )
    storage_widget._tree_view.setCurrentIndex(model_index)
    assert (
        tool._storage_info_widget._content_layout.currentIndex()
        == _WidgetType.ENSEMBLE_WIDGET
    )
    ensemble_widget = tool._storage_info_widget._content_layout.currentWidget()
    ensemble_widget._tab_widget.setCurrentIndex(_EnsembleWidgetTabs.PARAMETERS_TAB)

    assert isinstance(ensemble_widget._tab_widget.currentWidget(), QFrame)

    parameters_frame = ensemble_widget._tab_widget.currentWidget()
    assert parameters_frame.findChild(QPushButton) is not None
    assert parameters_frame.findChild(QTableWidget) is not None

    model = parameters_frame.findChild(QTableWidget).model()
    assert model.rowCount() == 5, "The Snake oil test case should have 5 realizations"
    assert model.columnCount() == 11, (
        "The Snake oil test case should have 11 parameters"
    )

    triggers = parameters_frame.findChild(QTableWidget).editTriggers()
    assert triggers == QAbstractItemView.EditTrigger.NoEditTriggers


@pytest.mark.usefixtures("copy_poly_case")
def test_that_sub_tab_persists_when_switching_ensembles(qtbot):
    config = ErtConfig.from_file("poly.ert")
    notifier = ErtNotifier()
    notifier.set_storage(config.ens_path)

    with notifier.write_storage() as storage:
        exp = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    pc.model_dump(mode="json")
                    for pc in config.ensemble_config.parameter_configuration
                ],
                "response_configuration": [
                    rc.model_dump(mode="json")
                    for rc in config.ensemble_config.response_configuration
                ],
            },
            name="my-experiment",
        )
        exp.create_ensemble(
            ensemble_size=config.runpath_config.num_realizations, name="prior"
        )
        exp.create_ensemble(
            ensemble_size=config.runpath_config.num_realizations, name="posterior"
        )

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
    )

    storage_widget = tool.findChild(StorageWidget)
    storage_widget._tree_view.expandAll()
    experiment_index = storage_widget._tree_view.model().index(0, 0)

    # Select first ensemble
    first_ensemble_index = storage_widget._tree_view.model().index(
        0, 0, experiment_index
    )
    storage_widget._tree_view.setCurrentIndex(first_ensemble_index)

    ensemble_widget = tool._storage_info_widget._content_layout.currentWidget()
    assert isinstance(ensemble_widget, _EnsembleWidget)

    # Switch to STATE_TAB
    ensemble_widget._tab_widget.setCurrentIndex(_EnsembleWidgetTabs.STATE_TAB)
    assert ensemble_widget._tab_widget.currentIndex() == _EnsembleWidgetTabs.STATE_TAB

    # Select second ensemble
    second_ensemble_index = storage_widget._tree_view.model().index(
        1, 0, experiment_index
    )
    storage_widget._tree_view.setCurrentIndex(second_ensemble_index)

    # Tab should remain on STATE_TAB, not reset to ENSEMBLE_TAB
    assert ensemble_widget._tab_widget.currentIndex() == _EnsembleWidgetTabs.STATE_TAB


def test_that_export_parameters_button_opens_the_export_dialog(
    qtbot, snake_oil_case_storage: ErtConfig, snake_oil_storage: Storage
):
    config = snake_oil_case_storage
    storage = snake_oil_storage

    notifier = ErtNotifier()
    notifier.set_storage(str(storage.path))

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
    )

    storage_widget = tool.findChild(StorageWidget)
    storage_widget._tree_view.expandAll()
    model_index = storage_widget._tree_view.model().index(
        0, 0, storage_widget._tree_view.model().index(0, 0)
    )
    storage_widget._tree_view.setCurrentIndex(model_index)
    assert (
        tool._storage_info_widget._content_layout.currentIndex()
        == _WidgetType.ENSEMBLE_WIDGET
    )
    ensemble_widget = tool._storage_info_widget._content_layout.currentWidget()
    ensemble_widget._tab_widget.setCurrentIndex(_EnsembleWidgetTabs.PARAMETERS_TAB)
    assert isinstance(ensemble_widget._tab_widget.currentWidget(), QFrame)

    parameters_frame = ensemble_widget._tab_widget.currentWidget()
    parameters_frame.findChild(QPushButton).click()
    assert isinstance(QApplication.activeModalWidget(), ExportDialog)


def _child_names_under(model, parent_index):
    return [
        model.data(model.index(r, 0, parent_index), Qt.ItemDataRole.DisplayRole)
        for r in range(model.rowCount(parent_index))
    ]


def _find_root_row_by_name(model, name: str):
    for r in range(model.rowCount()):
        idx = model.index(r, 0)
        if model.data(idx, Qt.ItemDataRole.DisplayRole) == name:
            return idx
    return None


@pytest.mark.usefixtures("copy_poly_case")
def test_that_storage_widget_sorts_by_name_and_created(qtbot):
    config = ErtConfig.from_file("poly.ert")
    notifier = ErtNotifier()
    notifier.set_storage(config.ens_path)

    with notifier.write_storage() as storage:
        exp = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    pc.model_dump(mode="json")
                    for pc in config.ensemble_config.parameter_configuration
                ],
                "response_configuration": [
                    rc.model_dump(mode="json")
                    for rc in config.ensemble_config.response_configuration
                ],
            },
            name="exp-sort",
        )

        # Create two ensembles with a small delay to ensure distinct started_at
        exp.create_ensemble(
            name="a-ens", ensemble_size=config.runpath_config.num_realizations
        )
        time.sleep(1)
        exp.create_ensemble(
            name="b-ens", ensemble_size=config.runpath_config.num_realizations
        )

    tool = ManageExperimentsPanel(
        config, notifier, config.runpath_config.num_realizations
    )
    storage_widget = tool.findChild(StorageWidget)
    assert storage_widget is not None

    tree_view = storage_widget._tree_view
    tree_view.expandAll()
    model = tree_view.model()
    exp_index = _find_root_row_by_name(model, "exp-sort")
    assert exp_index is not None

    def current_child_names():
        return _child_names_under(model, exp_index)

    tree_view.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    qtbot.waitUntil(lambda: current_child_names() == ["a-ens", "b-ens"], timeout=500)

    tree_view.sortByColumn(0, Qt.SortOrder.DescendingOrder)
    qtbot.waitUntil(lambda: current_child_names() == ["b-ens", "a-ens"], timeout=500)

    tree_view.sortByColumn(1, Qt.SortOrder.AscendingOrder)
    qtbot.waitUntil(lambda: current_child_names() == ["a-ens", "b-ens"], timeout=500)

    tree_view.sortByColumn(1, Qt.SortOrder.DescendingOrder)
    qtbot.waitUntil(lambda: current_child_names() == ["b-ens", "a-ens"], timeout=500)


RFT_OBSERVATION_SCHEMA = {
    "response_key": pl.String,
    "well": pl.String,
    "date": pl.String,
    "observation_key": pl.String,
    "east": pl.Float32,
    "north": pl.Float32,
    "tvd": pl.Float32,
    "md": pl.Float32,
    "zone": pl.String,
    "observations": pl.Float32,
    "std": pl.Float32,
    "radius": pl.Float32,
    "actual_zones": pl.List(pl.String),
    "well_connection_cell": pl.Array(pl.Int64, 3),
    "well_connection_cell_center": pl.Array(pl.Float32, 3),
    "expected_zone": pl.String,
    "qc_error": pl.String,
}

float_arr = partial(np.array, dtype=np.float32)


def test_that_rft_qc_widget_loads_observations_responses_and_file_rft(
    qtbot, mocked_files, mock_resfo_file
):
    # -- Fake observations (as returned by add_rft_metadata_and_qc) --
    observations = pl.DataFrame(
        {
            "response_key": ["WELL_A:2000-01-01:PRESSURE"] * 4
            + ["WELL_B:2001-01-01:PRESSURE"] * 3,
            "well": ["WELL_A"] * 4 + ["WELL_B"] * 3,
            "date": ["2000-01-01"] * 4 + ["2001-01-01"] * 3,
            "observation_key": ["OBS1", "OBS2", "OBS3", "OBS4", "OBS5", "OBS6", "OBS7"],
            "east": [100.0, 100.0, 110.0, 100.0, 200.0, 240.0, 180.0],
            "north": [100.0, 100.0, 100.0, 100.0, 200.0, 240.0, 180.0],
            "tvd": [100.0, 200.0, 290.0, 400.0, 300.0, 420.0, 480.0],
            "md": [110.0, 120.0, 130.0, 440.0, 330.0, 440.0, 500.0],
            "zone": ["wrong_zone"] + ["zone2"] * 5 + ["wrong_zone"],
            "observations": [111.0, 222.0, 333.0, 444.0, 555.0, 556.0, 655.0],
            "std": [5.0] * 7,
            "radius": [None] * 7,
            "actual_zones": [["zone2"]] * 7,
            "well_connection_cell": [
                [1, 1, 1],
                [1, 1, 2],
                [1, 1, 3],
                [1, 1, 4],
                None,
                [2, 2, 4],
                [2, 2, 5],
            ],
            "well_connection_cell_center": [
                [100.0, 100.0, 100.0],
                [100.0, 100.0, 200.0],
                [100.0, 100.0, 300.0],
                [100.0, 100.0, 400.0],
                None,
                [200.0, 200.0, 400.0],
                [200.0, 200.0, 500.0],
            ],
            "expected_zone": ["wrong_zone"] + ["zone2"] * 5 + ["wrong_zone"],
            "qc_error": [
                (
                    "expected zone 'wrong_zone' did not match any of the simulated "
                    "zones: zone2"
                )
            ]
            + [None] * 3
            + [("did not find grid coordinate for location 200, 200, 300")]
            + [None]
            + [
                (
                    "expected zone 'wrong_zone' did not match any of the simulated "
                    "zones: zone2"
                )
            ],
        },
        schema=RFT_OBSERVATION_SCHEMA,
    )

    # -- Fake responses (as returned by load_responses) --
    responses = pl.DataFrame(
        {
            "response_key": ["WELL_A:2000-01-01:PRESSURE"] * 5
            + ["WELL_B:2001-01-01:PRESSURE"],
            "well": ["WELL_A"] * 5 + ["WELL_B"],
            "date": ["2000-01-01"] * 5 + ["2001-01-01"],
            "property": ["PRESSURE"] * 4 + ["SWAT", "PRESSURE"],
            "time": [datetime.date(2000, 1, 1)] * 6,
            "depth": [100, 200, 300, 500, 500, 500],
            "values": [112.0, 223.0, 334.0, 556.0, 0.7, 555.0],
            "well_connection_cell": [
                [1, 1, 1],
                [1, 1, 2],
                [1, 1, 3],
                [1, 1, 5],
                [1, 1, 5],
                [2, 2, 5],
            ],
            "cell_center": [
                [100.0, 100.0, 100.0],
                [100.0, 100.0, 200.0],
                [100.0, 100.0, 300.0],
                [100.0, 100.0, 500.0],
                [100.0, 100.0, 500.0],
                [200.0, 200.0, 500.0],
            ],
            "cell_zones": [["zone2"]] * 6,
        },
        schema={
            "response_key": pl.String,
            "well": pl.String,
            "date": pl.String,
            "property": pl.String,
            "time": pl.Date,
            "depth": pl.Float32,
            "values": pl.Float32,
            "well_connection_cell": pl.Array(pl.Int64, 3),
            "cell_center": pl.Array(pl.Float32, 3),
            "cell_zones": pl.List(pl.String),
        },
    )

    # -- Mock ensemble --
    ensemble = MagicMock()
    experiment = MagicMock()
    experiment.observations = {
        "rft": observations.select(
            [
                c
                for c in observations.columns
                if c
                in {
                    "response_key",
                    "well",
                    "date",
                    "observation_key",
                    "east",
                    "north",
                    "tvd",
                    "md",
                    "zone",
                    "observations",
                    "std",
                    "radius",
                }
            ]
        )
    }
    type(ensemble).experiment = PropertyMock(return_value=experiment)
    ensemble.add_rft_metadata_and_qc.return_value = observations
    ensemble.load_responses.return_value = responses

    # -- Create widget and update --
    widget = RftQcWidget()
    qtbot.addWidget(widget)
    widget.update_realization(ensemble, 0)

    # -- Write a synthetic RFT + EGRID and load it through the widget --
    PATH_DOES_NOT_EXIST = "path/does/not/exist"
    egrid_path = f"{PATH_DOES_NOT_EXIST}/BASE.EGRID"
    rft_path = f"{PATH_DOES_NOT_EXIST}/BASE.RFT"
    zonemap_path = f"{PATH_DOES_NOT_EXIST}/zonemap.txt"
    mocked_files[zonemap_path] = "1 zone2\n2 zone2\n3 zone2\n4 zone2\n5 zone2\n"
    mock_resfo_file(egrid_path, create_egrid(2, 2, 5, 100, 100, 100, 50, 50, 50))
    pressure_values = np.linspace(110.0, 460.0, num=4, dtype=np.float32)
    mock_resfo_file(
        rft_path,
        [
            *cell_start(
                date=(1, 1, 2000),
                well_name=b"WELL_A_WITH_TYPO",
                ijks=[(1, 1, 4)],
            ),
            ("PRESSURE", [445.0]),
            *cell_start(
                date=(1, 1, 2000),
                well_name=b"WELL_C",
                ijks=[(1, 2, k) for k in range(1, 5)],
            ),
            ("PRESSURE", pressure_values),
            ("DEPTH   ", float_arr([100, 200, 300, 400])),
        ],
    )

    widget._current_runpath = PATH_DOES_NOT_EXIST
    widget._current_rft_config = RFTConfig(
        input_files=["BASE"],
        data_to_read={"*": {"*": ["*"]}},
        zonemap=Path("zonemap.txt"),
        approximate_missing_values=False,
    )
    widget._current_rft_file_path = widget._get_rft_file_path(
        widget._current_runpath, widget._current_rft_config
    )
    widget._on_toggle_file_rft(True)

    def _filter_values(list_widget) -> set[str]:
        return {
            list_widget.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(list_widget.count())
        }

    # Filters should be populated from observations, responses and the file RFT
    assert {"WELL_A", "WELL_B"} <= _filter_values(widget._filter_panel._well_list)
    assert {"2000-01-01", "2001-01-01"} <= _filter_values(
        widget._filter_panel._date_list
    )
    assert "PRESSURE" in _filter_values(widget._filter_panel._property_list)

    # Observations and responses should be stored
    assert not widget._observations.is_empty()
    assert not widget._responses.is_empty()

    # Toggle coordinate mode
    assert widget._use_utm is False
    widget._filter_panel._toggle_utm_coords.setChecked(True)
    assert widget._use_utm is True

    # File RFT was loaded and merged into the plot
    assert not widget._file_responses.is_empty()
    assert widget._current_rft_file_path is not None
    assert widget._load_rft_file_toggle.isEnabled()
    assert _PointStatus.FILE_RFT in _filter_values(widget._filter_panel._status_list)


def _obs_frame_for_attach_status(
    well_connection_cell: list[list[int] | None],
    expected_zone: list[str],
    actual_zones: list[list[str]],
    tag: list[str],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "response_key": ["WELL:2000-01-01:PRESSURE"] * len(tag),
            "well_connection_cell": well_connection_cell,
            "expected_zone": expected_zone,
            "actual_zones": actual_zones,
            "tag": tag,
        },
        schema={
            "response_key": pl.String,
            "well_connection_cell": pl.Array(pl.Int64, 3),
            "expected_zone": pl.String,
            "actual_zones": pl.List(pl.String),
            "tag": pl.String,
        },
    )


def test_that_attach_status_classifies_points_by_grid_zone_and_response():
    observations = _obs_frame_for_attach_status(
        well_connection_cell=[[1, 1, 1], [1, 1, 2], None, [1, 1, 3]],
        expected_zone=["zoneA", "zoneA", "zoneA", "wrong_zone"],
        actual_zones=[["zoneA"], ["zoneA"], ["zoneA"], ["zoneA"]],
        tag=["matched", "no_response", "not_in_grid", "invalid_zone"],
    )
    responses = pl.DataFrame(
        {
            "response_key": ["WELL:2000-01-01:PRESSURE"],
            "well_connection_cell": [[1, 1, 1]],
            "values": [42.0],
        },
        schema={
            "response_key": pl.String,
            "well_connection_cell": pl.Array(pl.Int64, 3),
            "values": pl.Float32,
        },
    )

    result = RftQcWidget._attach_status(observations, responses)

    status_by_tag = dict(
        zip(result["tag"].to_list(), result["status"].to_list(), strict=True)
    )
    assert status_by_tag == {
        "matched": _PointStatus.MATCHED.value,
        "no_response": _PointStatus.NO_RESPONSE.value,
        "not_in_grid": _PointStatus.NOT_IN_GRID.value,
        "invalid_zone": _PointStatus.INVALID_ZONE.value,
    }
    assert "values" not in result.columns


def test_that_attach_status_returns_empty_observations_unchanged():
    empty = _obs_frame_for_attach_status([], [], [], []).clear()
    responses = pl.DataFrame(
        {"response_key": [], "well_connection_cell": [], "values": []},
        schema={
            "response_key": pl.String,
            "well_connection_cell": pl.Array(pl.Int64, 3),
            "values": pl.Float32,
        },
    )

    result = RftQcWidget._attach_status(empty, responses)

    assert result.is_empty()
    assert "status" not in result.columns


def test_that_add_status_col_leaves_empty_dataframe_without_status_column():
    empty = pl.DataFrame(schema={"east": pl.Float32})

    result = _add_status_col_to_df(empty, _PointStatus.RESPONSE.value)

    assert "status" not in result.columns


def test_that_add_status_col_sets_status_on_nonempty_dataframe():
    df = pl.DataFrame({"east": [1.0, 2.0]})

    result = _add_status_col_to_df(df, _PointStatus.RESPONSE.value)

    assert result["status"].to_list() == [_PointStatus.RESPONSE.value] * 2


def test_that_unique_points_per_coordinate_keeps_highest_priority_status():
    df = pl.DataFrame(
        {
            "east": [1.0, 1.0, 2.0],
            "north": [1.0, 1.0, 2.0],
            "tvd": [1.0, 1.0, 2.0],
            "status": [
                _PointStatus.RESPONSE.value,
                _PointStatus.MATCHED.value,
                _PointStatus.NO_RESPONSE.value,
            ],
        }
    )

    result = _unique_points_per_coordinate(df)

    assert result.height == 2
    statuses = result["status"].to_list()
    # The duplicated coordinate keeps MATCHED (priority 0) and drops RESPONSE.
    assert _PointStatus.RESPONSE.value not in statuses
    assert sorted(statuses) == sorted(
        [_PointStatus.MATCHED.value, _PointStatus.NO_RESPONSE.value]
    )


def test_that_validate_required_columns_raises_listing_missing_columns():
    df = pl.DataFrame({"east": [1.0]})

    with pytest.raises(ValueError, match=r"Observations.*missing expected columns"):
        RftQcWidget._validate_required_columns(
            df, frozenset({"east", "north"}), context="Observations"
        )


def test_that_validate_required_columns_accepts_complete_dataframe():
    df = pl.DataFrame({"east": [1.0], "north": [1.0]})

    RftQcWidget._validate_required_columns(df, frozenset({"east", "north"}))


def test_that_get_rft_file_path_is_none_without_runpath_or_config():
    config = RFTConfig(input_files=["BASE"], data_to_read={}, zonemap=None)

    assert RftQcWidget._get_rft_file_path(None, config) is None
    assert RftQcWidget._get_rft_file_path("/run", None) is None


def test_that_get_rft_file_path_joins_runpath_and_expected_input_file():
    config = RFTConfig(input_files=["BASE"], data_to_read={}, zonemap=None)

    assert RftQcWidget._get_rft_file_path("/run", config) == Path("/run") / "BASE.RFT"


def _make_filter_panel() -> FilterPanel:
    return FilterPanel(
        on_item_selection_change=lambda: None,
        on_fit_to_selection_button_clicked=lambda: None,
        on_center_on_selected_button_clicked=lambda: None,
        on_toggle_utm_coords_clicked=lambda _checked: None,
        use_utm=False,
    )


def _select_only(list_widget, value: str) -> None:
    list_widget.clearSelection()
    for i in range(list_widget.count()):
        item = list_widget.item(i)
        if item.data(Qt.ItemDataRole.UserRole) == value:
            item.setSelected(True)


def test_that_filter_panel_filters_rows_to_selected_values(qtbot):
    panel = _make_filter_panel()
    qtbot.addWidget(panel)
    df = pl.DataFrame({"well": ["A", "B", "C"], "value": [1, 2, 3]})
    panel.populate_filters([df])

    _select_only(panel._well_list, "A")

    assert panel.apply_filter(df)["well"].to_list() == ["A"]


def test_that_filter_panel_facet_counts_reflect_other_filter_selections(qtbot):
    panel = _make_filter_panel()
    qtbot.addWidget(panel)
    df = pl.DataFrame({"well": ["A", "A", "B"], "status": ["S1", "S2", "S1"]})
    panel.populate_filters([df])

    _select_only(panel._well_list, "A")

    assert panel._facet_counts("status", [df]) == {"S1": 1, "S2": 1}


def test_that_toggling_file_rft_with_missing_file_keeps_file_responses_empty(qtbot):
    widget = RftQcWidget()
    qtbot.addWidget(widget)
    widget._current_runpath = "/does/not/exist"
    widget._current_rft_config = RFTConfig(
        input_files=["BASE"],
        data_to_read={"*": {"*": ["*"]}},
        zonemap=None,
    )
    widget._current_rft_file_path = Path("/does/not/exist/BASE.RFT")

    widget._on_toggle_file_rft(True)

    assert widget._load_rft_file is True
    assert widget._file_responses.is_empty()


def test_that_loading_realization_with_missing_observation_columns_raises(qtbot):
    # Observations missing the required "well_connection_cell_center" column.
    malformed_observations = pl.DataFrame(
        {
            "response_key": ["WELL:2000-01-01:PRESSURE"],
            "well_connection_cell": [[1, 1, 1]],
            "expected_zone": ["zoneA"],
            "actual_zones": [["zoneA"]],
            "east": [1.0],
            "north": [1.0],
            "tvd": [1.0],
        },
        schema={
            "response_key": pl.String,
            "well_connection_cell": pl.Array(pl.Int64, 3),
            "expected_zone": pl.String,
            "actual_zones": pl.List(pl.String),
            "east": pl.Float32,
            "north": pl.Float32,
            "tvd": pl.Float32,
        },
    )
    responses = pl.DataFrame(
        {
            "response_key": ["WELL:2000-01-01:PRESSURE"],
            "well_connection_cell": [[1, 1, 1]],
            "cell_center": [[1.0, 1.0, 1.0]],
            "values": [1.0],
        },
        schema={
            "response_key": pl.String,
            "well_connection_cell": pl.Array(pl.Int64, 3),
            "cell_center": pl.Array(pl.Float32, 3),
            "values": pl.Float32,
        },
    )

    ensemble = MagicMock()
    experiment = MagicMock()
    experiment.observations = {"rft": malformed_observations}
    type(ensemble).experiment = PropertyMock(return_value=experiment)
    ensemble.add_rft_metadata_and_qc.return_value = malformed_observations
    ensemble.load_responses.return_value = responses

    widget = RftQcWidget()
    qtbot.addWidget(widget)

    with pytest.raises(ValueError, match=r"Observations.*missing expected columns"):
        widget.update_realization(ensemble, 0)
