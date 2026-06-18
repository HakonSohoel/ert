from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from matplotlib.backend_bases import MouseEvent, PickEvent
from matplotlib.backends.backend_qt5agg import FigureCanvas  # type: ignore
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ert.config.response_config import InvalidResponseFile
from ert.config.rft_config import RFTConfig
from ert.runpaths import Runpaths
from ert.storage import Ensemble

if TYPE_CHECKING:
    from ert.config import ErtConfig

_FILTER_WIDTH = 160
_DETAILS_WIDTH = 230


class _PointStatus(StrEnum):
    MATCHED = "Has Response"
    INVALID_ZONE = "Invalid Zone"
    NOT_IN_GRID = "Not in grid"
    NO_RESPONSE = "No Response"
    RESPONSE = "Response"
    FILE_RFT = "File RFT"


TRANSPARENT = (0, 0, 0, 0)

# Okabe-Ito categorical palette (colorblind-safe, high-chroma for scatter plots).
OKABE_GREEN = "#009E73"
OKABE_ORANGE = "#E69F00"
OKABE_VERMILLION = "#D55E00"
OKABE_BLUE = "#0072B2"
OKABE_BLACK = "#000000"
OKABE_YELLOW = "#F0E442"

# Selection/hover rings use Okabe-Ito hues for palette consistency.
SELECTION_RING_COLOR = OKABE_BLUE
HOVER_RING_COLOR = OKABE_BLACK

_DEFAULT_STYLE: dict[str, Any] = {
    "facecolors": "gray",
    "edgecolors": "black",
    "linewidths": 0.5,
    "s": 50,
}

_POINT_STYLE: dict[str, Any] = {
    _PointStatus.MATCHED: {**_DEFAULT_STYLE, "facecolors": OKABE_GREEN},
    _PointStatus.INVALID_ZONE: {**_DEFAULT_STYLE, "facecolors": OKABE_ORANGE},
    _PointStatus.NOT_IN_GRID: {**_DEFAULT_STYLE, "facecolors": OKABE_VERMILLION},
    _PointStatus.NO_RESPONSE: {**_DEFAULT_STYLE, "facecolors": OKABE_VERMILLION},
    _PointStatus.RESPONSE: {**_DEFAULT_STYLE, "facecolors": "black", "s": 25},
    _PointStatus.FILE_RFT: {**_DEFAULT_STYLE, "facecolors": OKABE_YELLOW, "s": 25},
}


def _transform_point_style_to_overlay_style(
    point_style: dict[str, Any],
) -> dict[str, Any]:
    return {
        **point_style,
        "facecolors": TRANSPARENT,
        "edgecolors": point_style.get("facecolors", "gray"),
        "linewidths": 1.5,
        "s": 200,
    }


_CELL_CENTER_OVERLAY_STYLE: dict[str, Any] = {
    _PointStatus.MATCHED: _transform_point_style_to_overlay_style(
        _POINT_STYLE[_PointStatus.MATCHED]
    ),
    _PointStatus.INVALID_ZONE: _transform_point_style_to_overlay_style(
        _POINT_STYLE[_PointStatus.INVALID_ZONE]
    ),
    _PointStatus.NO_RESPONSE: _transform_point_style_to_overlay_style(
        _POINT_STYLE[_PointStatus.NO_RESPONSE]
    ),
    _PointStatus.RESPONSE: _transform_point_style_to_overlay_style(
        _POINT_STYLE[_PointStatus.RESPONSE]
    ),
}


_OBSERVATION_SCHEMA: dict[str, Any] = {
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
}

_OUTPUT_SCHEMA: dict[str, Any] = {
    "well": pl.String,
    "date": pl.String,
    "property": pl.String,
    "east": pl.Float32,
    "north": pl.Float32,
    "tvd": pl.Float32,
    "observations": pl.Float32,
    "std": pl.Float32,
    "expected_zone": pl.String,
    "actual_zones": pl.List(pl.String),
    "well_connection_cell": pl.Array(pl.Int64, 3),
    "simulated": pl.Float32,
    "qc_error": pl.String,
    "status": pl.String,
}


def _add_status_col_to_df(df: pl.DataFrame, status: str) -> pl.DataFrame:
    if not df.is_empty():
        df = df.with_columns(pl.lit(status).alias("status"))
    return df


def _ensure_well_connection_cell_center(df: pl.DataFrame) -> pl.DataFrame:
    if "well_connection_cell_center" in df.columns:
        return df
    if "cell_center" in df.columns:
        return df.rename({"cell_center": "well_connection_cell_center"})
    return df.with_columns(
        pl.lit(None).cast(pl.Array(pl.Float32, 3)).alias("well_connection_cell_center")
    )


# Lower number means higher priority when several points share a coordinate; the
# highest-priority status is the one kept for that coordinate.
_STATUS_PRIORITY: dict[str, int] = {
    _PointStatus.MATCHED: 0,
    _PointStatus.INVALID_ZONE: 1,
    _PointStatus.NOT_IN_GRID: 2,
    _PointStatus.NO_RESPONSE: 3,
    _PointStatus.RESPONSE: 4,
    _PointStatus.FILE_RFT: 5,
}


def _unique_points_per_coordinate(
    df: pl.DataFrame,
    coordinate_columns: Sequence[str] = ("east", "north", "tvd"),
) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("status").replace_strict(_STATUS_PRIORITY).alias("_priority")
        )
        .sort("_priority")
        .unique(subset=coordinate_columns, keep="first", maintain_order=True)
        .drop("_priority")
    )


class FilterPanel(QWidget):
    def __init__(
        self,
        on_item_selection_change,
        on_fit_to_selection_button_clicked,
        on_center_on_selected_button_clicked,
        on_toggle_utm_coords_clicked,
        use_utm,
    ) -> None:
        QWidget.__init__(self)

        self.setFixedWidth(_FILTER_WIDTH)
        filter_layout = QVBoxLayout()
        filter_layout.setContentsMargins(4, 4, 4, 4)
        filter_layout.setSpacing(2)

        self.dfs_to_filter: list[pl.DataFrame] = []

        self._well_list = self._make_filter_list(on_item_selection_change)
        self._date_list = self._make_filter_list(on_item_selection_change)
        self._property_list = self._make_filter_list(on_item_selection_change)
        self._status_list = self._make_filter_list(on_item_selection_change)

        self._filter_specs = {
            "well": self._well_list,
            "date": self._date_list,
            "property": self._property_list,
            "status": self._status_list,
        }

        for title, widget in [
            ("Well", self._well_list),
            ("Date", self._date_list),
            ("Property", self._property_list),
            ("Status", self._status_list),
        ]:
            lbl = QLabel(f"<b>{title}</b>")
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
            filter_layout.addWidget(lbl)
            filter_layout.addWidget(widget)
            filter_layout.addLayout(self._make_select_buttons(widget))

        filter_layout.addStretch()

        self._fit_button = QPushButton("Fit to selection")
        self._fit_button.clicked.connect(on_fit_to_selection_button_clicked)
        filter_layout.addWidget(self._fit_button)

        self._center_button = QPushButton("Center on selected")
        self._center_button.clicked.connect(on_center_on_selected_button_clicked)
        filter_layout.addWidget(self._center_button)

        self._toggle_utm_coords = QCheckBox("Show UTM coordinates")
        self._toggle_utm_coords.setChecked(use_utm)
        self._toggle_utm_coords.toggled.connect(on_toggle_utm_coords_clicked)
        filter_layout.addWidget(self._toggle_utm_coords)

        self.setLayout(filter_layout)

    def set_utm_enabled(self, enabled: bool) -> None:
        self._toggle_utm_coords.setEnabled(enabled)
        if not enabled and self._toggle_utm_coords.isChecked():
            self._toggle_utm_coords.blockSignals(True)
            self._toggle_utm_coords.setChecked(False)
            self._toggle_utm_coords.blockSignals(False)
        self._toggle_utm_coords.setToolTip(
            ""
            if enabled
            else "Some points are missing east/north/tvd coordinates; "
            "UTM view is unavailable"
        )

    def populate_filters(self, dataframes: list[pl.DataFrame]) -> None:
        self.dfs_to_filter = dataframes
        for col, list_widget in self._filter_specs.items():
            list_widget.blockSignals(True)
            list_widget.clear()
            values: set[str] = set()
            for df in self.dfs_to_filter:
                if col in df.columns:
                    values.update(
                        str(v) for v in df[col].drop_nulls().unique().to_list()
                    )
            for val in sorted(values, key=str):
                item = QListWidgetItem(val)
                item.setData(Qt.ItemDataRole.UserRole, val)
                list_widget.addItem(item)
            list_widget.selectAll()
            list_widget.blockSignals(False)

    def apply_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        return self._apply_filters_except(df, except_col=None)

    def refresh_facet_decorations(self) -> None:
        for col, list_widget in self._filter_specs.items():
            counts = self._facet_counts(col, self.dfs_to_filter)
            for item in [list_widget.item(i) for i in range(list_widget.count())]:
                if item is None:
                    continue
                val = item.data(Qt.ItemDataRole.UserRole)
                n = counts.get(str(val), 0)
                item.setText(f"{val}  ({n})")
                if n == 0:
                    item.setForeground(QColor("#999999"))
                else:
                    item.setData(Qt.ItemDataRole.ForegroundRole, None)

    def _make_filter_list(self, on_item_selection_change) -> QListWidget:
        lw = QListWidget()
        lw.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        lw.itemSelectionChanged.connect(on_item_selection_change)
        lw.itemSelectionChanged.connect(self.refresh_facet_decorations)
        return lw

    def _make_select_buttons(self, list_widget: QListWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 2)
        row.setSpacing(2)
        all_btn = QPushButton("Select All")
        none_btn = QPushButton("Clear")
        for btn in (all_btn, none_btn):
            btn.setFlat(True)
            btn.setStyleSheet("padding: 1px 4px; font-size: small;")
        all_btn.clicked.connect(list_widget.selectAll)
        none_btn.clicked.connect(list_widget.clearSelection)
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addStretch()
        return row

    def _apply_filters_except(
        self, df: pl.DataFrame, except_col: str | None
    ) -> pl.DataFrame:
        for col, list_widget in self._filter_specs.items():
            if col == except_col:
                continue
            selected = {
                item.data(Qt.ItemDataRole.UserRole)
                for item in list_widget.selectedItems()
            }
            if selected and col in df.columns:
                df = df.filter(pl.col(col).cast(pl.String).is_in(selected))
        return df

    def _facet_counts(self, col: str, dataframes: list[pl.DataFrame]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for df in dataframes:
            if df.is_empty() or col not in df.columns:
                continue
            filtered = self._apply_filters_except(df, except_col=col)
            for v, n in filtered.group_by(col).len().iter_rows():
                if v is None:
                    continue
                counts[str(v)] = counts.get(str(v), 0) + int(n)
        return counts


def _empty_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "well_connection_cell": pl.Array(pl.Int64, 3),
            "east": pl.Float32,
            "north": pl.Float32,
            "tvd": pl.Float32,
            "status": pl.String,
        }
    )


class RftPlot:
    def __init__(self, show_details: Callable) -> None:
        figure = Figure()
        self._canvas: FigureCanvas = FigureCanvas(figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._ax: Axes3D = figure.add_subplot(projection="3d")
        self._ax.mouse_init(rotate_btn=1, pan_btn=3, zoom_btn=2)
        self._canvas.mpl_connect("pick_event", self._on_pick)
        self._canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._default_limits: tuple[Any, Any, Any] | None = None

        self._point_artist: PathCollection | None = None
        self._point_coords: list[tuple[float, float, float]] = []
        self._display_points: pl.DataFrame = _empty_df()
        self._selection_artist: Any = None
        self._hover_artist: Any = None
        self._selected_index: int | None = None
        self._hover_index: int | None = None

        self._show_details = show_details

    @property
    def canvas(self) -> FigureCanvas:
        return self._canvas

    def redraw(
        self,
        obs_df: pl.DataFrame,
        response_df: pl.DataFrame,
        file_rft_df: pl.DataFrame,
        *,
        use_utm: bool = False,
        preserve_view: bool = True,
    ) -> None:
        prior_limits: tuple[Any, Any, Any] | None = None
        if preserve_view and self._default_limits is not None:
            prior_limits = (
                self._ax.get_xlim(),
                self._ax.get_ylim(),
                self._ax.get_zlim(),
            )
        self._ax.cla()
        self._ax.invert_zaxis()
        self._point_artist = None
        self._selection_artist = None
        self._hover_artist = None
        self._point_coords = []
        self._selected_index = None
        self._hover_index = None

        if obs_df.is_empty() and response_df.is_empty() and file_rft_df.is_empty():
            self._canvas.draw()
            return

        POINT_COLUMNS = ["well_connection_cell", "east", "north", "tvd", "status"]

        points = pl.concat(
            [
                obs_df.select(POINT_COLUMNS),
                response_df.select(POINT_COLUMNS),
                file_rft_df.select(POINT_COLUMNS),
            ]
        )

        if use_utm:
            # Keep only one point per coordinate for visualization
            points = _unique_points_per_coordinate(points)

            # Extract cell centers for observations where the observation location does
            # not exactly match the cell center.
            observation_points_cell_centers = obs_df.with_columns(
                pl.concat_arr("east", "north", "tvd")
                .cast(pl.Array(pl.Float32, 3))
                .alias("obs_location")
            ).filter(pl.col("well_connection_cell_center") != pl.col("obs_location"))

            # Prepare segments for showing relation between the observations and its
            # cell center (responses are always at cell centers)
            segments = [
                (tuple(p1), tuple(p2))
                for p1, p2 in zip(
                    observation_points_cell_centers[
                        "well_connection_cell_center"
                    ].to_list(),
                    observation_points_cell_centers["obs_location"].to_list(),
                    strict=True,
                )
            ]

            segment_colors = [
                _POINT_STYLE.get(_PointStatus(s), {}).get("facecolors", "gray")
                for s in observation_points_cell_centers["status"].to_list()
            ]

            observation_points_cell_centers = (
                observation_points_cell_centers.with_columns(
                    pl.col("well_connection_cell_center").arr.get(0).alias("east"),
                    pl.col("well_connection_cell_center").arr.get(1).alias("north"),
                    pl.col("well_connection_cell_center").arr.get(2).alias("tvd"),
                )
            )

            # Keep only one observation cell center overlay point per coordinate for
            # visualization
            observation_points_cell_centers = _unique_points_per_coordinate(
                observation_points_cell_centers.select(POINT_COLUMNS)
            )
            self._ax.set_xlabel("east", labelpad=6)
            self._ax.set_ylabel("north", labelpad=6)
            self._ax.set_zlabel("tvd", labelpad=6)

            point_style = {
                "facecolors": [
                    _POINT_STYLE.get(_PointStatus(s), {}).get("facecolors", "gray")
                    for s in points["status"].to_list()
                ]
                + [
                    _CELL_CENTER_OVERLAY_STYLE.get(_PointStatus(s), {}).get(
                        "facecolors", "gray"
                    )
                    for s in observation_points_cell_centers["status"].to_list()
                ],
                "s": [
                    _POINT_STYLE.get(_PointStatus(s), {}).get("s", 50)
                    for s in points["status"].to_list()
                ]
                + [
                    _CELL_CENTER_OVERLAY_STYLE.get(_PointStatus(s), {}).get("s", 50)
                    for s in observation_points_cell_centers["status"].to_list()
                ],
                "edgecolors": [
                    _POINT_STYLE.get(_PointStatus(s), {}).get("edgecolors", None)
                    for s in points["status"].to_list()
                ]
                + [
                    _CELL_CENTER_OVERLAY_STYLE.get(_PointStatus(s), {}).get(
                        "edgecolors", None
                    )
                    for s in observation_points_cell_centers["status"].to_list()
                ],
                "linewidths": [
                    _POINT_STYLE.get(_PointStatus(s), {}).get("linewidths", 1.0)
                    for s in points["status"].to_list()
                ]
                + [
                    _CELL_CENTER_OVERLAY_STYLE.get(_PointStatus(s), {}).get(
                        "linewidths", 1.0
                    )
                    for s in observation_points_cell_centers["status"].to_list()
                ],
            }
            points = pl.concat(
                [
                    points,
                    observation_points_cell_centers,
                ]
            )

            def get_utm_vectors(
                df: pl.DataFrame,
            ) -> tuple[list[float], list[float], list[float]]:
                return (
                    df["east"].to_list(),
                    df["north"].to_list(),
                    df["tvd"].to_list(),
                )

            # For testing observation east, north, tvd to cell center matching:
            if segments:
                lc = Line3DCollection(
                    segments,
                    colors=segment_colors,
                    linewidths=2,
                    linestyles="dashed",
                )
                self._ax.add_collection3d(lc)
            xs, ys, zs = get_utm_vectors(points)
            self._point_artist = self._ax.scatter(
                xs,
                ys,
                zs=zs,
                **point_style,
                picker=5,
                depthshade=False,
            )
            self._point_coords = list(zip(xs, ys, zs, strict=True))
        else:
            # Keep only one point per coordinate for visualization
            points = _unique_points_per_coordinate(
                points, coordinate_columns=["well_connection_cell"]
            )
            self._ax.set_xlabel("i", labelpad=6)
            self._ax.set_ylabel("j", labelpad=6)
            self._ax.set_zlabel("k", labelpad=6)
            points = points.filter(pl.col("well_connection_cell").is_not_null())
            point_style = {
                "facecolors": [
                    _POINT_STYLE.get(_PointStatus(s), {}).get("facecolors", "gray")
                    for s in points["status"].to_list()
                ],
                "s": [
                    _POINT_STYLE.get(_PointStatus(s), {}).get("s", 50)
                    for s in points["status"].to_list()
                ],
                "edgecolors": [
                    _POINT_STYLE.get(_PointStatus(s), {}).get("edgecolors", None)
                    for s in points["status"].to_list()
                ],
                "linewidths": [
                    _POINT_STYLE.get(_PointStatus(s), {}).get("linewidths", 1.0)
                    for s in points["status"].to_list()
                ],
            }
            xs = points["well_connection_cell"].arr.get(0).to_list()
            ys = points["well_connection_cell"].arr.get(1).to_list()
            zs = points["well_connection_cell"].arr.get(2).to_list()
            self._point_artist = self._ax.scatter(
                xs,
                ys,
                zs,
                **point_style,
                picker=5,
                depthshade=False,
            )
            self._point_coords = list(zip(xs, ys, zs, strict=True))

        self._display_points = points
        present: set[str] = set()
        if "status" in obs_df.columns:
            present.update(obs_df["status"].unique().to_list())
        if not file_rft_df.is_empty() and "status" in file_rft_df.columns:
            present.update(file_rft_df["status"].unique().to_list())
        for status in _PointStatus:
            if status.value in present:
                self._ax.scatter(
                    [],
                    [],
                    [],
                    facecolors=_POINT_STYLE[status].get("facecolors", "gray"),
                    edgecolors=_POINT_STYLE[status].get("edgecolors", TRANSPARENT),
                    label=status.value,
                    s=_POINT_STYLE[status].get("s", 50),
                )

        self._ax.legend(loc="upper left", fontsize="x-small")
        self._create_overlay_artists()
        self._canvas.draw()
        self._default_limits = (
            self._ax.get_xlim(),
            self._ax.get_ylim(),
            self._ax.get_zlim(),
        )
        # Keep the prior view only if every visible point still fits inside it;
        # otherwise let matplotlib's autoscaled limits stand.
        raw = self._data_bbox()
        if (
            prior_limits is not None
            and raw is not None
            and all(
                min(prior_lo, prior_hi) <= data_lo
                and max(prior_lo, prior_hi) >= data_hi
                for (prior_lo, prior_hi), (data_lo, data_hi) in zip(
                    prior_limits, raw, strict=True
                )
            )
        ):
            xlim, ylim, zlim = prior_limits
            self._ax.set_xlim(*xlim)
            self._ax.set_ylim(*ylim)
            self._ax.set_zlim(*zlim)
            self._canvas.draw_idle()

    def _data_bbox(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
        if not self._point_coords:
            return None
        xs = [p[0] for p in self._point_coords]
        ys = [p[1] for p in self._point_coords]
        zs = [p[2] for p in self._point_coords]
        return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))

    def _fit_view_to_visible(self) -> None:
        if self._default_limits is None:
            return
        xlim, ylim, zlim = self._default_limits
        self._ax.set_xlim(*xlim)
        self._ax.set_ylim(*ylim)
        self._ax.set_zlim(*zlim)
        self._canvas.draw_idle()

    def _center_on_selected(self) -> None:
        if self._selected_index is None or not (
            0 <= self._selected_index < len(self._point_coords)
        ):
            return
        cx, cy, cz = self._point_coords[self._selected_index]
        xlo, xhi = self._ax.get_xlim()
        ylo, yhi = self._ax.get_ylim()
        zlo, zhi = self._ax.get_zlim()
        hx = 0.5 * (xhi - xlo)
        hy = 0.5 * (yhi - ylo)
        hz = 0.5 * (zhi - zlo)
        self._ax.set_xlim(cx - hx, cx + hx)
        self._ax.set_ylim(cy - hy, cy + hy)
        self._ax.set_zlim(cz - hz, cz + hz)
        self._canvas.draw_idle()

    # ─── Pick / details ──────────────────────────────────────────────

    def _create_overlay_artists(self) -> None:
        self._selection_artist = self._ax.scatter(
            [],
            [],
            [],
            s=350,
            facecolors="none",
            edgecolors=SELECTION_RING_COLOR,
            linewidths=2.5,
            depthshade=False,
            zorder=20,
        )
        self._hover_artist = self._ax.scatter(
            [],
            [],
            [],
            s=300,
            facecolors="none",
            edgecolors=HOVER_RING_COLOR,
            linewidths=1.2,
            alpha=0.6,
            depthshade=False,
            zorder=19,
        )

    def _refresh_overlays(self) -> None:
        if self._selection_artist is None or self._hover_artist is None:
            return

        def _coords_for(
            idx: int | None,
        ) -> tuple[list[float], list[float], list[float]]:
            if idx is None or not (0 <= idx < len(self._point_coords)):
                return ([], [], [])
            x, y, z = self._point_coords[idx]
            return ([x], [y], [z])

        self._selection_artist._offsets3d = _coords_for(self._selected_index)
        hover_idx = (
            self._hover_index if self._hover_index != self._selected_index else None
        )
        self._hover_artist._offsets3d = _coords_for(hover_idx)
        self._canvas.draw_idle()

    def _on_pick(self, event: PickEvent) -> None:
        if self._point_artist is None or event.artist is not self._point_artist:
            return
        if not hasattr(event, "ind") or len(event.ind) == 0:
            return
        idx = int(event.ind[0])
        if not (0 <= idx < len(self._point_coords)):
            return
        self._selected_index = idx
        if 0 <= idx < len(self._display_points):
            self._show_details(self._display_points.row(idx, named=True))
        self._refresh_overlays()

    def _on_hover(self, event: MouseEvent) -> None:
        if (
            self._point_artist is None
            or event.inaxes is not self._ax
            or event.x is None
            or event.y is None
        ):
            new_index: int | None = None
        else:
            new_index = self._closest_point_within(event, radius_px=5)
        if new_index == self._hover_index:
            return
        self._hover_index = new_index
        self._refresh_overlays()

    def _closest_point_within(self, event: MouseEvent, radius_px: float) -> int | None:
        # Mirrors the picker=5 tolerance used by _on_pick so hover and pick
        # detection agree even for the smallest markers, whose marker paths
        # are otherwise too small for matplotlib's per-path contains() test.
        offsets = self._point_artist.get_offsets()
        if len(offsets) == 0:
            return None
        display_xy = self._point_artist.get_offset_transform().transform(offsets)
        dx = display_xy[:, 0] - event.x
        dy = display_xy[:, 1] - event.y
        d2 = dx * dx + dy * dy
        idx = int(d2.argmin())
        return idx if d2[idx] <= radius_px * radius_px else None

    def _on_scroll(self, event: MouseEvent) -> None:
        if event.inaxes is not self._ax or event.button not in {"up", "down"}:
            return
        scale = 0.8 if event.button == "up" else 1.25
        if self._selected_index is not None and 0 <= self._selected_index < len(
            self._point_coords
        ):
            cx, cy, cz = self._point_coords[self._selected_index]
        else:
            xlim, ylim, zlim = (
                self._ax.get_xlim(),
                self._ax.get_ylim(),
                self._ax.get_zlim(),
            )
            cx = 0.5 * (xlim[0] + xlim[1])
            cy = 0.5 * (ylim[0] + ylim[1])
            cz = 0.5 * (zlim[0] + zlim[1])
        for getter, setter, center in (
            (self._ax.get_xlim, self._ax.set_xlim, cx),
            (self._ax.get_ylim, self._ax.set_ylim, cy),
            (self._ax.get_zlim, self._ax.set_zlim, cz),
        ):
            lo, hi = getter()
            setter(center + (lo - center) * scale, center + (hi - center) * scale)
        self._canvas.draw_idle()


class RftQcWidget(QWidget):
    def __init__(self, ert_config: ErtConfig | None = None) -> None:
        QWidget.__init__(self)
        self._runpaths: Runpaths | None = (
            Runpaths.from_config(ert_config) if ert_config else None
        )
        self._current_rft_config: RFTConfig | None = None
        self._current_rft_file_path: Path | None = None
        self._current_ensemble: Ensemble | None = None
        self._current_realization: int | None = None
        self._current_runpath: str | None = None
        self._current_realization_loaded: bool = False

        self._observations: pl.DataFrame = _empty_df()
        self._responses: pl.DataFrame = _empty_df()
        self._file_responses: pl.DataFrame = _empty_df()

        self._load_rft_file: bool = False

        self._use_utm = False
        self._plot = RftPlot(self._show_details)
        self._filter_panel = FilterPanel(
            self._apply_filter_and_redraw,
            self._plot._fit_view_to_visible,
            self._plot._center_on_selected,
            self._on_coord_toggle,
            self._use_utm,
        )

        # ── Bottom row: File RFT load controls ────────────────────────
        file_rft_panel = QWidget()
        file_rft_layout = QHBoxLayout()
        file_rft_layout.setContentsMargins(4, 4, 4, 4)
        self._load_rft_file_toggle = QCheckBox("Load RFT file content into plot")
        self._load_rft_file_toggle.setChecked(self._load_rft_file)
        self._load_rft_file_toggle.setEnabled(False)
        self._load_rft_file_toggle.toggled.connect(self._on_toggle_file_rft)
        file_rft_layout.addWidget(self._load_rft_file_toggle)
        self._rft_file_label = QLabel("<i>No RFT file found</i>")
        self._rft_file_label.setTextFormat(Qt.TextFormat.RichText)
        self._rft_file_label.setWordWrap(True)
        self._rft_file_label.setStyleSheet("color: #555; font-size: small;")
        file_rft_layout.addWidget(self._rft_file_label, stretch=1)
        file_rft_panel.setLayout(file_rft_layout)

        # ── Right: point details panel ───────────────────────────────
        details_panel = QWidget()
        details_panel.setFixedWidth(_DETAILS_WIDTH)
        details_layout = QVBoxLayout()
        details_layout.setContentsMargins(4, 4, 4, 4)
        details_layout.addWidget(QLabel("<b>Point details</b>"))
        self._details = QTextEdit()
        self._details.setReadOnly(True)
        self._details.setPlaceholderText("Click a point to inspect.")
        details_layout.addWidget(self._details)
        details_panel.setLayout(details_layout)

        # ── Main layout ──────────────────────────────────────────────
        columns_layout = QHBoxLayout()
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.addWidget(self._filter_panel)
        columns_layout.addWidget(self._plot.canvas, stretch=1)
        columns_layout.addWidget(details_panel)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(columns_layout, stretch=1)
        layout.addWidget(file_rft_panel)
        self.setLayout(layout)

    def _update_load_rft_file_toggle_enabled_state(self) -> None:
        self._load_rft_file_toggle.setEnabled(
            self._load_rft_file
            or (
                self._current_rft_file_path is not None
                and self._current_rft_file_path.exists()
            )
        )

    def set_current_realization(self, ensemble: Ensemble, realization: int) -> None:
        self._current_ensemble = ensemble
        self._current_realization = realization
        self._current_runpath = self._get_runpath(ensemble, realization)
        self._current_realization_loaded = False
        self._current_rft_config = self._create_rft_config(ensemble)
        self._current_rft_file_path = self._get_rft_file_path(
            self._current_runpath, self._current_rft_config
        )
        self._update_load_rft_file_toggle_enabled_state()
        self._rft_file_label.setText(
            str(self._current_rft_file_path)
            if self._current_rft_file_path is not None
            and self._current_rft_file_path.exists()
            else "<i>No RFT file found</i>"
        )

    def load_current_realization(self) -> None:
        if (
            (not self._current_realization_loaded)
            and self._current_ensemble is not None
            and self._current_realization is not None
        ):
            self._load_realization(self._current_ensemble, self._current_realization)
            self._current_realization_loaded = True

    def update_realization(
        self, ensemble: Ensemble, realization: int, *, load: bool = True
    ) -> None:
        self.set_current_realization(ensemble, realization)
        if load:
            self.load_current_realization()

    def _get_runpath(self, ensemble: Ensemble, realization: int) -> str | None:
        if self._runpaths:
            return self._runpaths.get_paths([realization], ensemble.iteration)[0]
        return None

    def _create_rft_config(self, ensemble: Ensemble) -> RFTConfig | None:
        rft_cfg = ensemble.experiment.response_configuration.get("rft")
        if isinstance(rft_cfg, RFTConfig):
            return RFTConfig(
                input_files=rft_cfg.input_files,
                data_to_read={"*": {"*": ["*"]}},  # Read all available RFT data
                zonemap=rft_cfg.zonemap,
                approximate_missing_values=rft_cfg.approximate_missing_values,
            )
        return None

    @staticmethod
    def _get_rft_file_path(
        runpath: str | None, rft_config: RFTConfig | None
    ) -> Path | None:
        if runpath is not None and rft_config is not None:
            return Path(runpath) / rft_config.expected_input_files[0]
        return None

    @staticmethod
    def _validate_required_columns(
        obs_df: pl.DataFrame,
        required_columns: frozenset[str],
        context: str = "Observations",
    ) -> None:
        missing_cols = required_columns - set(obs_df.columns)
        if missing_cols:
            raise ValueError(
                f"{context} DataFrame is missing expected columns: {missing_cols}"
            )

    REQUIRED_OBS_COLUMNS = frozenset(
        {
            "well_connection_cell",
            "well_connection_cell_center",
            "east",
            "north",
            "tvd",
            "status",
        }
    )
    REQUIRED_RESPONSE_COLUMNS = frozenset(
        {
            "well_connection_cell",
            "well_connection_cell_center",
            "east",
            "north",
            "tvd",
            "status",
        }
    )
    REQUIRED_FILE_RESPONSE_COLUMNS = frozenset(
        {
            "well_connection_cell",
            "well_connection_cell_center",
            "east",
            "north",
            "tvd",
            "status",
        }
    )

    def _load_realization(self, ensemble: Ensemble, realization: int):
        observations = _ensure_well_connection_cell_center(
            self._get_observations(ensemble, realization)
        )
        responses = _ensure_well_connection_cell_center(
            ensemble.load_responses("rft", (realization,))
        ).with_columns(
            pl.col("well_connection_cell_center").arr.get(0).alias("east"),
            pl.col("well_connection_cell_center").arr.get(1).alias("north"),
            pl.col("well_connection_cell_center").arr.get(2).alias("tvd"),
        )
        self._observations = self._attach_status(observations, responses)
        self._responses = _add_status_col_to_df(responses, _PointStatus.RESPONSE)
        self._validate_required_columns(
            self._observations,
            self.REQUIRED_OBS_COLUMNS,
            context="Observations",
        )
        self._validate_required_columns(
            self._responses,
            self.REQUIRED_RESPONSE_COLUMNS,
            context="Responses",
        )
        if self._load_rft_file:
            self._load_file_rft()
        self._filter_panel.populate_filters(self._dfs_for_filters())
        self._refresh_utm_availability()
        self._apply_filter_and_redraw(preserve_view=False)

    def _apply_filter_and_redraw(self, *, preserve_view=True) -> None:
        obs_df = self._filter_panel.apply_filter(self._observations)
        response_df = self._filter_panel.apply_filter(self._responses)
        file_rft_df = self._filter_panel.apply_filter(self._file_responses)
        self._plot.redraw(
            obs_df,
            response_df,
            file_rft_df,
            use_utm=self._use_utm,
            preserve_view=preserve_view,
        )

    @staticmethod
    def _attach_status(
        observations: pl.DataFrame, responses: pl.DataFrame
    ) -> pl.DataFrame:
        if observations.is_empty():
            return observations
        if responses.is_empty():
            joined = observations.with_columns(pl.lit(None).alias("values"))
        else:
            joined = observations.join(
                responses.select("response_key", "well_connection_cell", "values"),
                on=["response_key", "well_connection_cell"],
                how="left",
            )
        return joined.with_columns(
            pl.when(pl.col("well_connection_cell").is_null())
            .then(pl.lit(_PointStatus.NOT_IN_GRID.value))
            .when(~RFTConfig.is_zone_valid())
            .then(pl.lit(_PointStatus.INVALID_ZONE.value))
            .when(pl.col("values").is_not_null())
            .then(pl.lit(_PointStatus.MATCHED.value))
            .otherwise(pl.lit(_PointStatus.NO_RESPONSE.value))
            .alias("status")
        ).drop("values")

    # ─── Data building ───────────────────────────────────────────────

    def _get_observations(self, ensemble: Ensemble, realization: int) -> pl.DataFrame:
        observations = ensemble.experiment.observations.get(
            "rft", pl.DataFrame(schema=_OBSERVATION_SCHEMA)
        )
        return ensemble.add_rft_metadata_and_qc(observations, realization).with_columns(
            pl.col("response_key").str.split(":").list.last().alias("property")
        )

    def _dfs_for_filters(self) -> list[pl.DataFrame]:
        return [self._observations, self._responses, self._file_responses]

    def _utm_coords_available(self) -> bool:
        coord_columns = ("east", "north", "tvd")
        for df in self._dfs_for_filters():
            if df.is_empty():
                continue
            if not all(c in df.columns for c in coord_columns):
                return False
            if any(df.select(coord_columns).null_count().row(0)):
                return False
        return True

    def _refresh_utm_availability(self) -> None:
        utm_available = self._utm_coords_available()
        if not utm_available:
            self._use_utm = False
        self._filter_panel.set_utm_enabled(utm_available)

    # ─── Coordinate toggle ─────────────────────────────────────────────

    def _on_coord_toggle(self, checked: bool) -> None:
        self._use_utm = checked
        self._apply_filter_and_redraw(preserve_view=False)

    # ─── File RFT loading ──────────────────────────────────────────────

    def _on_toggle_file_rft(self, checked: bool) -> None:
        self._load_rft_file = checked
        if self._load_rft_file:
            self._load_file_rft()
        else:
            self._file_responses = _empty_df()
        self._update_load_rft_file_toggle_enabled_state()
        self._filter_panel.populate_filters(self._dfs_for_filters())
        self._refresh_utm_availability()
        self._apply_filter_and_redraw(preserve_view=False)

    def _load_file_rft(self) -> None:
        if (
            self._current_rft_config is not None
            and self._current_runpath is not None
            and self._current_rft_file_path is not None
            and self._current_rft_file_path.exists()
        ):
            try:
                rft_file_df = _ensure_well_connection_cell_center(
                    self._current_rft_config.read_from_file(self._current_runpath, 0, 0)
                ).with_columns(
                    pl.col("well_connection_cell_center").arr.get(0).alias("east"),
                    pl.col("well_connection_cell_center").arr.get(1).alias("north"),
                    pl.col("well_connection_cell_center").arr.get(2).alias("tvd"),
                )
                self._file_responses = _add_status_col_to_df(
                    rft_file_df, _PointStatus.FILE_RFT
                )

            except InvalidResponseFile as err:
                self._file_responses = _empty_df()
                self._rft_file_label.setText(str(err))

            self._validate_required_columns(
                self._file_responses,
                self.REQUIRED_FILE_RESPONSE_COLUMNS,
                context="File Responses",
            )

    def _show_details(self, point: dict[str, Any]) -> None:
        def _fmt(v: Any, label: str) -> str | None:
            if v is None:
                return None
            if isinstance(v, float):
                return f"{v:.4g}"
            if isinstance(v, list):
                return ", ".join(str(x) for x in v) or None
            return str(v)

        def _labeled_item(val: Any, label: str) -> str:
            val = _fmt(val, label)
            if val:
                return f"<tr><td>{label}:</td><td>{val}</td></tr>"
            return ""

        def _status_note(val: Any) -> str:
            return (
                "<tr><td colspan=2><i style='color:gray;font-size:small;'>"
                + str(val)
                + "</i></td></tr>"
                if val
                else ""
            )

        cell = point.get("well_connection_cell")
        cell_str = (
            f"i={cell[0]}, j={cell[1]}, k={cell[2]}"
            if cell is not None
            else "Not in grid"
        )

        point_observations = self._observations.filter(
            (pl.col("well_connection_cell") == point.get("well_connection_cell"))
            | (
                (pl.col("east") == point.get("east"))
                & (pl.col("north") == point.get("north"))
                & (pl.col("tvd") == point.get("tvd"))
            )
        )

        point_responses = self._responses.filter(
            pl.col("well_connection_cell") == point.get("well_connection_cell")
        )

        combined = point_observations.join(
            point_responses,
            on=[
                "response_key",
                "well",
                "date",
                "property",
                "well_connection_cell",
                "well_connection_cell_center",
            ],
            how="full",
            coalesce=True,
        )

        zones: set[str] = set()
        for zone_col in ("actual_zones", "cell_zones"):
            if zone_col in combined.columns:
                for entries in combined[zone_col].drop_nulls().to_list():
                    zones.update(entries)

        def _obs_coords(row: dict[str, Any]) -> str | None:
            obs_location = (
                row.get("east"),
                row.get("north"),
                row.get("tvd"),
            )
            if None in obs_location:
                return None
            return ",".join([str(x) for x in obs_location])

        inner_html = ""
        for row in combined.iter_rows(named=True):
            inner_html += f"""
                <tr><td colspan=2><b>{row.get("well")} - {row.get("date")} - {row.get("property")}</b></td></tr>
                {_labeled_item(row.get("observations"), "Observation")}
                {_labeled_item(row.get("values"), "Response")}
                {_labeled_item(_obs_coords(row), "Observation Coordinates")}
                {_labeled_item(row.get("std"), "Error")}
                {_labeled_item(row.get("zone"), "Expected Zone")}
                {_labeled_item(row.get("status"), "Status")}
                {_status_note(row.get("qc_error"))}
            """  # noqa: E501

        if not self._file_responses.is_empty() and cell is not None:
            file_rows = self._file_responses.filter(
                pl.col("well_connection_cell") == point.get("well_connection_cell")
            )
            for row in file_rows.iter_rows(named=True):
                inner_html += f"""
                <tr><td colspan=2><b>[File RFT] {row.get("well")} - {row.get("date")} - {row.get("property")}</b></td></tr>
                {_labeled_item(row.get("values"), "File value")}
                """  # noqa: E501
        utm_coordinates = (
            f"<tr><td>Utm Coordinates: </td><td>{point.get('east')}, "
            f"{point.get('north')}, {point.get('tvd')}</td></tr>"
        )
        zones_label = "Zones" if len(zones) > 1 else "Zone"
        html = f"""
                <table style="border-spacing:2px 4px;">
                    <tr><td>Grid cell: </td><td>{cell_str}</td></tr>
                    {utm_coordinates if self._use_utm else ""}
                    {_labeled_item(sorted(zones), zones_label)}
                    {inner_html}
                </table>
        """
        self._details.setHtml(html)
