from __future__ import annotations

from pathlib import Path
import csv
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


SURFACE_RESOLUTION = 35
FIXED_DEFAULTS = {"R1": 0.04, "R5": 0.11}
MODEL_DIR = Path(__file__).resolve().parent / "tmp_unified_test" / "太陽光パネル有り"

MODEL_CONFIG = {
    "t1": {"csv": "t1_NN_pv.csv", "input_mode": "direct10"},
    "t2": {"csv": "t2_NN_pv.csv", "input_mode": "full12_ignored"},
    "t3": {"csv": "t3_NN_pv.csv", "input_mode": "direct10"},
    "t4": {"csv": "t4_NN_pv.csv", "input_mode": "full12_ignored"},
}

VARIABLE_CONFIG = [
    {"key": "J", "range_index": 0, "label": "J"},
    {"key": "Je", "range_index": 1, "label": "Je"},
    {"key": "R1", "range_index": None, "label": "R1"},
    {"key": "R2", "range_index": 2, "label": "R2"},
    {"key": "R4", "range_index": 3, "label": "R4"},
    {"key": "R5", "range_index": None, "label": "R5"},
    {"key": "as", "range_index": 4, "label": "as"},
    {"key": "al", "range_index": 5, "label": "al"},
    {"key": "ep1", "range_index": 6, "label": "ε1"},
    {"key": "ep2", "range_index": 7, "label": "ε2"},
    {"key": "ti", "range_index": 8, "label": "ti"},
    {"key": "to", "range_index": 9, "label": "to"},
]

ACTIVE_VARIABLES = [item for item in VARIABLE_CONFIG if item["range_index"] is not None]
ACTIVE_KEYS = [item["key"] for item in ACTIVE_VARIABLES]


class ResponseSurfaceModel:
    def __init__(self, csv_path: Path, input_mode: str) -> None:
        self.input_mode = input_mode
        self.n_input = 10
        if not csv_path.exists():
            raise FileNotFoundError(f"モデル定義ファイルが見つかりません: {csv_path}")

        with csv_path.open(newline="", encoding="utf-8-sig") as csvfile:
            filereader = csv.reader(csvfile)
            next(filereader)
            next(filereader)

            self.x_range = [[0.0, 0.0] for _ in range(10)]
            for i in range(10):
                self.x_range[i] = [float(value) for value in next(filereader)]

            next(filereader)
            self.y_range = [float(next(filereader)[0]) for _ in range(2)]

            next(filereader)
            self.out_range = [float(next(filereader)[0]) for _ in range(2)]

            next(filereader)
            self.w1 = [[float(value) for value in next(filereader)] for _ in range(80)]

            next(filereader)
            self.b1 = [float(next(filereader)[0]) for _ in range(80)]

            next(filereader)
            self.w2 = [[float(value) for value in next(filereader)] for _ in range(1)]

            next(filereader)
            self.b2 = [float(next(filereader)[0]) for _ in range(1)]

    def evaluate(self, values: dict[str, float]) -> float:
        if self.input_mode == "direct10":
            xx = [
                values["J"],
                values["Je"],
                values["R2"],
                values["R4"],
                values["as"],
                values["al"],
                values["ep1"],
                values["ep2"],
                values["ti"],
                values["to"],
            ]
        else:
            xx = [
                values["J"],
                values["Je"],
                values["R2"],
                values["R4"],
                values["as"],
                values["al"],
                values["ep1"],
                values["ep2"],
                values["ti"],
                values["to"],
            ]

        xn = [0.0 for _ in range(self.n_input)]
        for i in range(self.n_input):
            xn[i] = (
                2 * xx[i] - self.x_range[i][0] - self.x_range[i][1]
            ) / (self.x_range[i][1] - self.x_range[i][0])

        n1 = [0.0 for _ in range(len(self.w1))]
        for i in range(len(self.w1)):
            n1[i] = self.b1[i]
            for j in range(len(self.w1[0])):
                n1[i] += self.w1[i][j] * xn[j]

        y1 = [0.0 for _ in range(len(self.w1))]
        for i in range(len(self.w1)):
            try:
                exp_value = math.exp(-2.0 * n1[i])
            except OverflowError:
                exp_value = math.inf

            if exp_value == math.inf:
                y1[i] = -1.0
            else:
                y1[i] = (1.0 - exp_value) / (1.0 + exp_value)

        n2 = [0.0 for _ in range(len(self.w2))]
        for i in range(len(self.w2)):
            n2[i] = self.b2[i]
            for j in range(len(self.w2[0])):
                n2[i] += self.w2[i][j] * y1[j]

        return self.y_range[0] + (
            (self.y_range[1] - self.y_range[0])
            / (self.out_range[1] - self.out_range[0])
            * (n2[0] - self.out_range[0])
        )


def midpoint(value_range: list[float]) -> float:
    return float((value_range[0] + value_range[1]) / 2)


def input_step(value_range: list[float]) -> float:
    span = float(value_range[1] - value_range[0])
    return max(span / 100.0, 0.0001) if span else 0.1


def variable_label(key: str) -> str:
    for item in VARIABLE_CONFIG:
        if item["key"] == key:
            return item["label"]
    return key


def initialize_session_state(model: ResponseSurfaceModel) -> None:
    pending_values = st.session_state.pop("pending_clicked_values", None)
    if isinstance(pending_values, dict):
        for key, value in pending_values.items():
            st.session_state[f"input_{key}"] = float(value)

    for item in ACTIVE_VARIABLES:
        state_key = f"input_{item['key']}"
        if state_key not in st.session_state:
            st.session_state[state_key] = midpoint(model.x_range[item["range_index"]])

    for key, value in FIXED_DEFAULTS.items():
        state_key = f"input_{key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = float(value)

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = "t2"
    if "x_axis" not in st.session_state:
        st.session_state["x_axis"] = "R2"
    if "y_axis" not in st.session_state:
        st.session_state["y_axis"] = "R4"
    if "last_clicked_point" not in st.session_state:
        st.session_state["last_clicked_point"] = ""


def apply_preset(model: ResponseSurfaceModel, preset: str) -> None:
    for item in ACTIVE_VARIABLES:
        value_range = model.x_range[item["range_index"]]
        if preset == "minimum":
            value = float(value_range[0])
        elif preset == "maximum":
            value = float(value_range[1])
        else:
            value = midpoint(value_range)
        st.session_state[f"input_{item['key']}"] = value


def current_values() -> dict[str, float]:
    values = {key: float(st.session_state[f"input_{key}"]) for key in FIXED_DEFAULTS}
    for item in ACTIVE_VARIABLES:
        values[item["key"]] = float(st.session_state[f"input_{item['key']}"])
    return values


def compute_sat(values: dict[str, float]) -> float:
    return values["to"] + (1.0 / 23.0) * (values["as"] * values["J"] - values["al"] * values["Je"])


def compute_r3(values: dict[str, float]) -> float:
    try:
        rad_term = (
            1.0 / (max(values["ep1"], 1e-10) ** -1 + max(values["ep2"], 1e-10) ** -1 - 1.0)
        ) * 4.0 * (293.0**3) * 5.67e-8
        return 1.0 / (25.0 + rad_term)
    except ZeroDivisionError:
        return 0.0


def build_surface(
    model: ResponseSurfaceModel,
    x_key: str,
    y_key: str,
    values: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_config = next(item for item in ACTIVE_VARIABLES if item["key"] == x_key)
    y_config = next(item for item in ACTIVE_VARIABLES if item["key"] == y_key)

    x_axis_grid = np.linspace(
        model.x_range[x_config["range_index"]][0],
        model.x_range[x_config["range_index"]][1],
        SURFACE_RESOLUTION,
    )
    y_axis_grid = np.linspace(
        model.x_range[y_config["range_index"]][0],
        model.x_range[y_config["range_index"]][1],
        SURFACE_RESOLUTION,
    )
    x_mesh, y_mesh = np.meshgrid(x_axis_grid, y_axis_grid)
    z_values = np.zeros((SURFACE_RESOLUTION, SURFACE_RESOLUTION))

    for i in range(SURFACE_RESOLUTION):
        for j in range(SURFACE_RESOLUTION):
            surface_values = dict(values)
            surface_values[x_key] = float(x_mesh[i, j])
            surface_values[y_key] = float(y_mesh[i, j])
            z_values[i, j] = model.evaluate(surface_values)

    return x_mesh, y_mesh, z_values


def apply_chart_selection(selection: object, x_key: str, y_key: str) -> None:
    if not isinstance(selection, dict):
        return
    points = selection.get("selection", {}).get("points", [])
    if not points:
        return

    point = points[0]
    x_value = point.get("x")
    y_value = point.get("y")
    point_index = point.get("point_index")
    if x_value is None or y_value is None or point_index is None:
        return

    click_signature = f"{point_index}|{x_key}|{y_key}|{st.session_state['selected_model']}"
    if st.session_state["last_clicked_point"] == click_signature:
        return

    st.session_state["pending_clicked_values"] = {x_key: float(x_value), y_key: float(y_value)}
    st.session_state["last_clicked_point"] = click_signature
    st.rerun()


def build_summary_dataframe(
    model: ResponseSurfaceModel,
    values: dict[str, float],
    x_key: str,
    y_key: str,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for item in VARIABLE_CONFIG:
        value = values[item["key"]]
        if item["range_index"] is not None:
            value_range = model.x_range[item["range_index"]]
            role = "X軸" if item["key"] == x_key else "Y軸" if item["key"] == y_key else "固定値"
            rows.append(
                {
                    "変数": item["label"],
                    "現在値": f"{value:.3f}",
                    "下限": f"{value_range[0]:.3f}",
                    "上限": f"{value_range[1]:.3f}",
                    "役割": role,
                }
            )
        else:
            rows.append(
                {
                    "変数": item["label"],
                    "現在値": f"{value:.3f}",
                    "下限": "-",
                    "上限": "-",
                    "役割": "固定値",
                }
            )
    return pd.DataFrame(rows)


def render_sidebar(model: ResponseSurfaceModel) -> None:
    st.sidebar.header("入力条件")
    st.sidebar.selectbox("表示する温度", list(MODEL_CONFIG.keys()), key="selected_model")

    preset_col1, preset_col2, preset_col3 = st.sidebar.columns(3)
    if preset_col1.button("最小"):
        apply_preset(model, "minimum")
    if preset_col2.button("中央"):
        apply_preset(model, "midpoint")
    if preset_col3.button("最大"):
        apply_preset(model, "maximum")

    st.sidebar.markdown("---")
    st.sidebar.subheader("主要変数")
    for item in ACTIVE_VARIABLES:
        value_range = model.x_range[item["range_index"]]
        st.sidebar.number_input(
            item["label"],
            min_value=float(value_range[0]),
            max_value=float(value_range[1]),
            step=input_step(value_range),
            format="%.3f",
            key=f"input_{item['key']}",
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("表示設定")
    st.sidebar.selectbox("X軸", ACTIVE_KEYS, key="x_axis", format_func=variable_label)
    y_candidates = [key for key in ACTIVE_KEYS if key != st.session_state["x_axis"]]
    if st.session_state["y_axis"] == st.session_state["x_axis"]:
        st.session_state["y_axis"] = "R4" if st.session_state["x_axis"] != "R4" else "R2"
    st.sidebar.selectbox("Y軸", y_candidates, key="y_axis", format_func=variable_label)


def render_intro() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(28, 122, 98, 0.12), transparent 28%),
                linear-gradient(180deg, #f7faf8 0%, #eff5f2 100%);
        }
        .hero-card {
            background: linear-gradient(135deg, #0f4c45 0%, #1f7a6c 100%);
            color: #f6f8f6;
            padding: 1.1rem 1.3rem;
            border-radius: 16px;
            margin-bottom: 1rem;
        }
        .hero-card h1 { margin: 0 0 0.2rem 0; font-size: 1.85rem; }
        .hero-card p { margin: 0.15rem 0; }
        </style>
        <div class="hero-card">
            <h1>太陽光パネル付き 応答曲面ダッシュボード</h1>
            <p>t1 から t4 までを 1 つの画面で切り替えて表示できます。</p>
            <p>右の等高線グラフで選んだ位置が入力値に反映されます。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_model(model_key: str) -> ResponseSurfaceModel:
    model_info = MODEL_CONFIG[model_key]
    csv_path = MODEL_DIR / model_info["csv"]
    return ResponseSurfaceModel(csv_path, model_info["input_mode"])


def main() -> None:
    st.set_page_config(page_title="応答曲面ダッシュボード", layout="wide")

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = "t2"

    try:
        model = load_model(st.session_state["selected_model"])
    except Exception as error:
        st.error(str(error))
        return

    initialize_session_state(model)
    render_sidebar(model)

    selected_model = st.session_state["selected_model"]
    try:
        model = load_model(selected_model)
    except Exception as error:
        st.error(str(error))
        return

    render_intro()

    values = current_values()
    prediction = model.evaluate(values)
    sat_value = compute_sat(values)
    r3_value = compute_r3(values)
    x_key = st.session_state["x_axis"]
    y_key = st.session_state["y_axis"]

    top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns(5)
    top_col1.metric(f"推定温度 {selected_model}", f"{prediction:.3f}")
    top_col2.metric("SAT", f"{sat_value:.3f}")
    top_col3.metric("R3", f"{r3_value:.3f}")
    top_col4.metric("R1", f"{values['R1']:.3f}")
    top_col5.metric("R5", f"{values['R5']:.3f}")
    st.caption("赤い丸点が、現在入力している条件に対応する位置です。")

    chart_tab, table_tab = st.tabs(["グラフ", "入力一覧"])

    with chart_tab:
        x_mesh, y_mesh, z_values = build_surface(model, x_key, y_key, values)
        current_x = values[x_key]
        current_y = values[y_key]
        current_z = prediction
        z_floor = float(np.min(z_values))

        left_col, right_col = st.columns([1.2, 1])
        with left_col:
            surface_fig = go.Figure(
                data=[
                    go.Surface(
                        x=x_mesh,
                        y=y_mesh,
                        z=z_values,
                        colorscale="Viridis",
                        opacity=0.95,
                        colorbar={"title": selected_model},
                    ),
                    go.Scatter3d(
                        x=[current_x, current_x],
                        y=[current_y, current_y],
                        z=[z_floor, current_z],
                        mode="lines",
                        line={"color": "#b4232c", "width": 2},
                        showlegend=False,
                    ),
                    go.Scatter3d(
                        x=[current_x],
                        y=[current_y],
                        z=[current_z],
                        mode="markers+text",
                        marker={"size": 5, "color": "#b4232c", "symbol": "circle"},
                        text=[f"{selected_model}={current_z:.3f}"],
                        textposition="top center",
                        name="現在点",
                    ),
                ]
            )
            surface_fig.update_layout(
                scene={
                    "xaxis_title": variable_label(x_key),
                    "yaxis_title": variable_label(y_key),
                    "zaxis_title": selected_model,
                },
                margin={"l": 0, "r": 0, "b": 0, "t": 10},
                height=620,
            )
            st.plotly_chart(surface_fig, use_container_width=True)

        with right_col:
            contour_fig = go.Figure(
                data=[
                    go.Contour(
                        x=np.unique(x_mesh),
                        y=np.unique(y_mesh),
                        z=z_values,
                        colorscale="Viridis",
                        contours={"showlabels": True},
                        colorbar={"title": selected_model},
                    ),
                    go.Scatter(
                        x=x_mesh.flatten(),
                        y=y_mesh.flatten(),
                        mode="markers",
                        marker={"size": 10, "color": "rgba(30, 60, 80, 0.01)"},
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=[current_x],
                        y=[current_y],
                        mode="markers+text",
                        marker={"size": 7, "color": "#b4232c", "symbol": "circle"},
                        text=["現在点"],
                        textposition="top center",
                        name="現在点",
                    ),
                ]
            )
            contour_fig.update_layout(
                xaxis_title=variable_label(x_key),
                yaxis_title=variable_label(y_key),
                height=620,
                margin={"l": 10, "r": 10, "b": 10, "t": 10},
                clickmode="event+select",
            )
            selection = st.plotly_chart(
                contour_fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="contour_chart",
            )
            st.caption("右の等高線グラフで位置を選ぶと、その場所に近い条件へ更新されます。")
            apply_chart_selection(selection, x_key, y_key)

    with table_tab:
        st.dataframe(
            build_summary_dataframe(model, values, x_key, y_key),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
