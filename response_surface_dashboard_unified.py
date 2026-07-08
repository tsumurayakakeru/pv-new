from __future__ import annotations

from pathlib import Path
import csv
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st


SURFACE_RESOLUTION = 35
MODEL_BASE_DIR = Path(__file__).resolve().parent / "tmp_unified_test"

PANEL_CONFIG = {
    "pv": {
        "label": "太陽光パネル有り",
        "folder": "太陽光パネル有り",
        "default_model": "t2",
        "default_x_axis": "R2",
        "default_y_axis": "R4",
        "fixed_defaults": {"R1": 0.04, "R5": 0.11},
        "variables": [
            {"key": "J", "range_index": 0, "label": "J"},
            {"key": "Je", "range_index": 1, "label": "Je"},
            {"key": "R1", "range_index": None, "label": "R1"},
            {"key": "R2", "range_index": 2, "label": "R2"},
            {"key": "R4", "range_index": 3, "label": "R4"},
            {"key": "R5", "range_index": None, "label": "R5"},
            {"key": "as", "range_index": 4, "label": "as"},
            {"key": "al", "range_index": 5, "label": "al"},
            {"key": "ep1", "range_index": 6, "label": "\u03b51"},
            {"key": "ep2", "range_index": 7, "label": "\u03b52"},
            {"key": "ti", "range_index": 8, "label": "ti"},
            {"key": "to", "range_index": 9, "label": "to"},
        ],
        "models": {
            "t1": {"csv": "t1_NN_pv.csv", "n_input": 10, "hidden_size": 80, "input_keys": ["J", "Je", "R2", "R4", "as", "al", "ep1", "ep2", "ti", "to"]},
            "t2": {"csv": "t2_NN_pv.csv", "n_input": 10, "hidden_size": 80, "input_keys": ["J", "Je", "R2", "R4", "as", "al", "ep1", "ep2", "ti", "to"]},
            "t3": {"csv": "t3_NN_pv.csv", "n_input": 10, "hidden_size": 80, "input_keys": ["J", "Je", "R2", "R4", "as", "al", "ep1", "ep2", "ti", "to"]},
            "t4": {"csv": "t4_NN_pv.csv", "n_input": 10, "hidden_size": 80, "input_keys": ["J", "Je", "R2", "R4", "as", "al", "ep1", "ep2", "ti", "to"]},
        },
    },
    "notpv": {
        "label": "太陽光パネル無し",
        "folder": "太陽光パネル無し",
        "default_model": "t3",
        "default_x_axis": "as",
        "default_y_axis": "R4",
        "fixed_defaults": {"R1": 0.04, "R5": 0.11},
        "variables": [
            {"key": "J", "range_index": 0, "label": "J"},
            {"key": "Je", "range_index": 1, "label": "Je"},
            {"key": "R1", "range_index": None, "label": "R1"},
            {"key": "R4", "range_index": 2, "label": "R4"},
            {"key": "R5", "range_index": None, "label": "R5"},
            {"key": "as", "range_index": 3, "label": "as"},
            {"key": "al", "range_index": 4, "label": "al"},
            {"key": "ti", "range_index": 5, "label": "ti"},
            {"key": "to", "range_index": 6, "label": "to"},
        ],
        "models": {
            "t3": {"csv": "t3_NN_notpv.csv", "n_input": 7, "hidden_size": 100, "input_keys": ["J", "Je", "R4", "as", "al", "ti", "to"]},
            "t4": {"csv": "t4_NN_notpv.csv", "n_input": 7, "hidden_size": 100, "input_keys": ["J", "Je", "R4", "as", "al", "ti", "to"]},
        },
    },
}


class ResponseSurfaceModel:
    def __init__(self, csv_path: Path, n_input: int, hidden_size: int, input_keys: list[str]) -> None:
        self.n_input = n_input
        self.hidden_size = hidden_size
        self.input_keys = input_keys

        if not csv_path.exists():
            raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")

        with csv_path.open(newline="", encoding="utf-8-sig") as csvfile:
            filereader = csv.reader(csvfile)
            next(filereader)
            next(filereader)

            self.x_range = [[0.0, 0.0] for _ in range(self.n_input)]
            for i in range(self.n_input):
                self.x_range[i] = [float(value) for value in next(filereader)]

            next(filereader)
            self.y_range = [float(next(filereader)[0]) for _ in range(2)]

            next(filereader)
            self.out_range = [float(next(filereader)[0]) for _ in range(2)]

            next(filereader)
            self.w1 = [[float(value) for value in next(filereader)] for _ in range(self.hidden_size)]

            next(filereader)
            self.b1 = [float(next(filereader)[0]) for _ in range(self.hidden_size)]

            next(filereader)
            self.w2 = [[float(value) for value in next(filereader)] for _ in range(1)]

            next(filereader)
            self.b2 = [float(next(filereader)[0]) for _ in range(1)]

    def evaluate(self, values: dict[str, float]) -> float:
        xx = [values[key] for key in self.input_keys]

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


def current_panel_key() -> str:
    return st.session_state.get("selected_panel", "pv")


def current_panel_config() -> dict:
    return PANEL_CONFIG[current_panel_key()]


def active_variables(panel: dict) -> list[dict]:
    return [item for item in panel["variables"] if item["range_index"] is not None]


def active_keys(panel: dict) -> list[str]:
    return [item["key"] for item in active_variables(panel)]


def variable_label(panel: dict, key: str) -> str:
    for item in panel["variables"]:
        if item["key"] == key:
            return item["label"]
    return key


def load_model(base_dir: Path, panel: dict, model_key: str) -> ResponseSurfaceModel:
    model_info = panel["models"][model_key]
    csv_path = base_dir / panel["folder"] / model_info["csv"]
    return ResponseSurfaceModel(
        csv_path=csv_path,
        n_input=model_info["n_input"],
        hidden_size=model_info["hidden_size"],
        input_keys=model_info["input_keys"],
    )


def initialize_panel_state(panel: dict, model: ResponseSurfaceModel) -> None:
    pending_values = st.session_state.pop("pending_clicked_values", None)
    if isinstance(pending_values, dict):
        for key, value in pending_values.items():
            st.session_state[f"input_{key}"] = float(value)

    for item in active_variables(panel):
        state_key = f"input_{item['key']}"
        if state_key not in st.session_state:
            st.session_state[state_key] = midpoint(model.x_range[item["range_index"]])

    for key, value in panel["fixed_defaults"].items():
        state_key = f"input_{key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = float(value)

    if st.session_state.get("selected_model") not in panel["models"]:
        st.session_state["selected_model"] = panel["default_model"]

    valid_axis_keys = active_keys(panel)
    if st.session_state.get("x_axis") not in valid_axis_keys:
        st.session_state["x_axis"] = panel["default_x_axis"]

    if st.session_state.get("y_axis") not in valid_axis_keys or st.session_state["y_axis"] == st.session_state["x_axis"]:
        fallback_y = panel["default_y_axis"]
        if fallback_y == st.session_state["x_axis"]:
            for key in valid_axis_keys:
                if key != st.session_state["x_axis"]:
                    fallback_y = key
                    break
        st.session_state["y_axis"] = fallback_y

    if "last_clicked_point" not in st.session_state:
        st.session_state["last_clicked_point"] = ""


def current_values(panel: dict) -> dict[str, float]:
    values = {}
    for key in panel["fixed_defaults"]:
        values[key] = float(st.session_state[f"input_{key}"])
    for item in active_variables(panel):
        values[item["key"]] = float(st.session_state[f"input_{item['key']}"])
    return values


def apply_preset(panel: dict, model: ResponseSurfaceModel, preset: str) -> None:
    for item in active_variables(panel):
        value_range = model.x_range[item["range_index"]]
        if preset == "minimum":
            value = float(value_range[0])
        elif preset == "maximum":
            value = float(value_range[1])
        else:
            value = midpoint(value_range)
        st.session_state[f"input_{item['key']}"] = value


def compute_sat(values: dict[str, float]) -> float:
    return values["to"] + (1.0 / 23.0) * (values["as"] * values["J"] - values["al"] * values["Je"])


def compute_r3(panel_key: str, values: dict[str, float]) -> float:
    if panel_key == "notpv":
        return values["R1"]

    try:
        rad_term = (
            1.0 / (max(values["ep1"], 1e-10) ** -1 + max(values["ep2"], 1e-10) ** -1 - 1.0)
        ) * 4.0 * (293.0**3) * 5.67e-8
        return 1.0 / (25.0 + rad_term)
    except ZeroDivisionError:
        return 0.0


def build_surface(
    panel: dict,
    model: ResponseSurfaceModel,
    x_key: str,
    y_key: str,
    values: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_config = next(item for item in active_variables(panel) if item["key"] == x_key)
    y_config = next(item for item in active_variables(panel) if item["key"] == y_key)

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

    click_signature = (
        f"{point_index}|{current_panel_key()}|"
        f"{st.session_state['selected_model']}|{x_key}|{y_key}"
    )
    if st.session_state["last_clicked_point"] == click_signature:
        return

    st.session_state["pending_clicked_values"] = {x_key: float(x_value), y_key: float(y_value)}
    st.session_state["last_clicked_point"] = click_signature
    st.rerun()


def build_summary_rows(
    panel: dict,
    model: ResponseSurfaceModel,
    values: dict[str, float],
    x_key: str,
    y_key: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in panel["variables"]:
        value = values[item["key"]]
        if item["range_index"] is not None:
            value_range = model.x_range[item["range_index"]]
            if item["key"] == x_key:
                role = "X軸"
            elif item["key"] == y_key:
                role = "Y軸"
            else:
                role = "固定"
            rows.append(
                {
                    "変数": item["label"],
                    "現在値": f"{value:.3f}",
                    "最小": f"{value_range[0]:.3f}",
                    "最大": f"{value_range[1]:.3f}",
                    "役割": role,
                }
            )
        else:
            rows.append(
                {
                    "変数": item["label"],
                    "現在値": f"{value:.3f}",
                    "最小": "-",
                    "最大": "-",
                    "役割": "固定",
                }
            )
    return rows


def render_sidebar(panel: dict, model: ResponseSurfaceModel) -> None:
    st.sidebar.selectbox(
        "種類",
        list(PANEL_CONFIG.keys()),
        key="selected_panel",
        format_func=lambda key: PANEL_CONFIG[key]["label"],
    )

    panel = current_panel_config()
    if st.session_state.get("selected_model") not in panel["models"]:
        st.session_state["selected_model"] = panel["default_model"]

    st.sidebar.selectbox("表示する温度", list(panel["models"].keys()), key="selected_model")

    st.sidebar.header("入力条件")
    col1, col2, col3 = st.sidebar.columns(3)
    if col1.button("最小"):
        apply_preset(panel, model, "minimum")
    if col2.button("中央"):
        apply_preset(panel, model, "midpoint")
    if col3.button("最大"):
        apply_preset(panel, model, "maximum")

    st.sidebar.markdown("---")
    st.sidebar.subheader("主な変数")
    for item in active_variables(panel):
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
    st.sidebar.subheader("固定値")
    for key, value in panel["fixed_defaults"].items():
        st.sidebar.caption(f"{key} = {value:.3f}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("表示軸")
    valid_keys = active_keys(panel)
    st.sidebar.selectbox("X軸", valid_keys, key="x_axis", format_func=lambda key: variable_label(panel, key))
    y_candidates = [key for key in valid_keys if key != st.session_state["x_axis"]]
    if st.session_state["y_axis"] == st.session_state["x_axis"]:
        st.session_state["y_axis"] = y_candidates[0]
    st.sidebar.selectbox("Y軸", y_candidates, key="y_axis", format_func=lambda key: variable_label(panel, key))


def render_intro(panel: dict) -> None:
    if current_panel_key() == "pv":
        detail_line = "t1 から t4 までを1つの画面で切り替えて表示できます。"
    else:
        detail_line = "t3 と t4 を1つの画面で切り替えて表示できます。"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top right, rgba(28, 122, 98, 0.12), transparent 28%),
                linear-gradient(180deg, #f7faf8 0%, #eff5f2 100%);
        }}
        .hero-card {{
            background: linear-gradient(135deg, #0f4c45 0%, #1f7a6c 100%);
            color: #f6f8f6;
            padding: 1.1rem 1.3rem;
            border-radius: 16px;
            margin-bottom: 1rem;
        }}
        .hero-card h1 {{ margin: 0 0 0.2rem 0; font-size: 1.85rem; }}
        .hero-card p {{ margin: 0.15rem 0; }}
        </style>
        <div class="hero-card">
            <h1>応答曲面ダッシュボード</h1>
            <p>{panel["label"]} を内部で切り替えて表示できます。</p>
            <p>{detail_line}</p>
            <p>右の等高線グラフをクリックすると、その場所に入力値が更新されます。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(panel_key: str, selected_model: str, values: dict[str, float], prediction: float, sat_value: float, r3_value: float) -> None:
    if panel_key == "pv":
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(f"予測温度 {selected_model}", f"{prediction:.3f}")
        col2.metric("SAT", f"{sat_value:.3f}")
        col3.metric("R3", f"{r3_value:.3f}")
        col4.metric("R1", f"{values['R1']:.3f}")
        col5.metric("R5", f"{values['R5']:.3f}")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"予測温度 {selected_model}", f"{prediction:.3f}")
        col2.metric("SAT", f"{sat_value:.3f}")
        col3.metric("R1", f"{values['R1']:.3f}")
        col4.metric("R5", f"{values['R5']:.3f}")


def main() -> None:
    st.set_page_config(page_title="応答曲面ダッシュボード", layout="wide")

    if "selected_panel" not in st.session_state:
        st.session_state["selected_panel"] = "pv"

    base_dir = MODEL_BASE_DIR
    panel = current_panel_config()

    if st.session_state.get("selected_model") not in panel["models"]:
        st.session_state["selected_model"] = panel["default_model"]

    initial_model = load_model(base_dir, panel, st.session_state["selected_model"])
    initialize_panel_state(panel, initial_model)
    render_sidebar(panel, initial_model)

    panel = current_panel_config()
    selected_model = st.session_state["selected_model"]
    model = load_model(base_dir, panel, selected_model)
    initialize_panel_state(panel, model)

    render_intro(panel)

    values = current_values(panel)
    prediction = model.evaluate(values)
    sat_value = compute_sat(values)
    r3_value = compute_r3(current_panel_key(), values)
    x_key = st.session_state["x_axis"]
    y_key = st.session_state["y_axis"]

    render_metrics(current_panel_key(), selected_model, values, prediction, sat_value, r3_value)
    st.caption("左は3D表示、右はクリックで位置を変えるための等高線表示です。")

    chart_tab, table_tab = st.tabs(["グラフ", "入力一覧"])

    with chart_tab:
        x_mesh, y_mesh, z_values = build_surface(panel, model, x_key, y_key, values)
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
                        name="現在位置",
                    ),
                ]
            )
            surface_fig.update_layout(
                scene={
                    "xaxis_title": variable_label(panel, x_key),
                    "yaxis_title": variable_label(panel, y_key),
                    "zaxis_title": selected_model,
                },
                margin={"l": 0, "r": 0, "b": 0, "t": 10},
                height=620,
            )
            st.plotly_chart(surface_fig, width="stretch")

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
                        text=["現在位置"],
                        textposition="top center",
                        name="現在位置",
                    ),
                ]
            )
            contour_fig.update_layout(
                xaxis_title=variable_label(panel, x_key),
                yaxis_title=variable_label(panel, y_key),
                height=620,
                margin={"l": 10, "r": 10, "b": 10, "t": 10},
                clickmode="event+select",
            )
            selection = st.plotly_chart(
                contour_fig,
                width="stretch",
                on_select="rerun",
                selection_mode="points",
                key="contour_chart",
            )
            st.caption("右の等高線グラフをクリックすると、その場所の値に更新されます。")
            apply_chart_selection(selection, x_key, y_key)

    with table_tab:
        st.dataframe(build_summary_rows(panel, model, values, x_key, y_key), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
