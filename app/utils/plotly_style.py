from __future__ import annotations

from collections.abc import Iterable
from numbers import Number
from typing import Any

import plotly.graph_objects as go
import streamlit as st

GREYSCALE_TOGGLE_KEY = "plotly_greyscale_mode"

GREYSCALE_COLORWAY = [
    "#111111",
    "#3D3D3D",
    "#6B6B6B",
    "#8C8C8C",
    "#B0B0B0",
    "#D0D0D0",
]

GREYSCALE_COLORSCALE = [
    [0.0, "#111111"],
    [0.2, "#3D3D3D"],
    [0.4, "#6B6B6B"],
    [0.6, "#8C8C8C"],
    [0.8, "#B0B0B0"],
    [1.0, "#E6E6E6"],
]

PATTERN_SHAPES = ["/", "\\", "x", "-", "|", ".", "+"]
LINE_DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
MARKER_SYMBOLS = [
    "circle",
    "square",
    "diamond",
    "cross",
    "x",
    "triangle-up",
    "triangle-down",
    "star",
]


def render_plot_style_toggle_in_sidebar() -> None:
    """Render the greyscale + pattern toggle in the sidebar."""
    with st.sidebar:
        st.toggle(
            "Paper-ready greyscale",
            value=bool(st.session_state.get(GREYSCALE_TOGGLE_KEY, False)),
            key=GREYSCALE_TOGGLE_KEY,
            help="Switch all Plotly charts to greyscale with patterns and dashes.",
        )


def is_greyscale_mode() -> bool:
    return bool(st.session_state.get(GREYSCALE_TOGGLE_KEY, False))


def plotly_chart(fig: Any, container: Any | None = None, **kwargs):
    if fig is None:
        if container is None:
            return st.plotly_chart(fig, **kwargs)
        return container.plotly_chart(fig, **kwargs)
    styled = apply_greyscale_style(fig)
    if container is None:
        return st.plotly_chart(styled, **kwargs)
    return container.plotly_chart(styled, **kwargs)


def apply_greyscale_style(fig: Any):
    if not is_greyscale_mode():
        return fig
    styled = go.Figure(fig)
    styled.update_layout(
        colorway=GREYSCALE_COLORWAY,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#111111"),
        legend=dict(font=dict(color="#111111")),
    )
    styled.update_xaxes(
        showgrid=True,
        gridcolor="#D0D0D0",
        zerolinecolor="#B0B0B0",
        linecolor="#333333",
        tickcolor="#333333",
    )
    styled.update_yaxes(
        showgrid=True,
        gridcolor="#D0D0D0",
        zerolinecolor="#B0B0B0",
        linecolor="#333333",
        tickcolor="#333333",
    )
    _apply_coloraxis(styled)
    for idx, trace in enumerate(styled.data):
        _style_trace(trace, idx)
    _style_shapes(styled)
    _style_annotations(styled)
    return styled


def _apply_coloraxis(fig: go.Figure) -> None:
    for axis_name in fig.layout:
        if not axis_name.startswith("coloraxis"):
            continue
        axis = fig.layout[axis_name]
        if axis is None:
            continue
        axis.colorscale = GREYSCALE_COLORSCALE


def _style_trace(trace: Any, idx: int) -> None:
    trace_type = getattr(trace, "type", "") or ""
    color = _color_for_index(idx)
    dash = LINE_DASHES[idx % len(LINE_DASHES)]
    symbol = MARKER_SYMBOLS[idx % len(MARKER_SYMBOLS)]
    pattern = PATTERN_SHAPES[idx % len(PATTERN_SHAPES)]

    if trace_type in ("bar", "histogram"):
        _style_bar_like(trace, color, pattern)
        return
    if trace_type in ("pie",):
        _style_pie(trace, idx)
        return
    if trace_type in ("scatter", "scattergl"):
        _style_scatter(trace, color, dash, symbol)
        return
    if trace_type in ("box", "violin"):
        _style_box_violin(trace, color)
        return
    if trace_type in ("heatmap", "histogram2d", "histogram2dcontour", "contour"):
        _style_heatmap(trace)
        return
    if trace_type in ("indicator",):
        _style_indicator(trace)
        return
    _style_generic(trace, color, dash, symbol, pattern)


def _style_bar_like(trace: Any, color: str, pattern: str) -> None:
    marker = getattr(trace, "marker", None)
    if marker is None:
        return
    if _marker_is_continuous(marker):
        trace.update(marker=dict(colorscale=GREYSCALE_COLORSCALE))
        return
    color_values = getattr(marker, "color", None)
    if _is_iterable(color_values):
        colors = _repeat_palette(len(color_values))
        update = {"color": colors, "line": {"color": "#111111", "width": 1}}
        if _marker_supports_pattern(marker):
            update["pattern"] = dict(
                shape=_repeat_patterns(len(color_values)),
                fgcolor="#111111",
                bgcolor=colors,
                solidity=0.35,
            )
        trace.update(marker=update)
        return
    update = {"color": color, "line": {"color": "#111111", "width": 1}}
    if _marker_supports_pattern(marker):
        update["pattern"] = dict(
            shape=pattern,
            fgcolor="#111111",
            bgcolor=color,
            solidity=0.35,
        )
    trace.update(marker=update)


def _style_pie(trace: Any, idx: int) -> None:
    marker = getattr(trace, "marker", None)
    if marker is None:
        return
    count = _segment_count(trace, marker)
    if count:
        colors = _repeat_palette(count)
        update = {"colors": colors}
        if _marker_supports_pattern(marker):
            update["pattern"] = dict(
                shape=_repeat_patterns(count),
                fgcolor="#111111",
                bgcolor=colors,
                solidity=0.35,
            )
        trace.update(marker=update)
        return
    color = _color_for_index(idx)
    update = {"colors": [color]}
    if _marker_supports_pattern(marker):
        update["pattern"] = dict(shape=[PATTERN_SHAPES[idx % len(PATTERN_SHAPES)]])
    trace.update(marker=update)


def _style_scatter(trace: Any, color: str, dash: str, symbol: str) -> None:
    marker = getattr(trace, "marker", None)
    line = getattr(trace, "line", None)
    mode = getattr(trace, "mode", None)
    mode_str = str(mode) if mode is not None else ""
    no_markers = _trace_disables_markers(trace)
    if marker is not None and _marker_is_continuous(marker):
        trace.update(marker=dict(colorscale=GREYSCALE_COLORSCALE))
    else:
        if line is not None:
            line_update = {"color": color}
            if hasattr(line, "dash"):
                line_update["dash"] = dash
            if getattr(line, "width", None) is None:
                line_update["width"] = 2
            trace.update(line=line_update)
        if (not no_markers) and marker is not None:
            marker_update = {"color": color}
            if getattr(marker, "symbol", None) is None:
                marker_update["symbol"] = symbol
            if getattr(marker, "size", None) is None and (line is not None or "lines" in mode_str):
                marker_update["size"] = 7
            if "maxdisplayed" in marker and getattr(marker, "maxdisplayed", None) is None:
                marker_update["maxdisplayed"] = 200
            if "line" in marker and getattr(marker, "line", None) is None:
                marker_update["line"] = {"color": "#111111", "width": 1}
            trace.update(marker=marker_update)
        if not no_markers:
            if line is not None or "lines" in mode_str:
                if "markers" not in mode_str:
                    next_mode = "lines+markers" if not mode_str else f"{mode_str}+markers"
                    trace.update(mode=next_mode)
        else:
            if "markers" in mode_str:
                trace.update(mode="lines")
    if getattr(trace, "fill", None):
        trace.update(fillcolor=_rgba_from_hex(color, 0.2))


def _style_box_violin(trace: Any, color: str) -> None:
    line = getattr(trace, "line", None)
    if line is not None:
        line_update = {"color": "#111111"}
        if getattr(line, "width", None) is None:
            line_update["width"] = 1
        trace.update(line=line_update)
    trace.update(fillcolor=_rgba_from_hex(color, 0.25))
    marker = getattr(trace, "marker", None)
    if marker is not None:
        trace.update(marker=dict(color=color))


def _style_heatmap(trace: Any) -> None:
    trace.update(colorscale=GREYSCALE_COLORSCALE)


def _style_indicator(trace: Any) -> None:
    gauge = getattr(trace, "gauge", None)
    if gauge is not None:
        steps = []
        if getattr(gauge, "steps", None):
            for idx, step in enumerate(gauge.steps):
                if isinstance(step, dict):
                    updated = dict(step)
                else:
                    try:
                        updated = dict(step)
                    except Exception:
                        updated = {}
                updated["color"] = _step_color(idx)
                steps.append(updated)
        gauge_update = {"bar": {"color": "#333333"}}
        if steps:
            gauge_update["steps"] = steps
        threshold = getattr(gauge, "threshold", None)
        if threshold is not None and getattr(threshold, "line", None) is not None:
            gauge_update["threshold"] = {"line": {"color": "#111111", "width": 2}}
        trace.update(gauge=gauge_update)
    number = getattr(trace, "number", None)
    if number is not None:
        trace.update(number={"font": {"color": "#111111"}})
    title = getattr(trace, "title", None)
    if title is not None:
        trace.update(title={"font": {"color": "#111111"}})


def _style_generic(trace: Any, color: str, dash: str, symbol: str, pattern: str) -> None:
    line = getattr(trace, "line", None)
    if line is not None:
        line_update = {"color": color}
        if hasattr(line, "dash"):
            line_update["dash"] = dash
        if getattr(line, "width", None) is None:
            line_update["width"] = 2
        trace.update(line=line_update)
    marker = getattr(trace, "marker", None)
    if marker is not None:
        if _marker_is_continuous(marker):
            trace.update(marker=dict(colorscale=GREYSCALE_COLORSCALE))
            return
        update = {"color": color}
        if _marker_supports_pattern(marker):
            update["pattern"] = dict(
                shape=pattern,
                fgcolor="#111111",
                bgcolor=color,
                solidity=0.35,
            )
        if getattr(marker, "symbol", None) is None:
            update["symbol"] = symbol
        trace.update(marker=update)


def _style_shapes(fig: go.Figure) -> None:
    shapes = getattr(fig.layout, "shapes", None)
    if not shapes:
        return
    updated = []
    for idx, shape in enumerate(shapes):
        try:
            payload = dict(shape)
        except Exception:
            try:
                payload = shape.to_plotly_json()
            except Exception:
                updated.append(shape)
                continue
        raw_line = payload.get("line", {})
        if isinstance(raw_line, dict):
            line = dict(raw_line)
        else:
            try:
                line = dict(raw_line)
            except Exception:
                line = {}
        line["color"] = _color_for_index(idx)
        payload["line"] = line
        if payload.get("fillcolor"):
            payload["fillcolor"] = _rgba_from_hex(_color_for_index(idx), 0.15)
        updated.append(payload)
    fig.update_layout(shapes=updated)


def _style_annotations(fig: go.Figure) -> None:
    annotations = getattr(fig.layout, "annotations", None)
    if not annotations:
        return
    updated = []
    for ann in annotations:
        try:
            payload = dict(ann)
        except Exception:
            try:
                payload = ann.to_plotly_json()
            except Exception:
                updated.append(ann)
                continue
        raw_font = payload.get("font", {})
        if isinstance(raw_font, dict):
            font = dict(raw_font)
        else:
            try:
                font = dict(raw_font)
            except Exception:
                font = {}
        font["color"] = "#111111"
        payload["font"] = font
        updated.append(payload)
    fig.update_layout(annotations=updated)


def _segment_count(trace: Any, marker: Any) -> int:
    for attr in ("values", "labels"):
        data = getattr(trace, attr, None)
        if _is_iterable(data):
            try:
                return len(data)
            except Exception:
                pass
    colors = getattr(marker, "colors", None)
    if _is_iterable(colors):
        try:
            return len(colors)
        except Exception:
            pass
    return 0


def _repeat_palette(count: int) -> list[str]:
    return [GREYSCALE_COLORWAY[i % len(GREYSCALE_COLORWAY)] for i in range(count)]


def _repeat_patterns(count: int) -> list[str]:
    return [PATTERN_SHAPES[i % len(PATTERN_SHAPES)] for i in range(count)]


def _color_for_index(idx: int) -> str:
    return GREYSCALE_COLORWAY[idx % len(GREYSCALE_COLORWAY)]


def _step_color(idx: int) -> str:
    step_colors = ["#E0E0E0", "#B5B5B5", "#8C8C8C"]
    return step_colors[idx % len(step_colors)]


def _marker_supports_pattern(marker: Any) -> bool:
    try:
        return "pattern" in marker
    except Exception:
        return False


def _marker_is_continuous(marker: Any) -> bool:
    coloraxis = getattr(marker, "coloraxis", None)
    if coloraxis:
        return True
    colorscale = getattr(marker, "colorscale", None)
    if colorscale:
        return True
    color_values = getattr(marker, "color", None)
    return _is_numeric_iterable(color_values)


def _trace_disables_markers(trace: Any) -> bool:
    meta = getattr(trace, "meta", None)
    if isinstance(meta, dict):
        return bool(meta.get("no_markers"))
    return False


def _is_numeric_iterable(value: Any) -> bool:
    if not _is_iterable(value):
        return False
    checked = 0
    for item in value:
        if item is None:
            continue
        checked += 1
        if checked > 6:
            break
        if not isinstance(item, Number):
            return False
    return checked > 0


def _is_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes))


def _rgba_from_hex(color: str, alpha: float) -> str:
    if not isinstance(color, str):
        return f"rgba(0,0,0,{alpha})"
    if not color.startswith("#"):
        return color
    hex_color = color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    if len(hex_color) != 6:
        return color
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except Exception:
        return color
    return f"rgba({r},{g},{b},{alpha})"
