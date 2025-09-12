from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import make_metric, thresholds_text
from .common import (
    chat_mask,
    sessions_from_timestamps,
    is_inbound,
    is_outbound,
    fmt_from_template,
    latex_fill_from_template,
    update_extras_with_value,
)

# ----------------------------- metric schema ---------------------------------

@dataclass(frozen=True)
class MetricDef:
    id: str
    label: str
    dist_col: str
    ok: Optional[float]
    higher_is_worse: bool
    fmt_tpl: str
    latex_formula: Optional[str] = None
    latex_numbers_tpl: Optional[str] = None
    explanation_md: Optional[str] = None
    missing_md: Optional[str] = None


C5_DEFS: List[MetricDef] = [
    MetricDef(
        id="C5/F1",
        label="Wi‑Fi/DHCP re‑association events per hour",
        dist_col="C5_F1_DhcpPerHour",
        ok=1.0, higher_is_worse=True, fmt_tpl="{v:.2f} / h",
        latex_formula=r"r=\frac{N_{\mathrm{DHCP}}}{H}\quad\text{with}\ H=\text{hours observed in day}",
        latex_numbers_tpl=r"N_{\mathrm{DHCP}}={events_total},\ H={hours_considered:.1f}\ \Rightarrow\ r={value:.2f}\ /\mathrm{h}",
        explanation_md=(
            "This metric approximates how often the device **re‑associates** to the Wi‑Fi network by "
            "counting DHCP request/response packets and normalizing by the **hours of captured traffic** in the day. "
            "Frequent re‑associations can reflect restless device handling or moving in/out of coverage—behaviors "
            "consistent with psychomotor agitation. Interpret relative to your baseline; router/AP issues can also raise it."
        ),
        missing_md="Needs UDP ports 67/68 or 546/547 (DHCP) in the trace; if not present, the value is N/A.",
    ),
    MetricDef(
        id="C5/F2",
        label="Mean Wi‑Fi dwell‑time (min between DHCP events)",
        dist_col="C5_F2_WifiDwellMin",
        ok=30.0, higher_is_worse=False, fmt_tpl="{v:.1f} min",
        latex_formula=r"\overline{T}_{\mathrm{dwell}}=\frac{1}{K-1}\sum_{i=1}^{K-1}(t_{i+1}-t_i)\quad\text{for consecutive DHCP events}",
        latex_numbers_tpl=r"K={n_events}\ \Rightarrow\ \overline{T}_{\mathrm{dwell}}={value:.0f}\ \mathrm{min}",
        explanation_md=(
            "Average **dwell time** between DHCP events. Shorter dwell suggests frequent disconnections/"
            "toggling and can indicate restlessness. Persistent reductions are more informative than one‑off drops."
        ),
        missing_md="Needs at least two DHCP events in the day; otherwise N/A.",
    ),
    MetricDef(
        id="C5/F3",
        label="Median inter‑keystroke gap in chat (s)",
        dist_col="C5_F3_MedianIKS",
        ok=1.0, higher_is_worse=True, fmt_tpl="{v:.2f} s",
        latex_formula=r"\mathrm{IKS}_{\mathrm{med}}=\mathrm{median}\{\Delta t_i: \Delta t_i\le 3\ \mathrm{s},\ \text{consecutive outbound packets in chat flows}\}",
        latex_numbers_tpl=r"n_{\mathrm{gaps}}={n_gaps}\ \Rightarrow\ \mathrm{IKS}_{\mathrm{med}}={value:.2f}\ \mathrm{s}",
        explanation_md=(
            "Typing speed proxy based on **time between consecutive outbound packets** during messaging sessions. "
            "We cap at ≤3 s to focus on active typing. Larger medians → slower typing (psychomotor retardation)."
        ),
        missing_md="Requires enough chat activity with outbound packets; if none, the metric is N/A.",
    ),
    MetricDef(
        id="C5/F4",
        label="Typing‑speed variability (SD of IKS, s)",
        dist_col="C5_F4_IKSStd",
        ok=0.4, higher_is_worse=True, fmt_tpl="{v:.2f} s",
        latex_formula=r"\sigma_{\mathrm{IKS}}=\sqrt{\frac{1}{n-1}\sum_i(\Delta t_i-\overline{\Delta t})^2}",
        latex_numbers_tpl=r"n_{\mathrm{gaps}}={n_gaps},\ \sigma_{\mathrm{IKS}}={value:.2f}\ \mathrm{s}",
        explanation_md=(
            "Standard deviation of inter‑keystroke gaps within messaging. Higher variability suggests erratic rhythm "
            "consistent with slowing or fluctuating effort."
        ),
        missing_md="Needs ≥3 qualifying gaps; with fewer gaps, SD is unreliable and may be N/A.",
    ),
    MetricDef(
        id="C5/F5",
        label="Sub‑30 s sessions per day",
        dist_col="C5_F5_Sub30sSessions",
        ok=12.0, higher_is_worse=True, fmt_tpl="{v:.0f}",
        latex_formula=r"N_{<30\mathrm{s}}=\#\{\text{sessions with }L<30\ \mathrm{s}\}\quad(\text{5‑min idle gap segmentation})",
        latex_numbers_tpl=r"N_{<30\mathrm{s}}={n_sessions_short}\ /\ N_{\mathrm{all}}={n_sessions_total}",
        explanation_md=(
            "Number of **very short sessions** (<30 s) after segmenting traffic with a 5‑min idle gap. "
            "A high number of quick 'peeks' is characteristic of restless screen‑checking."
        ),
        missing_md="Requires detectable sessions; with extremely sparse data, the count may be biased low.",
    ),
    MetricDef(
        id="C5/F6",
        label="Session‑duration Fano factor",
        dist_col="C5_F6_SessionFano",
        ok=1.5, higher_is_worse=True, fmt_tpl="{v:.2f}",
        latex_formula=r"F=\frac{\operatorname{Var}(L)}{\operatorname{E}[L]}\quad\text{for session lengths }L\ (\text{seconds})",
        latex_numbers_tpl=r"\bar L={mean_len_sec:.1f}\ \mathrm{s},\ \mathrm{Var}={var_len_sec:.1f},\ F={value:.2f}\ (n={n_sessions})",
        explanation_md=(
            "The Fano factor is a scale‑free measure of burstiness: it divides the variance of session lengths by their mean. "
            "Higher values indicate a mix of very short and very long sessions, a pattern often seen with agitation or binge‑scrolling "
            "interspersed with rapid checks. Because the ratio normalizes by the mean, it compares fairly across people who naturally "
            "use their devices for different average durations. Days with only a handful of sessions make the estimate unstable; the "
            "metric is N/A if there is not enough data. Network segmentation issues can inflate variance; check whether many sessions "
            "are just over the five‑minute gap threshold. Use together with the short‑session count to confirm a bursty interaction style."
        ),
        missing_md="Needs at least 3 sessions to estimate a variance; otherwise N/A.",
    ),
]

# ----------------------------- helpers ----------------------------------------

_DHCP_PORTS_V4 = {67, 68}
_DHCP_PORTS_V6 = {546, 547}

def _is_dhcp_row(row) -> bool:
    try:
        if str(row.get("Protocol", "")).upper() != "UDP":
            return False
        ports = set()
        sp = row.get("Source Port", None); dp = row.get("Destination Port", None)
        if pd.notna(sp): ports.add(int(sp))
        if pd.notna(dp): ports.add(int(dp))
        return len(ports & (_DHCP_PORTS_V4 | _DHCP_PORTS_V6)) > 0
    except Exception:
        return False

def _dhcp_reassoc_per_hour(df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    if df is None or df.empty or "Timestamp" not in df.columns:
        return (np.nan, {})
    d = df.loc[df.apply(_is_dhcp_row, axis=1)].sort_values("Timestamp")
    events_total = int(d.shape[0])
    if df["Timestamp"].notna().any():
        span_h = (df["Timestamp"].max() - df["Timestamp"].min()).total_seconds() / 3600.0
        hours_considered = float(min(24.0, max(1.0, span_h)))
    else:
        hours_considered = 24.0
    val = float(events_total / hours_considered) if hours_considered > 0 else np.nan
    return (val, {"events_total": events_total, "hours_considered": hours_considered})

def _mean_wifi_dwell_min(df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    d = df.loc[df.apply(_is_dhcp_row, axis=1)].sort_values("Timestamp")
    n = int(d.shape[0])
    if n < 2:
        return (np.nan, {})
    gaps_min = (d["Timestamp"].diff().dt.total_seconds().dropna() / 60.0).to_numpy()
    return (float(np.mean(gaps_min)) if gaps_min.size else np.nan, {"n_events": n})

def _chat_outbound_ts(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "Timestamp" not in df.columns:
        return pd.Series([], dtype="datetime64[ns]")
    chat = df.loc[chat_mask(df)]
    if chat.empty or (not {"Source IP", "Destination IP"}.issubset(chat.columns)):
        return pd.Series([], dtype="datetime64[ns]")
    chat = chat.loc[chat.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1)]
    return chat["Timestamp"].sort_values()

def _median_iks_sec(df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    ts = _chat_outbound_ts(df)
    if ts.empty:
        return (np.nan, {})
    gaps = ts.diff().dt.total_seconds().dropna()
    gaps = gaps[(gaps > 0) & (gaps <= 3.0)]
    if gaps.empty:
        return (np.nan, {})
    return (float(gaps.median()), {"n_gaps": int(gaps.shape[0])})

def _iks_std_sec(df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    ts = _chat_outbound_ts(df)
    if ts.empty:
        return (np.nan, {})
    gaps = ts.diff().dt.total_seconds().dropna()
    gaps = gaps[(gaps > 0) & (gaps <= 3.0)]
    if gaps.shape[0] < 3:
        return (np.nan, {})
    return (float(gaps.std(ddof=1)), {"n_gaps": int(gaps.shape[0])})

def _sub30s_session_count(df: pd.DataFrame, gap_sec: int = 300) -> Tuple[float, Dict[str, Any]]:
    if df is None or df.empty or "Timestamp" not in df.columns:
        return (0.0, {"n_sessions_short": 0, "n_sessions_total": 0})
    L = sessions_from_timestamps(df.sort_values("Timestamp"), gap_sec=gap_sec)
    if not L:
        return (0.0, {"n_sessions_short": 0, "n_sessions_total": 0})
    durations = np.array([(b - a).total_seconds() for a, b in L], dtype=float)
    n_short = int((durations < 30.0).sum())
    return (float(n_short), {"n_sessions_short": n_short, "n_sessions_total": int(durations.size)})

def _session_fano(df: pd.DataFrame, gap_sec: int = 300) -> Tuple[float, Dict[str, Any]]:
    if df is None or df.empty or "Timestamp" not in df.columns:
        return (np.nan, {})
    L = sessions_from_timestamps(df.sort_values("Timestamp"), gap_sec=gap_sec)
    n_sessions = len(L)
    if n_sessions < 3:
        return (np.nan, {"n_sessions": n_sessions})
    durations = np.array([(b - a).total_seconds() for a, b in L], dtype=float)
    mean_len = float(durations.mean()) if durations.size else np.nan
    var_len = float(durations.var(ddof=1)) if durations.size > 1 else np.nan
    fano = var_len / mean_len if mean_len > 0 else np.nan
    return (fano, {"mean_len_sec": mean_len, "var_len_sec": var_len, "n_sessions": n_sessions})

# ------------------------------ main class ------------------------------------

class Criterion5:
    """Psychomotor agitation / retardation — C5 (self-contained)."""

    def _compute_today_fields(self, df_day: pd.DataFrame) -> Dict[str, Any]:
        rec: Dict[str, Any] = {}
        rec["C5_F1_DhcpPerHour"], rec["_EX_F1"] = _dhcp_reassoc_per_hour(df_day)
        rec["C5_F2_WifiDwellMin"], rec["_EX_F2"] = _mean_wifi_dwell_min(df_day)
        rec["C5_F3_MedianIKS"], rec["_EX_F3"] = _median_iks_sec(df_day)
        rec["C5_F4_IKSStd"],   rec["_EX_F4"] = _iks_std_sec(df_day)
        rec["C5_F5_Sub30sSessions"], rec["_EX_F5"] = _sub30s_session_count(df_day, gap_sec=300)
        rec["C5_F6_SessionFano"], rec["_EX_F6"] = _session_fano(df_day, gap_sec=300)
        return rec

    def _latex_ctx(self, mdef: MetricDef, value: Any, today_fields: Dict[str, Any]) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {}
        ctx = update_extras_with_value(ctx, value)
        if mdef.id == "C5/F1": ctx.update(today_fields.get("_EX_F1", {}))
        if mdef.id == "C5/F2": ctx.update(today_fields.get("_EX_F2", {}))
        if mdef.id == "C5/F3": ctx.update(today_fields.get("_EX_F3", {}))
        if mdef.id == "C5/F4": ctx.update(today_fields.get("_EX_F4", {}))
        if mdef.id == "C5/F5": ctx.update(today_fields.get("_EX_F5", {}))
        if mdef.id == "C5/F6": ctx.update(today_fields.get("_EX_F6", {}))
        return ctx

    def compute(
        self,
        df_day: pd.DataFrame,
        today: dict,         # not used (kept for parity)
        aux_ctx: dict,       # not used here
        ALL_DAILY: pd.DataFrame
    ) -> List[Dict[str, Any]]:

        today = dict(today or {})
        need_calc = any(
            (m.dist_col not in today) or pd.isna(today.get(m.dist_col)) for m in C5_DEFS
        )
        today_fields = (
            self._compute_today_fields(df_day) if need_calc and df_day is not None and not df_day.empty else {}
        )
        for k, v in today_fields.items():
            today.setdefault(k, v)

        out: List[Dict[str, Any]] = []

        for mdef in C5_DEFS:
            val = today.get(mdef.dist_col, np.nan)

            # Status & ranges
            if (val is None) or (isinstance(val, float) and (np.isnan(val) or not np.isfinite(val))) or (mdef.ok is None):
                status_tuple = ("N/A", "blue")
                ranges_str   = "" if mdef.ok is None else thresholds_text(mdef.ok, higher_is_worse=mdef.higher_is_worse)
            else:
                v = float(val)
                is_ok = (v <= mdef.ok) if mdef.higher_is_worse else (v >= mdef.ok)
                status_tuple = ("OK", "green") if is_ok else ("Caution", "orange")
                ranges_str   = thresholds_text(mdef.ok, higher_is_worse=mdef.higher_is_worse)

            # Formatter & LaTeX
            fmt = fmt_from_template(mdef.fmt_tpl)
            latex_numbers = None
            if isinstance(mdef.latex_numbers_tpl, str) and mdef.latex_numbers_tpl.strip():
                lctx = self._latex_ctx(mdef, val, today_fields)
                latex_numbers = latex_fill_from_template(mdef.latex_numbers_tpl, lctx)

            out.append(
                make_metric(
                    label=mdef.label,
                    value=val,
                    fmt=fmt,
                    status_tuple=status_tuple,
                    ranges_str=ranges_str,
                    latex_formula=mdef.latex_formula,
                    latex_numbers=latex_numbers,
                    heuristic_md=mdef.explanation_md,
                    missing_md=mdef.missing_md,
                    dist_col=mdef.dist_col,
                    range_cfg={"ok": mdef.ok, "higher_is_worse": mdef.higher_is_worse} if mdef.ok is not None else None,
                )
            )

        return out
