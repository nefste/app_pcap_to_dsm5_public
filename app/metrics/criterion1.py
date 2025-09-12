from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from .base import make_metric, thresholds_text
from .common import (  # light, fast utilities
    chat_mask,
    sessions_from_timestamps,
    streaming_inbound_mask,
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

# All 14 C1 metrics (values computed below)
C1_DEFS: List[MetricDef] = [
    MetricDef(
        id="C1/F1",
        label="Late Night Share",
        dist_col="LateNightShare",
        ok=0.15, higher_is_worse=True, fmt_tpl="{v:.3f}",
        latex_formula=r"\mathrm{LNS}=\frac{N_{\text{night}}}{N_{\text{total}}}",
        latex_numbers_tpl=r"\mathrm{LNS}=\frac{{n_night}}{{n_total}}={value:.3f}",
        explanation_md=(
            "Proportion of packets observed during the late‑night window relative to the full day. Higher values suggest a delayed sleep phase or nocturnal activity."

        ),
    ),
    MetricDef(
        id="C1/F2",
        label="Longest Inactivity (h)",
        dist_col="LongestInactivityHours",
        ok=6.0, higher_is_worse=False, fmt_tpl="{v:.2f} h",
        latex_formula=r"\mathrm{LIH}=\max(\Delta t)\,/\,3600",
        explanation_md=(
            "Maximum gap between two successive packets within the day, converted to hours. Higher values indicate consolidated inactivity consistent with sleep."
        ),
    ),
    MetricDef(
        id="C1/F3",
        label="Active Night Minutes (00–05)",
        dist_col="ActiveNightMinutes",
        ok=30.0, higher_is_worse=True, fmt_tpl="{v:.0f} min",
        latex_formula=r"\mathrm{ANM}=\sum_{m\in[00{:}00,05{:}00)}\mathbf{1}[x_m>0]",
        explanation_md=(
            "Number of minutes with any network activity between 00:00 and 05:00. Frequent activity here may indicate sleep fragmentation or delayed bedtimes."
        ),
    ),
    MetricDef(
        id="C1/F4",
        label="Interdaily Stability (IS)",
        dist_col="IS",
        ok=0.5, higher_is_worse=False, fmt_tpl="{v:.2f}",
        latex_formula=r"\mathrm{IS}=\frac{N\sum_{h=0}^{23}(\bar{x}_h-\bar{x})^2}{24\sum_i(x_i-\bar{x})^2}",
        explanation_md=(
            "Regularity of the 24‑hour activity pattern across days. Higher values indicate a more stable circadian rhythm."
        ),
    ),
    MetricDef(
        id="C1/F5",
        label="Intradaily Variability (IV)",
        dist_col="IV",
        ok=0.8, higher_is_worse=True, fmt_tpl="{v:.2f}",
        latex_formula=r"\mathrm{IV}=\frac{N\sum_i(x_i-x_{i-1})^2}{(N-1)\sum_i(x_i-\bar{x})^2}",
        explanation_md=(
            "Fragmentation of the daily activity rhythm. Higher values reflect more start‑stop behavior."
        ),
    ),
    MetricDef(
        id="C1/F6",
        label="Night/Day Activity Ratio (mins)",
        dist_col="ND_Ratio",
        ok=0.2, higher_is_worse=True, fmt_tpl="{v:.2f}",
        latex_formula=r"\mathrm{ND}=\frac{\text{Active minutes at night}}{\text{Active minutes at day}}",
        explanation_md=(
            "Ratio of night‑time active minutes to day‑time active minutes. Elevated ratios indicate nocturnal shifting."
        ),
    ),
    MetricDef(
        id="C1/F7",
        label="Distinct social domains",
        dist_col="F1_DistinctSocial",
        ok=3.0, higher_is_worse=False, fmt_tpl="{v:.0f}",
        latex_formula=r"\mathrm{DistinctSLD}=\left|\{\,\mathrm{SLD}\in\text{social}\,\}\right|",
        explanation_md=(
            "Number of distinct social domains contacted. Lower values suggest a narrower social interaction footprint."
        ),
    ),
    MetricDef(
        id="C1/F8",
        label="Mean social‑session duration (s)",
        dist_col="F2_MeanSocialDurSec",
        ok=120.0, higher_is_worse=True, fmt_tpl="{v:.0f} s",
        latex_formula=r"\bar{T}_{\text{social}}=\frac{1}{K}\sum_{k=1}^K (t^{\text{end}}_k-t^{\text{start}}_k)",
        explanation_md=(
            "Average duration of social sessions segmented by a 5‑minute idle gap. Shorter durations may reflect fragmented or cursory interactions."
        ),
    ),
    MetricDef(
        id="C1/F9",
        label="Long streaming flows (≥2h)",
        dist_col="F3_LongStreams",
        ok=0.0, higher_is_worse=True, fmt_tpl="{v:.0f}",
        latex_formula=r"\#\{\text{streaming sessions with }T\ge 2\,\mathrm{h}\}",
        explanation_md=(
            "Number of long, continuous streaming episodes (≥2 hours). High counts can signal late‑night media consumption."
        ),
    ),
    MetricDef(
        id="C1/F10",
        label="Down/Up byte ratio",
        dist_col="F4_DownUpRatio",
        ok=20.0, higher_is_worse=True, fmt_tpl="{v:.2f}",
        latex_formula=r"\mathrm{R}=\frac{B_{\downarrow}}{B_{\uparrow}}",
        latex_numbers_tpl=r"\mathrm{R}=\frac{{bytes_down}}{{bytes_up}}={value:.2f}",
        explanation_md=(
            "Ratio of downstream to upstream bytes (private↔public IP direction as approximation). Extremely high ratios point to passive media consumption with limited outgoing activity."
        ),
    ),
    MetricDef(
        id="C1/F11",
        label="Revisit‑ratio (DNS)",
        dist_col="F5_RevisitRatio",
        ok=0.5, higher_is_worse=False, fmt_tpl="{v:.2f}",
        latex_formula=r"\mathrm{Revisit}=1-\frac{|\text{unique SLD}|}{|\text{hits}|}",
        explanation_md=(
            "Share of unique domains among DNS queries. Lower ratios mean frequent revisits to the same sites."
        ),
    ),
    MetricDef(
        id="C1/F12",
        label="Short‑interval repeats (Fano)",
        dist_col="F6_Fano",
        ok=1.5, higher_is_worse=True, fmt_tpl="{v:.2f}",
        latex_formula=r"\mathrm{Fano}=\frac{\mathrm{Var}(N)}{\mathbb{E}[N]}",
        latex_numbers_tpl=r"\mathrm{Fano}={value:.2f}",
        explanation_md=(
            "Fano factor of short‑interval repeat counts per client/domain within 30‑minute windows. Higher values capture bursty checking patterns."
        ),
    ),
    MetricDef(
        id="C1/F13",
        label="Hourly CV",
        dist_col="F7_HourlyCV",
        ok=0.3, higher_is_worse=False, fmt_tpl="{v:.2f}",
        latex_formula=r"\mathrm{CV}_{\text{hour}}=\frac{\sigma(\mathbf{h})}{\mu(\mathbf{h})},\ \mathbf{h}\in\mathbb{R}^{24}",
        latex_numbers_tpl=r"\mathrm{CV}={value:.2f}",
        explanation_md=(
            "Coefficient of variation across hourly bins (0–23). Lower values imply a flatter day/night rhythm."
        ),
    ),
    MetricDef(
        id="C1/F14",
        label="Night/Day traffic ratio (pkts)",
        dist_col="F8_NightDayRatioPkts",
        ok=0.4, higher_is_worse=True, fmt_tpl="{v:.2f}",
        latex_formula=r"\mathrm{R}=\frac{N_{\text{pkts, night}}}{N_{\text{pkts, day}}}",
        latex_numbers_tpl=r"\mathrm{R}=\frac{{night_pkts_today}}{{day_pkts_today}}={value:.2f}",
        explanation_md=(
            "Packet‑count ratio for night vs day. High ratios indicate a pronounced nocturnal shift."
        ),
    ),
]

# Replace short explanations with clearer, non-technical descriptions (4–6 sentences each)
_C1_LONG_EXPL: Dict[str, str] = {
    "C1/F1": (
        "Share of all daily network packets that occur during the late-night window (00:00-05:00). "
        "A higher share means more of your activity happens when most people sleep, which can signal a delayed sleep phase or disrupted sleep. "
        "Short-term spikes can be caused by travel, late work, or a movie night, so look for patterns across several days rather than single-day changes. "
        "Values near zero are typical for consolidated sleep. "
        "This is a simple, privacy-preserving count and does not depend on the content of your traffic."
    ),
    "C1/F2": (
        "The longest time gap between any two packets in the day, expressed in hours. "
        "Large gaps are consistent with consolidated, uninterrupted sleep or other long offline periods. "
        "Very short maximum gaps can indicate fragmented nights with repeated wakefulness or frequent device use. "
        "Daytime naps may reduce this value, so interpret in context with night-specific metrics. "
        "This metric is robust and does not require sensitive content information."
    ),
    "C1/F3": (
        "Counts how many minutes between 00:00 and 05:00 show any network activity at all. "
        "Higher counts suggest late bedtimes, frequent awakenings, or nocturnal device use. "
        "Occasional late nights are normal; look for sustained increases over several consecutive days. "
        "Background updates and notifications can add a few minutes, but large values usually reflect user activity. "
        "Compare against your personal baseline instead of a universal rule."
    ),
    "C1/F4": (
        "Interdaily Stability measures how regular your 24-hour activity pattern is across days. "
        "Higher values mean your daily schedule (sleep/wake and active periods) repeats consistently. "
        "Lower values indicate day-to-day variability, which can happen with shift work, travel, or irregular routines. "
        "It is a standard actigraphy-style measure and complements the fragmentation score (IV). "
        "Use trends over time; it is not a diagnosis on its own."
    ),
    "C1/F5": (
        "Intradaily Variability captures how choppy or fragmented your activity is within a day. "
        "Higher values reflect frequent switching between active and inactive periods, like many brief check-ins. "
        "Lower values reflect smoother blocks of activity and rest. "
        "This measure comes from actigraphy research and is computed from minute-level activity. "
        "Consider it together with IS (regularity) and night-specific indicators."
    ),
    "C1/F6": (
        "Ratio of active minutes at night (roughly 00:00-05:00) to active minutes during the day. "
        "A larger ratio means a greater share of your activity occurs at night. "
        "Values near zero are typical when most activity is daytime-centered. "
        "This is a simple, interpretable indicator of nocturnal shift and complements the packet-based night/day ratio. "
        "If daytime minutes are extremely low, the ratio can be undefined or very large—inspect both parts."
    ),
    "C1/F7": (
        "Counts how many different social platforms or messaging domains you engaged with. "
        "A lower number over time can point to a narrower social footprint or fewer channels of interaction. "
        "Spikes and dips can be normal—weekends, holidays, and focused work days affect this. "
        "The count is privacy-preserving: it considers only domain categories, not message content. "
        "Interpret together with session counts and reply latency for a fuller picture."
    ),
    "C1/F8": (
        "Average duration of social-messaging sessions, where a 5-minute pause ends a session. "
        "Longer sessions suggest more sustained conversations; shorter ones indicate brief check-ins. "
        "Network conditions and notification batching can influence timing, but trends across days are informative. "
        "Use alongside session count and upstream share to separate engagement from passive presence. "
        "There is no single 'good' value—compare to your usual pattern."
    ),
    "C1/F9": (
        "Counts long inbound-streaming sessions lasting at least two hours (5-minute merging rule). "
        "Occasional long sessions are normal (e.g., a movie), but frequent or late-night ones can delay sleep. "
        "We only look at traffic shape, not content, to protect privacy. "
        "Combine with late-night share and night minutes to understand whether media use overlaps with typical sleep time. "
        "Use multi-day trends rather than isolated events."
    ),
    "C1/F10": (
        "Ratio of downstream (to your device) to upstream (from your device) bytes. "
        "Higher ratios indicate more passive consumption (reading, watching) relative to active creation (typing, posting). "
        "Extremely high values or 'infinite' can happen when there is virtually no upload traffic. "
        "Consider this together with social metrics to distinguish quiet browsing from active chatting. "
        "This metric is content-agnostic and aims to capture interaction style."
    ),
    "C1/F11": (
        "Measures how often you revisit the same domains compared with visiting new ones. "
        "A higher ratio means more repeat visits relative to unique sites, which can reflect habitual checking. "
        "Automated refreshes and background polls can increase this slightly; trends matter more than single days. "
        "This is not good or bad on its own—pair it with engagement and timing metrics. "
        "Privacy is preserved: only domain names are considered, not content."
    ),
    "C1/F12": (
        "Fano factor compares variability to the average of 1-minute packet counts. "
        "Values near 1 resemble random (Poisson-like) activity; values above 1 indicate bursty periods, and below 1 indicate regular pacing. "
        "Higher Fano values can reflect bursts of short checks or notifications. "
        "Sparse days can make this unstable—use together with other daily activity measures. "
        "It summarizes within-day variability without needing any content details."
    ),
    "C1/F13": (
        "Coefficient of variation (standard deviation divided by mean) across the 24 hourly packet counts. "
        "Lower values mean activity is more evenly distributed across hours; higher values mean sharp peaks. "
        "Work schedules, travel, or unusual events can temporarily raise this. "
        "Track relative changes over time to see whether your pattern becomes more or less concentrated."
    ),
    "C1/F14": (
        "Ratio of total packet counts at night to packet counts during the day. "
        "High ratios indicate a stronger shift of traffic into typical sleep hours. "
        "Because it counts packets rather than minutes, it complements the night/day activity-minutes ratio. "
        "Background downloads may inflate packet counts; interpret with the late-night share and active-minutes metrics. "
        "Use sustained trends across days to decide whether this reflects a meaningful change."
    ),
}

# Build a new list with upgraded explanations (MetricDef is frozen, so rebuild)
C1_DEFS = [
    MetricDef(
        id=m.id,
        label=m.label if m.id != "C1/F3" else "Active Night Minutes (00-05)",
        dist_col=m.dist_col,
        ok=m.ok,
        higher_is_worse=m.higher_is_worse,
        fmt_tpl=m.fmt_tpl,
        latex_formula=m.latex_formula,
        latex_numbers_tpl=m.latex_numbers_tpl,
        explanation_md=_C1_LONG_EXPL.get(m.id, m.explanation_md),
        missing_md=m.missing_md,
    )
    for m in C1_DEFS
]

# ----------------------------- helpers (C1 only) -------------------------------

def _per_min_active_series(df: pd.DataFrame) -> pd.Series:
    if df.empty or "Timestamp" not in df.columns:
        return pd.Series([], dtype=float)
    per_min = df.set_index("Timestamp").assign(cnt=1)["cnt"].resample("1Min").sum()
    return (per_min > 0).astype(float)

def _active_minutes(df: pd.DataFrame, hours: set[int]) -> int:
    s = _per_min_active_series(df)
    if s.empty:
        return 0
    return int(s[s.index.hour.isin(hours)].sum())

def _longest_inactivity_hours(df: pd.DataFrame) -> float:
    if df.empty or "Timestamp" not in df.columns:
        return np.nan
    d = df["Timestamp"].dropna().sort_values()
    if d.shape[0] < 2:
        return np.nan
    gaps_s = np.diff(d.values).astype("timedelta64[s]").astype(int)
    return float(np.max(gaps_s) / 3600.0) if gaps_s.size else np.nan

def _chat_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[chat_mask(df)].copy() if not df.empty else df

def _distinct_social_sld(df: pd.DataFrame) -> float:
    d = _chat_subset(df)
    if d.empty or "SLD" not in d.columns:
        return np.nan
    return float(d["SLD"].dropna().nunique())

def _mean_social_session_sec(df: pd.DataFrame, gap_sec: int = 300) -> float:
    d = _chat_subset(df)
    if d.empty or "Timestamp" not in d.columns:
        return np.nan
    sess = sessions_from_timestamps(d, gap_sec=gap_sec)
    if not sess:
        return np.nan
    L = [(b - a).total_seconds() for a, b in sess]
    return float(np.mean(L)) if L else np.nan

def _count_long_streams_ge_2h(df: pd.DataFrame, gap_sec: int = 300) -> Tuple[float, Dict[str, Any]]:
    rows = df.loc[streaming_inbound_mask(df)] if "Timestamp" in df.columns else pd.DataFrame()
    if rows.empty:
        return 0.0, {"long_streams": 0}
    sess = sessions_from_timestamps(rows, gap_sec=gap_sec)
    count = sum(1 for (a, b) in sess if (b - a).total_seconds() >= 7200.0)
    return float(count), {"long_streams": int(count)}

def _down_up_byte_ratio(df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    if df.empty or not {"Source IP", "Destination IP", "Length"}.issubset(df.columns):
        return np.nan, {}
    down = float(df.loc[df.apply(lambda r: is_inbound(r["Source IP"], r["Destination IP"]), axis=1), "Length"].sum())
    up   = float(df.loc[df.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1), "Length"].sum())
    if up <= 0 and down <= 0:
        return np.nan, {}
    if up <= 0:
        return np.inf, {"bytes_down": int(down), "bytes_up": 0}
    return (down / up), {"bytes_down": int(down), "bytes_up": int(up)}

def _revisit_ratio_dns(df: pd.DataFrame) -> float:
    """(hits - unique)/hits over SLD; falls back to DNS_QNAME/ResolvedHost if SLD missing."""
    col = None
    for c in ("SLD", "DNS_QNAME", "DNS_ANS_NAME", "ResolvedHost"):
        if c in df.columns and df[c].notna().any():
            col = c; break
    if not col:
        return np.nan
    s = df[col].dropna().astype(str)
    total = int(s.shape[0])
    if total == 0:
        return np.nan
    uniq = int(s.nunique())
    return float((total - uniq) / total)

def _fano_1min(df: pd.DataFrame) -> float:
    """Fano factor of 1-min packet counts."""
    if df.empty or "Timestamp" not in df.columns:
        return np.nan
    per_min = df.set_index("Timestamp").assign(cnt=1)["cnt"].resample("1Min").sum()
    if per_min.empty:
        return np.nan
    m = float(per_min.mean())
    if m <= 0:
        return np.nan
    v = float(per_min.var(ddof=0))
    return float(v / m)

def _hourly_cv(df: pd.DataFrame) -> float:
    if df.empty or "Timestamp" not in df.columns:
        return np.nan
    per_h = df.set_index("Timestamp").assign(cnt=1)["cnt"].resample("1h").sum()
    if per_h.empty:
        return np.nan
    mu = float(per_h.mean())
    if mu <= 0:
        return np.nan
    sd = float(per_h.std(ddof=0))
    return float(sd / mu)

def _night_day_ratio_pkts(df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    if df.empty or "Timestamp" not in df.columns:
        return np.nan, {}
    d = df.copy()
    d["Hour"] = pd.to_datetime(d["Timestamp"]).dt.hour
    night = int(d["Hour"].isin([0,1,2,3,4,5]).sum())
    day   = int(d["Hour"].isin(range(6,22)).sum())
    if day <= 0 and night <= 0:
        return np.nan, {}
    if day <= 0:
        return np.inf, {"night_pkts_today": night, "day_pkts_today": 0}
    return float(night / day), {"night_pkts_today": night, "day_pkts_today": day}

# ------------------------------ main class ------------------------------------

class Criterion1:
    """Sleep disturbance — C1 (self-contained, no Excel catalog)."""

    def _compute_today_fields(self, df_day: pd.DataFrame) -> Dict[str, Any]:
        """Compute all C1 DistCols from df_day (robust against missing cols)."""
        rec: Dict[str, Any] = {}

        # F1..F6 come from today/aux in the dashboard pipeline, but we ensure fallbacks:
        # F1 LateNightShare
        if "Timestamp" in df_day.columns:
            t = pd.to_datetime(df_day["Timestamp"], errors="coerce")
            night = int(t.dt.hour.isin([0,1,2,3,4,5]).sum())
            total = int(t.notna().sum())
            rec["LateNightShare"] = (night / float(total)) if total > 0 else np.nan
        else:
            rec["LateNightShare"] = np.nan

        # F2 LongestInactivityHours
        rec["LongestInactivityHours"] = _longest_inactivity_hours(df_day)

        # F3 ActiveNightMinutes
        rec["ActiveNightMinutes"] = float(_active_minutes(df_day, {0,1,2,3,4,5}))

        # IS/IV & ND-Ratio are provided via aux_ctx, but compute safe fallbacks if needed
        # (the dashboard already computes IS/IV/ND for you — we won't redo heavy calc here)

        # F7 Distinct social SLD
        rec["F1_DistinctSocial"] = _distinct_social_sld(df_day)

        # F8 Mean social-session duration (s)
        rec["F2_MeanSocialDurSec"] = _mean_social_session_sec(df_day, gap_sec=300)

        # F9 Long streaming sessions (>=2h)
        rec["F3_LongStreams"], _ = _count_long_streams_ge_2h(df_day, gap_sec=300)

        # F10 Down/Up ratio
        val_du, ex_du = _down_up_byte_ratio(df_day)
        rec["F4_DownUpRatio"] = val_du
        # Keep extras for LaTeX
        rec["_EXTRAS_F4"] = ex_du

        # F11 Revisit ratio (DNS/SLD)
        rec["F5_RevisitRatio"] = _revisit_ratio_dns(df_day)

        # F12 Fano
        rec["F6_Fano"] = _fano_1min(df_day)

        # F13 Hourly CV
        rec["F7_HourlyCV"] = _hourly_cv(df_day)

        # F14 Night/Day ratio (pkts)
        val_nd, ex_nd = _night_day_ratio_pkts(df_day)
        rec["F8_NightDayRatioPkts"] = val_nd
        rec["_EXTRAS_F8"] = ex_nd

        return rec

    def _latex_context_for(self, mdef: MetricDef, value: Any, aux_ctx: Dict[str, Any], today_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Build a robust LaTeX context per metric."""
        ctx: Dict[str, Any] = {}
        # Generic placeholders
        ctx = update_extras_with_value(ctx, value)
        # Map F1 needs
        if mdef.id == "C1/F1":
            ctx.setdefault("n_night", aux_ctx.get("n_night_packets_today"))
            ctx.setdefault("n_total", aux_ctx.get("n_total_packets_today"))
        # F10 needs bytes
        if mdef.id == "C1/F10":
            ex = today_fields.get("_EXTRAS_F4", {})
            ctx.setdefault("bytes_down", ex.get("bytes_down", aux_ctx.get("down_bytes_today")))
            ctx.setdefault("bytes_up", ex.get("bytes_up", aux_ctx.get("up_bytes_today")))
        # F14 needs pkts today
        if mdef.id == "C1/F14":
            ex = today_fields.get("_EXTRAS_F8", {})
            ctx.setdefault("night_pkts_today", ex.get("night_pkts_today", aux_ctx.get("night_pkts_today")))
            ctx.setdefault("day_pkts_today", ex.get("day_pkts_today", aux_ctx.get("day_pkts_today")))
        return ctx

    def compute(
        self,
        df_day: pd.DataFrame,
        today: dict,         # today_base from base_features.compute_daily_base_record
        aux_ctx: dict,       # IS/IV/ND + counts already computed in the page
        ALL_DAILY: pd.DataFrame
    ) -> List[Dict[str, Any]]:

        aux_ctx = dict(aux_ctx or {})
        today = dict(today or {})

        need_calc = any(
            (m.dist_col not in today) or pd.isna(today.get(m.dist_col)) for m in C1_DEFS
        )
        today_fields = (
            self._compute_today_fields(df_day) if need_calc and df_day is not None and not df_day.empty else {}
        )

        # Prefer pipeline-provided values where available (IS/IV/ND)
        if "IS_val" in aux_ctx:
            today.setdefault("IS", aux_ctx.get("IS_val"))
        if "IV_val" in aux_ctx:
            today.setdefault("IV", aux_ctx.get("IV_val"))
        if "nd_ratio" in aux_ctx:
            today.setdefault("ND_Ratio", aux_ctx.get("nd_ratio"))

        for k, v in today_fields.items():
            today.setdefault(k, v)

        metrics_out: List[Dict[str, Any]] = []

        for mdef in C1_DEFS:
            value = today.get(mdef.dist_col, np.nan)

            # Build status tuple & ranges text
            if (value is None) or (isinstance(value, float) and (np.isnan(value) or not np.isfinite(value))) or (mdef.ok is None):
                status_tuple = ("N/A", "blue")
                ranges_str = "" if mdef.ok is None else thresholds_text(mdef.ok, higher_is_worse=mdef.higher_is_worse)
            else:
                v = float(value)
                is_ok = (v <= mdef.ok) if mdef.higher_is_worse else (v >= mdef.ok)
                status_tuple = ("OK", "green") if is_ok else ("Caution", "orange")
                ranges_str   = thresholds_text(mdef.ok, higher_is_worse=mdef.higher_is_worse)

            # Human formatter
            fmt = fmt_from_template(mdef.fmt_tpl)

            # LaTeX numbers (safe fill)
            latex_numbers = None
            if isinstance(mdef.latex_numbers_tpl, str) and mdef.latex_numbers_tpl.strip():
                lctx = self._latex_context_for(mdef, value, aux_ctx, today_fields)
                latex_numbers = latex_fill_from_template(mdef.latex_numbers_tpl, lctx)

            # Assemble metric dict (consumed by the dashboard)
            metrics_out.append(
                make_metric(
                    label=mdef.label,
                    value=value,
                    fmt=fmt,
                    status_tuple=status_tuple,
                    ranges_str=ranges_str,
                    latex_formula=mdef.latex_formula,
                    latex_numbers=latex_numbers,
                    heuristic_md=mdef.explanation_md,
                    missing_md=mdef.missing_md,
                    dist_col=mdef.dist_col,
                    range_cfg=({"ok": mdef.ok, "higher_is_worse": mdef.higher_is_worse} if mdef.ok is not None else None),
                )
            )

        return metrics_out
