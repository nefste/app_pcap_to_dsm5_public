from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import make_metric, thresholds_text
from .common import (
    sessions_from_timestamps,
    fmt_from_template,
    latex_fill_from_template,
    update_extras_with_value,
    resolve_value,
)


@dataclass(frozen=True)
class MetricDef:
    id: str
    label: str
    dist_col: str
    value_ref: str
    ok: Optional[float]
    higher_is_worse: bool
    fmt_tpl: str
    latex_formula: Optional[str] = None
    latex_numbers_tpl: Optional[str] = None
    explanation_md: Optional[str] = None
    missing_md: Optional[str] = None


# ---------------------------- helpers ---------------------------------

def _longest_midday_idle(df: pd.DataFrame) -> tuple[float, Dict[str, Any]]:
    if df is None or df.empty or "Timestamp" not in df.columns:
        return (np.nan, {})
    d = df.copy()
    d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="coerce")
    d = d.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    day = d["Timestamp"].dt.normalize().iloc[0] if not d.empty else pd.NaT
    start = day + pd.Timedelta(hours=12)
    end = day + pd.Timedelta(hours=18)
    mid = d[(d["Timestamp"] >= start) & (d["Timestamp"] <= end)]["Timestamp"].sort_values()
    if mid.empty:
        gap = (end - start).total_seconds() / 60.0
        return (gap, {"gap_min": gap})
    times = pd.concat([pd.Series([start]), mid, pd.Series([end])])
    gaps = times.diff().dt.total_seconds().iloc[1:] / 60.0
    gap = float(gaps.max()) if gaps.size else np.nan
    return (gap, {"gap_min": gap})


def _day_active_session_count(df: pd.DataFrame, gap_sec: int = 300) -> tuple[float, Dict[str, Any]]:
    if df is None or df.empty or "Timestamp" not in df.columns:
        return (np.nan, {})
    sessions = sessions_from_timestamps(df.sort_values("Timestamp"), gap_sec=gap_sec)
    count = 0
    for a, _ in sessions:
        if 9 <= a.hour < 18:
            count += 1
    return (float(count), {"count": count})


def _day_idle_ratio(df: pd.DataFrame) -> tuple[float, Dict[str, Any]]:
    if df is None or df.empty or "Timestamp" not in df.columns:
        return (np.nan, {})
    d = df.copy()
    d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="coerce")
    d = d.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    per_min = d.set_index("Timestamp").assign(cnt=1)["cnt"].resample("1Min").sum()
    window = per_min.between_time("09:00", "17:59")
    idle = int((window == 0).sum())
    ratio = idle / 540.0 if len(window) else np.nan
    return (ratio, {"idle_min": idle})


# ---------------------------- definitions ------------------------------

C6_DEFS: List[MetricDef] = [
    MetricDef(
        id="C6/F1",
        label="Longest midday idle gap (12–18 h)",
        dist_col="C6_F1_LongestMiddayIdleMin",
        value_ref="TR:C6_F1_LongestMiddayIdleMin",
        ok=90.0, higher_is_worse=True, fmt_tpl="{v:.0f} min",
        latex_formula=r"\mathrm{Gap}_{12\text{–}18}=\max\limits_{t\in[12{:}00,18{:}00]} \Delta t",
        latex_numbers_tpl=r"\mathrm{Gap}_{12\text{–}18}={gap_min}\,\mathrm{min}",
        explanation_md=(
            "This metric looks at the longest continuous period with **no network traffic** between 12:00 and 18:00. "
            "It approximates extended mid‑day inactivity when someone may be too tired to engage with their device. "
            "Values above ~90 minutes, especially if they exceed the person’s typical pattern, suggest reduced daytime energy. "
            "Prolonged mid‑day gaps have been associated with self‑reported fatigue in passive sensing studies of students and workers. "
            "Interpretation should consider schedule context (e.g., planned meetings) and Wi‑Fi coverage. Shorter gaps indicate more frequent engagement and are less suggestive of low energy."
        ),
        missing_md="If timestamps are sparse or the capture misses mid‑day hours, the gap may be over‑ or under‑estimated. In that case, treat the status as **N/A** and rely on the other C6 indicators.",
    ),
    MetricDef(
        id="C6/F2",
        label="Day‑active session count (09–18 h)",
        dist_col="C6_F2_DayActiveSessionCount",
        value_ref="TR:C6_F2_DayActiveSessionCount",
        ok=5.0, higher_is_worse=False, fmt_tpl="{v:.0f}",
        latex_formula=r"N_{\text{sess,day}}=\#\{\text{sessions starting in }[09{:}00,18{:}00]\}",
        latex_numbers_tpl=r"N_{\text{sess,day}}={count}",
        explanation_md=(
            "We merge packets into user sessions using a 5‑minute idle rule and count those that **start** between 09:00 and 18:00. "
            "A lower count over several days can reflect reduced initiation of tasks or social exchanges during typical work hours. "
            "This aligns with reports that decreased daily activity rhythms accompany depressive fatigue. The measure captures **starts**, not duration, so it is robust to a few very long sessions. Context matters: holidays or off‑days may naturally reduce this count. Higher values indicate more frequent engagement and are generally a good sign for energy."
        ),
        missing_md="If sessionization fails (e.g., missing timestamps), this metric is **N/A**. Rely on C6/F1 and C6/F3 for daytime inactivity in such cases.",
    ),
    MetricDef(
        id="C6/F3",
        label="Day‑time idle ratio (09–18 h)",
        dist_col="C6_F3_DayIdleRatio_09_18",
        value_ref="TR:C6_F3_DayIdleRatio_09_18",
        ok=0.4, higher_is_worse=True, fmt_tpl="{v:.2f}",
        latex_formula=r"R_{\text{idle}}=\dfrac{\text{idle mins}_{09\text{–}18}}{540}",
        latex_numbers_tpl=r"R_{\text{idle}}=\frac{ {idle_min} }{540}={value}",
        explanation_md=(
            "This ratio is the fraction of minutes between 09:00 and 18:00 with **no traffic at all**. "
            "Higher values suggest long blocks of inactivity and can be a behavioral sign of low energy. Because it is normalized by the 9‑hour window, it is comparable across days. It is robust to brief notifications; only **minute‑level** silence counts as idle. Changes should be interpreted relative to the person’s typical weekday pattern. Sustained values above 0.40 often coincide with days where people report feeling unusually tired."
        ),
        missing_md="If hourly coverage is incomplete (e.g., device offline), the ratio may be biased. Treat as **N/A** and corroborate with C6/F1 and C6/F2.",
    ),
    MetricDef(
        id="C6/F4",
        label="Upstream bytes per active hour",
        dist_col="C6_F4_UpstreamBpsPerActiveHour",
        value_ref="FUNC:UpstreamBpsPerActiveHour",
        ok=1_000_000.0, higher_is_worse=False, fmt_tpl="{v:.1f} B/s",
        latex_formula=r"\frac{B_{\uparrow}}{\text{active hours}}",
        latex_numbers_tpl=r"\dfrac{ {sum_up_bytes} }{ {active_hours} }={value}\ \mathrm{B/h}",
        explanation_md=(
            "We sum bytes sent **from the device to public IPs** (upstream) and divide by the number of hours that had any traffic. "
            "Upstream activity usually comes from typing, sending messages, syncing documents, or uploading media—actions that take effort. "
            "Lower values over several days can signal less productive output and reduced initiative. Because we normalize by the number of active hours, the metric is less sensitive to days with shorter device use. Absolute values vary by person and apps; track trends relative to the personal baseline. Spikes may also reflect automated backups, so always cross‑check with session counts (C6/F2)."
        ),
        missing_md="If IP direction cannot be inferred (missing Source/Destination IP), upstream bytes cannot be computed; status becomes **N/A**. In that case, rely on C6/F2 and C6/F5 for effortful interaction.",
    ),
    MetricDef(
        id="C6/F5",
        label="POST + chat messages / day (proxy)",
        dist_col="C6_F5_PostChatCount",
        value_ref="FUNC:PostChatCount",
        ok=50.0, higher_is_worse=False, fmt_tpl="{v:.0f}",
        latex_formula=r"N_{\text{POST+chat}}=\#\{\text{outbound to }(\text{MSG ports}\cup\text{social SLDs})\}",
        latex_numbers_tpl=r"N_{\text{POST+chat}}={count}",
        explanation_md=(
            "This is a **proxy** for the number of expressive actions, combining outbound packets to messaging ports and well‑known social domains. "
            "A sustained decline suggests less initiating or responding, which often accompanies low‑energy days. Because HTTP method parsing is not available in PCAP metadata here, we do not isolate POST requests directly. Still, the combination of messaging ports and social SLDs tracks communication effort well in practice. Interpretation of large day‑to‑day swings requires caution; batching effects from certain apps may occur. Use together with C6/F4 to distinguish passive browsing from active output."
        ),
        missing_md="If destination ports or SLD enrichment are missing, counts may be incomplete; treat as **N/A** and consult C6/F2 and C6/F4.",
    ),
    MetricDef(
        id="C6/F6",
        label="Median inter‑request interval (s)",
        dist_col="C6_F6_MedianInterReqSec",
        value_ref="FUNC:MedianInterReqSec",
        ok=10.0, higher_is_worse=True, fmt_tpl="{v:.1f} s",
        latex_formula=r"\tilde{\ell}=\operatorname{median}(\Delta t_i)\quad\text{over DNS/GET requests}",
        latex_numbers_tpl=r"\tilde{\ell}={median_sec}\ \mathrm{s}",
        explanation_md=(
            "We measure the time between successive **requests** during browsing (preferably DNS queries). The daily median grows when a person proceeds more slowly from action to action, which can reflect fatigue. Compared to raw packet gaps, request‑level gaps better capture user‑driven steps. Short medians indicate a steady tempo; long medians suggest slower progress through tasks. The measure should be compared against the individual’s normal range rather than an absolute cutoff. If there are very few requests, the median may be unstable."
        ),
        missing_md="If DNS traffic is absent or heavily cached, the request stream may be too sparse; mark **N/A** and use C6/F7 and C6/F2 instead.",
    ),
    MetricDef(
        id="C6/F7",
        label="Inter‑request Fano factor",
        dist_col="C6_F7_InterReqFano",
        value_ref="FUNC:InterReqFano",
        ok=1.5, higher_is_worse=True, fmt_tpl="{v:.2f}",
        latex_formula=r"F=\dfrac{\operatorname{Var}(\ell)}{\operatorname{E}[\ell]}\quad\text{(request gaps } \ell)",
        latex_numbers_tpl=r"F=\frac{ {var_sec} }{ {mean_sec} }={value}",
        explanation_md=(
            "The Fano factor compares the variability of request gaps to their average length. Values near 1 indicate regular tempo; higher values indicate **bursty** activity with long pauses punctuated by short spurts. Such burstiness is commonly seen on low‑energy days where effort comes in brief windows. Because F is ratio‑based, it is relatively stable across different absolute browsing speeds. Track trends across weeks; persistent values above ~1.5 merit attention. Use together with C6/F6 to distinguish “slow but steady” from “stop‑and‑go” patterns."
        ),
        missing_md="With too few requests, variance and mean are unreliable; mark **N/A** and interpret alongside C6/F6.",
    ),
    MetricDef(
        id="C6/F8",
        label="First‑activity time (after 04:00)",
        dist_col="C6_F8_FirstActivityMin",
        value_ref="FUNC:FirstActivityMin",
        ok=540.0, higher_is_worse=True, fmt_tpl="{v:.0f} min",
        latex_formula=r"t_{\text{first}}=\min\{t\ge 04{:}00\}",
        latex_numbers_tpl=r"t_{\text{first}}={first_hhmm}\ \,({value}\ \mathrm{min})",
        explanation_md=(
            "We record the first packet time after 04:00 as a proxy for the day’s **activation**. Much later‑than‑usual first activity suggests slower morning start, a typical fatigue marker. Because schedules differ widely, this metric is most meaningful relative to each person’s baseline. Weekends often shift later; consider weekday/weekend context in interpretation. If the device stays offline until noon due to travel or maintenance, this will appear as a late activation but should be disregarded. Combine with C6/F9 to quantify deviation from the personal norm."
        ),
        missing_md="If no traffic occurs after 04:00, the value is undefined (**N/A**). Use C6/F3 to assess daytime inactivity instead.",
    ),
    MetricDef(
        id="C6/F9",
        label="Activation delay vs baseline (min)",
        dist_col="C6_F9_ActivationDelayVsBase28d",
        value_ref="FUNC:DELTA_FROM_MEDIAN(col='C6_F8_FirstActivityMin', window=28)",
        ok=60.0, higher_is_worse=True, fmt_tpl="{v:.0f} min",
        latex_formula=r"\Delta t=t_{\text{first}}-\operatorname{median}(t_{\text{first}})_{28\ \text{days}}",
        latex_numbers_tpl=r"\Delta t={delta_min}\ \mathrm{min}\ \ (\text{baseline } {baseline_first_hhmm})",
        explanation_md=(
            "This measures how much later (in minutes) today’s first activity starts compared with the **28‑day personal median**. Positive values mean delayed activation; sustained delays point to a low‑energy phase. Using a personal baseline makes the measure robust to chronotype differences (early birds vs night owls). We recommend inspecting the time‑series; step‑ups after stressful events or illness are common. Large negative values (starting earlier) can also be informative, e.g., due to early obligations. If the baseline cannot be computed yet (too little history), the metric is **N/A**."
        ),
        missing_md="If ALL_DAILY lacks the historical first‑activity column, the delay cannot be computed; mark **N/A**. Once the baseline is available, the metric will populate automatically.",
    ),
]


# ----------------------------- main class --------------------------------

class Criterion6:
    """Fatigue / low energy — C6 (self-contained)."""

    def _compute_today_fields(self, df_day: pd.DataFrame) -> Dict[str, Any]:
        rec: Dict[str, Any] = {}
        rec["C6_F1_LongestMiddayIdleMin"], rec["_EX_F1"] = _longest_midday_idle(df_day)
        rec["C6_F2_DayActiveSessionCount"], rec["_EX_F2"] = _day_active_session_count(df_day)
        rec["C6_F3_DayIdleRatio_09_18"], rec["_EX_F3"] = _day_idle_ratio(df_day)
        return rec

    def compute(
        self,
        df_day: pd.DataFrame,
        today: dict,
        aux_ctx: dict,
        ALL_DAILY: pd.DataFrame,
    ) -> List[Dict[str, Any]]:

        aux = dict(aux_ctx or {})
        aux.setdefault("ALL_DAILY", ALL_DAILY)
        today = dict(today or {})

        need_calc = any(
            (m.dist_col not in today) or pd.isna(today.get(m.dist_col)) for m in C6_DEFS if m.value_ref.startswith("TR:")
        )
        if need_calc and df_day is not None and not df_day.empty:
            today_fields = self._compute_today_fields(df_day)
            for k, v in today_fields.items():
                today.setdefault(k, v)
        metrics_out: List[Dict[str, Any]] = []
        for mdef in C6_DEFS:
            val = today.get(mdef.dist_col)
            extras: Dict[str, Any] = {}
            if (val is None) or (isinstance(val, float) and np.isnan(val)):
                val, extras = resolve_value(mdef.value_ref, df_day, today, aux, today)
                today.setdefault(mdef.dist_col, val)
            elif df_day is not None and not df_day.empty:
                # value already cached; optionally recompute extras when raw data present
                _, extras = resolve_value(mdef.value_ref, df_day, today, aux, today)
            extras = update_extras_with_value(extras, val)

            if (val is None) or (isinstance(val, float) and (np.isnan(val) or not np.isfinite(val))) or (mdef.ok is None):
                status_tuple = ("N/A", "blue")
                ranges_str = "" if mdef.ok is None else thresholds_text(mdef.ok, higher_is_worse=mdef.higher_is_worse)
            else:
                v = float(val)
                is_ok = (v <= mdef.ok) if mdef.higher_is_worse else (v >= mdef.ok)
                status_tuple = ("OK", "green") if is_ok else ("Caution", "orange")
                ranges_str = thresholds_text(mdef.ok, higher_is_worse=mdef.higher_is_worse)

            fmt = fmt_from_template(mdef.fmt_tpl)
            latex_numbers = (
                latex_fill_from_template(mdef.latex_numbers_tpl, extras)
                if isinstance(mdef.latex_numbers_tpl, str) and mdef.latex_numbers_tpl.strip()
                else None
            )

            metrics_out.append(
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
                    range_cfg={"ok": mdef.ok, "higher_is_worse": mdef.higher_is_worse}
                    if mdef.ok is not None
                    else None,
                )
            )

        return metrics_out

