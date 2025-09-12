from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd

from .base import make_metric, thresholds_text
from .common import (
    fmt_from_template,
    latex_fill_from_template,
    update_extras_with_value,
    sessions_from_timestamps,
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


# ----------------------------- helpers ---------------------------------------

def _hhmm(ts: pd.Timestamp | None) -> str | None:
    if ts is None or (isinstance(ts, float) and (np.isnan(ts) or not np.isfinite(ts))):
        return None
    try:
        return pd.to_datetime(ts).strftime("%H:%M")
    except Exception:
        return None

def _minutes_from_2200(ts: pd.Timestamp) -> float:
    """Minutes between 22:00 (same local day anchor) and timestamp; wraps across midnight."""
    anchor = ts.normalize() + pd.Timedelta(hours=22)
    if ts < anchor:
        ts = ts + pd.Timedelta(days=1)
    return float((ts - anchor).total_seconds() / 60.0)

def _find_night_idle_gap(df: pd.DataFrame, min_gap_min: int = 30) -> Tuple[pd.Timestamp | None, pd.Timestamp | None, float | None]:
    """Return (onset_ts, wake_ts, gap_minutes) for the first ≥min_gap within 22–04h window."""
    if df is None or df.empty or "Timestamp" not in df.columns:
        return None, None, None
    d = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    night = d[d["Timestamp"].dt.hour.isin([22, 23, 0, 1, 2, 3, 4])]
    if night.shape[0] < 2:
        return None, None, None
    ts = night["Timestamp"].reset_index(drop=True)
    diffs = (ts.shift(-1) - ts).dt.total_seconds() / 60.0
    idx = diffs[diffs >= float(min_gap_min)].index
    if len(idx) == 0:
        return None, None, None
    i = int(idx[0])
    return ts.iloc[i], ts.iloc[i + 1], float(diffs.iloc[i])

def _nocturnal_micro_session_count(df: pd.DataFrame) -> float:
    if df is None or df.empty or "Timestamp" not in df.columns:
        return np.nan
    night = df[df["Timestamp"].dt.hour.isin([1, 2, 3, 4, 5, 6])]
    if night.empty:
        return 0.0
    sess = sessions_from_timestamps(night, gap_sec=300)
    if not sess:
        return 0.0
    minutes = [(b - a).total_seconds() / 60.0 for a, b in sess]
    return float(sum(1 for m in minutes if m <= 5.0))

def _mean_inter_awak_gap_min(df: pd.DataFrame) -> float:
    if df is None or df.empty or "Timestamp" not in df.columns:
        return np.nan
    night = df[df["Timestamp"].dt.hour.isin([1, 2, 3, 4, 5, 6])]
    if night.empty:
        return np.nan
    sess = sessions_from_timestamps(night, gap_sec=300)
    if len(sess) < 2:
        return np.nan
    gaps = [(sess[i + 1][0] - sess[i][1]).total_seconds() / 60.0 for i in range(len(sess) - 1)]
    gaps = [g for g in gaps if g >= 0]
    return float(np.mean(gaps)) if gaps else np.nan

def _daytime_idle_ratio(df: pd.DataFrame) -> Tuple[float, Dict[str, int]]:
    if df is None or df.empty or "Timestamp" not in df.columns:
        return (np.nan, {})
    per_min = df.set_index("Timestamp").assign(cnt=1)["cnt"].resample("1Min").sum()
    day = per_min[per_min.index.hour.isin(range(8, 18))]
    if day.empty:
        return (np.nan, {})
    idle = int((day == 0).sum())
    return (idle / float(len(day)), {"idle_day_minutes": idle, "total_day_minutes": int(len(day))})

def _night_day_ratio_bytes(df: pd.DataFrame) -> Tuple[float, Dict[str, int]]:
    """Night (00–05) vs Day (06–21) ratio using bytes if available, else packet counts."""
    if df is None or df.empty or "Timestamp" not in df.columns:
        return (np.nan, {})
    has_len = "Length" in df.columns
    d = df.copy()
    d["bucket"] = d["Timestamp"].dt.hour.apply(lambda h: "night" if h in [0,1,2,3,4,5] else ("day" if 6 <= h < 22 else "ignore"))
    dn = d[d["bucket"] == "night"]
    dd = d[d["bucket"] == "day"]
    if has_len:
        bn = float(dn["Length"].sum()); bd = float(dd["Length"].sum())
    else:
        bn = float(len(dn));          bd = float(len(dd))
    if bd <= 0 and bn <= 0: return (np.nan, {})
    if bd <= 0:              return (np.inf, {"bytes_night": int(bn), "bytes_day": 0})
    return (bn / bd, {"bytes_night": int(bn), "bytes_day": int(bd)})

def _circular_std_minutes(mins: np.ndarray) -> float | None:
    if mins.size < 3:
        return None
    ang = 2 * np.pi * (mins / 1440.0)
    C = np.cos(ang).sum(); S = np.sin(ang).sum()
    R = np.sqrt(C**2 + S**2) / mins.size
    if R <= 0:
        return None
    std_rad = np.sqrt(-2.0 * np.log(R))
    return float(std_rad * (1440.0 / (2 * np.pi)))


# ----------------------------- definitions -----------------------------------

C4_DEFS: List[MetricDef] = [
    MetricDef(
        id="C4/F1",
        label="Sleep onset delay (from 22:00, min)",
        dist_col="C4_F1_OnsetDelayFrom2200Min",
        ok=120.0, higher_is_worse=True, fmt_tpl="{v:.0f} min",
        latex_formula=r"\Delta t_{\mathrm{onset}} = (t_{\mathrm{onset}} - 22{:}00)_+ \ \text{in minutes}",
        latex_numbers_tpl=r"t_{\mathrm{onset}} \approx {onset_hhmm},\ \Delta t = {value:.0f}\ \mathrm{min}",
        explanation_md=(
            "This metric estimates when you fell asleep by finding the last network activity before a continuous quiet period of at least 30 minutes during the night window (22:00–04:00). We then measure how many minutes that onset is after 22:00; larger values indicate later bedtimes and potential difficulty initiating sleep. The approach uses only timing of connections, not their content, and avoids personal details. Occasional late nights are normal; persistent delays compared with your usual routine are more informative. Shift workers or irregular schedules can produce large values without indicating a sleep disorder. Use this together with sleep duration and fragmentation features for a complete picture."
        ),
        missing_md="No idle gap ≥30 min found between 22:00–04:00 → N/A.",
    ),
    MetricDef(
        id="C4/F2",
        label="Wake time after 04:00 (min)",
        dist_col="C4_F2_WakeAfter0400Min",
        ok=240.0, higher_is_worse=True, fmt_tpl="{v:.0f} min",
        latex_formula=r"\max(0,\ t_{\mathrm{wake}}-04{:}00)\ \text{in minutes}",
        latex_numbers_tpl=r"t_{\mathrm{wake}} \approx {wake_hhmm},\ \Delta t = {value:.0f}\ \mathrm{min}",
        explanation_md=(
            "Wake‑up time is approximated as the first network activity after the night’s quiet period. We report how many minutes this is after 04:00; higher values indicate later waking and may reflect hypersomnia. Very early awakenings (before 04:00) are clipped to zero in this single‑sided metric; consider combining with onset delay to detect shortened sleep. Because this method relies on device activity, unusual settings like scheduled updates can shift the estimate. Trends across several days are more meaningful than a single data point. Interpret alongside sleep duration and daytime idleness to distinguish schedule changes from sleep problems."
        ),
        missing_md="No qualifying night gap detected → N/A.",
    ),
    MetricDef(
        id="C4/F3",
        label="Onset time variability (14 d, circ. SD, min)",
        dist_col="C4_F3_OnsetVar14dMin",
        ok=120.0, higher_is_worse=True, fmt_tpl="{v:.0f} min",
        latex_formula=r"S=\mathrm{circ\_std}\big(\{t_{\mathrm{onset,d}}\}_{d-13..d}\big)",
        latex_numbers_tpl=r"S={value:.0f}\ \mathrm{min}",
        explanation_md=(
            "This captures how consistent your bedtime is over the past two weeks using circular statistics that wrap the 24‑hour clock. Larger values mean bedtimes vary widely from day to day, which is associated with worse sleep quality and higher depressive symptoms. Regularity is a protective factor: smaller variability often accompanies stable routines and better mood. Because the metric needs multiple nights, it is shown as N/A until sufficient history is available. Vacations or shift work naturally increase variability; look for patterns that persist beyond short periods. Use together with sleep duration and fragmentation metrics to understand overall sleep health."
        ),
        missing_md="Needs ≥3 days with C4_F1_OnsetLocalMin in ALL_DAILY.",
    ),
    MetricDef(
        id="C4/F4",
        label="Sleep duration |z| vs 30d baseline",
        dist_col="C4_F4_SleepDurationZAbs30d",
        ok=2.0, higher_is_worse=True, fmt_tpl="{v:.2f}",
        latex_formula=r"|z|=\left|\frac{L_{\mathrm{today}}-\mu_{30}}{\sigma_{30}}\right|",
        latex_numbers_tpl=r"|z|={value:.2f}",
        explanation_md=(
            "We approximate nightly sleep duration as the longest period of continuous network silence of at least 30 minutes within the night. Values above six hours are generally consistent with adequate sleep, while much shorter durations can indicate insomnia or disrupted nights. This proxy has been shown to track actigraphy with reasonable error in passive sensing studies. Keep in mind that silence can also occur when devices are powered off or disconnected; look for repeated patterns rather than one‑off anomalies. Comparing today to your own history is more meaningful than comparing to other people. Combine with onset delay and micro‑session counts for a fuller view of sleep continuity."
        ),
        missing_md="Needs ≥7 historical days with LongestInactivityHours.",
    ),
    MetricDef(
        id="C4/F5",
        label="Nocturnal micro‑sessions (01–06, ≤5 min)",
        dist_col="C4_F5_NocturnalMicroSessionCount0106",
        ok=2.0, higher_is_worse=True, fmt_tpl="{v:.0f}",
        latex_formula=r"\#\{\,\text{sessions in }01\text{–}06 \le 5\,\mathrm{min}\,\}",
        latex_numbers_tpl=r"{value_int}",
        explanation_md=(
            "This metric shows how unusual tonight’s sleep duration is compared with your own last‑30‑day history. We take the absolute z‑score so both unusually short and unusually long nights raise the value. Large deviations can signal either insomnia or hypersomnia depending on the direction, and they are more informative if they persist. Because it is personalized, the same number can reflect different absolute durations for different people. A stable baseline makes the estimate more reliable; with very few historical nights the value is N/A. Use the time‑series in the details dialog to see how tonight compares to prior weeks."
        ),
        missing_md="Not enough traffic to segment nighttime sessions → N/A.",
    ),
    MetricDef(
        id="C4/F6",
        label="Mean gap between awakenings (01–06, min)",
        dist_col="C4_F6_MeanInterAwakGapMin0106",
        ok=30.0, higher_is_worse=False, fmt_tpl="{v:.0f} min",
        latex_formula=r"\overline{\Delta}=\frac{1}{K-1}\sum_{i=1}^{K-1}(s_{i+1}^{\text{start}}-s_{i}^{\text{end}})",
        latex_numbers_tpl=r"\overline{\Delta}={value:.0f}\ \mathrm{min}",
        explanation_md=(
            "We count short bursts of network activity during the core sleep window (01–06 h). Multiple micro‑sessions suggest night‑time awakenings or phone use in bed, both linked to fragmented sleep. Occasional brief checks are common; sustained patterns are more meaningful for interpretation. This feature uses only timing and duration of activity, not content. Very sparse traffic or airplane mode can hide awakenings; interpret alongside the main sleep gap length. The threshold can be adjusted in the details dialog to match your routine."
        ),
        missing_md="Need ≥2 nighttime sessions (01–06).",
    ),
    MetricDef(
        id="C4/F7",
        label="Daytime idle ratio (08–18)",
        dist_col="C4_F7_DaytimeIdleRatio0818",
        ok=0.50, higher_is_worse=True, fmt_tpl="{v:.2f}",
        latex_formula=r"\mathrm{IdleRatio}=\frac{\text{idle minutes}}{\text{total minutes in }08\text{–}18}",
        latex_numbers_tpl=r"\frac{{idle_day_minutes}}{{total_day_minutes}}={value:.2f}",
        explanation_md=(
            "This is the average time between the end of one micro‑session and the start of the next during 01–06 h. Longer gaps mean awakenings are farther apart, while short gaps indicate denser fragmentation. The measure is N/A if there are fewer than two micro‑sessions in the window. Because phones sometimes fetch updates automatically, cross‑checking with the micro‑session count helps avoid false alarms. Look for repeated nights with short gaps; one noisy night can be misleading. The OK threshold can be tuned if you usually wake earlier or later than typical."
        ),
        missing_md="No minute‑level activity bins available → N/A.",
    ),
    MetricDef(
        id="C4/F8",
        label="Night/Day traffic ratio (bytes)",
        dist_col="C4_F8_NightDayTrafficRatioBytes",
        ok=0.40, higher_is_worse=True, fmt_tpl="{v:.2f}",
        latex_formula=r"R=\frac{B_{\text{00–05}}}{B_{\text{06–21}}}",
        latex_numbers_tpl=r"\frac{{bytes_night}}{{bytes_day}}={value:.2f}",
        explanation_md=(
            "This shows how much of the daytime window (08–18 h) has no network activity at all. Very high idleness can reflect daytime napping or extremely low device engagement, which may accompany hypersomnia or low energy. Healthy patterns differ widely; working offline or being outdoors can also increase idleness in a harmless way. Comparisons against your own baseline are more informative than absolute thresholds. Combine with night metrics to distinguish oversleep from focused offline work. You can adjust the threshold if your schedule differs from a typical 08–18 pattern."
        ),
        missing_md="No timestamps or byte lengths available → N/A.",
    ),
]

# ----------------------------- main class ------------------------------------

class Criterion4:
    """Insomnia / Hypersomnia — C4 (self-contained)."""

    def _compute_today_fields(self, df_day: pd.DataFrame) -> Dict[str, Any]:
        rec: Dict[str, Any] = {}
        onset, wake, gap_m = _find_night_idle_gap(df_day, min_gap_min=30)
        rec["__onset_ts"] = onset
        rec["__wake_ts"] = wake

        # F1: onset delay from 22:00 + store local time‑of‑day (for variability metric)
        if onset is not None:
            rec["C4_F1_OnsetDelayFrom2200Min"] = _minutes_from_2200(onset)
            rec["C4_F1_OnsetLocalMin"] = float(onset.hour * 60 + onset.minute + onset.second / 60.0)
        else:
            rec["C4_F1_OnsetDelayFrom2200Min"] = np.nan
            rec["C4_F1_OnsetLocalMin"] = np.nan

        # F2: wake after 04:00 (clamped at 0)
        if wake is not None:
            anchor = wake.normalize() + pd.Timedelta(hours=4)
            delta = (wake - anchor).total_seconds() / 60.0
            rec["C4_F2_WakeAfter0400Min"] = float(delta) if delta > 0 else 0.0
        else:
            rec["C4_F2_WakeAfter0400Min"] = np.nan

        # F5: nocturnal micro‑sessions (01–06)
        rec["C4_F5_NocturnalMicroSessionCount0106"] = _nocturnal_micro_session_count(df_day)

        # F6: mean inter‑awakening gap (01–06)
        rec["C4_F6_MeanInterAwakGapMin0106"] = _mean_inter_awak_gap_min(df_day)

        # F7: daytime idle ratio (08–18)
        val, ex = _daytime_idle_ratio(df_day)
        rec["C4_F7_DaytimeIdleRatio0818"] = val
        rec["_EXTRAS_F7"] = ex

        # F8: night/day traffic ratio (bytes or counts)
        val_nd, ex_nd = _night_day_ratio_bytes(df_day)
        rec["C4_F8_NightDayTrafficRatioBytes"] = val_nd
        rec["_EXTRAS_F8"] = ex_nd

        return rec

    def _compute_hist_fields(self, today_fields: Dict[str, Any], today: dict, ALL_DAILY: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # F3: Onset variability (14d) from ALL_DAILY["C4_F1_OnsetLocalMin"]
        if ALL_DAILY is not None and not ALL_DAILY.empty and "C4_F1_OnsetLocalMin" in ALL_DAILY.columns:
            hist = pd.to_numeric(ALL_DAILY["C4_F1_OnsetLocalMin"], errors="coerce").dropna().tail(14).to_numpy(dtype=float)
            out["C4_F3_OnsetVar14dMin"] = float(_circular_std_minutes(hist)) if hist.size >= 3 else np.nan
        else:
            out["C4_F3_OnsetVar14dMin"] = np.nan

        # F4: |z| of LongestInactivityHours vs last 30 days
        L_today = (today or {}).get("LongestInactivityHours")
        if L_today is None and "LongestInactivityHours" in today_fields:
            L_today = today_fields["LongestInactivityHours"]
        if ALL_DAILY is None or ALL_DAILY.empty or "LongestInactivityHours" not in ALL_DAILY.columns:
            out["C4_F4_SleepDurationZAbs30d"] = np.nan
        else:
            hist = pd.to_numeric(ALL_DAILY["LongestInactivityHours"], errors="coerce").dropna().tail(30).to_numpy(dtype=float)
            if hist.size >= 7 and np.isfinite(L_today):
                mu = float(np.mean(hist)); sd = float(np.std(hist, ddof=1))
                out["C4_F4_SleepDurationZAbs30d"] = (abs((float(L_today) - mu)/sd) if (np.isfinite(sd) and sd > 0) else np.nan)
            else:
                out["C4_F4_SleepDurationZAbs30d"] = np.nan
        return out

    def compute(
        self,
        df_day: pd.DataFrame,
        today: dict,         # daily base record (from base_features)
        aux_ctx: dict,       # not required here
        ALL_DAILY: pd.DataFrame
    ) -> List[Dict[str, Any]]:

        today = dict(today or {})
        have_all = all(
            (m.dist_col in today) and (not pd.isna(today.get(m.dist_col))) for m in C4_DEFS
        )
        if have_all:
            today_fields = today
        else:
            today_fields = self._compute_today_fields(df_day)
            hist_fields = self._compute_hist_fields(today_fields, today, ALL_DAILY)
            today_fields.update(hist_fields)

        metrics_out: List[Dict[str, Any]] = []

        for mdef in C4_DEFS:
            value = today_fields.get(mdef.dist_col, np.nan)

            # Status
            if (value is None) or (isinstance(value, float) and (np.isnan(value) or not np.isfinite(value))) or (mdef.ok is None):
                status_tuple = ("N/A", "blue")
                ranges_str = "" if mdef.ok is None else thresholds_text(mdef.ok, higher_is_worse=mdef.higher_is_worse)
            else:
                v = float(value)
                is_ok = (v <= mdef.ok) if mdef.higher_is_worse else (v >= mdef.ok)
                status_tuple = ("OK", "green") if is_ok else ("Caution", "orange")
                ranges_str   = thresholds_text(mdef.ok, higher_is_worse=mdef.higher_is_worse)

            # Formatter + LaTeX numbers (build extras context)
            fmt = fmt_from_template(mdef.fmt_tpl)
            extras: Dict[str, Any] = {}
            extras = update_extras_with_value(extras, value)

            # Provide nice HH:MM where useful
            onset_ts = today_fields.get("__onset_ts")
            wake_ts  = today_fields.get("__wake_ts")
            if onset_ts is not None:
                extras.setdefault("onset_hhmm", _hhmm(onset_ts))
            if wake_ts is not None:
                extras.setdefault("wake_hhmm", _hhmm(wake_ts))

            # Pass counters for ratios if present
            if mdef.id == "C4/F7":
                ex = today_fields.get("_EXTRAS_F7", {})
                extras.update(ex)
            if mdef.id == "C4/F8":
                ex = today_fields.get("_EXTRAS_F8", {})
                extras.update({"bytes_night": ex.get("bytes_night"), "bytes_day": ex.get("bytes_day")})

            latex_numbers = None
            if isinstance(mdef.latex_numbers_tpl, str) and mdef.latex_numbers_tpl.strip():
                latex_numbers = latex_fill_from_template(mdef.latex_numbers_tpl, extras)

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
