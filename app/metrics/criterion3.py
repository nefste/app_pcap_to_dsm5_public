from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import make_metric, thresholds_text
from .common import (
    resolve_value,             # unified AUX:/TODAY:/TR:/CONST:/FUNC: resolver
    fmt_from_template,         # "{v:.2f}" -> callable
    latex_fill_from_template,  # robust templating with safe missing keys
    update_extras_with_value,  # inject {value}, {value_int} into LaTeX ctx
)

# ----------------------------- metric schema ---------------------------------

@dataclass(frozen=True)
class MetricDef:
    id: str
    label: str
    dist_col: str
    value_ref: str                  # e.g., "FUNC:FoodDeliveryHitsDay"
    ok: Optional[float]             # None -> no status
    higher_is_worse: bool
    fmt_tpl: str
    latex_formula: Optional[str] = None
    latex_numbers_tpl: Optional[str] = None
    explanation_md: Optional[str] = None
    missing_md: Optional[str] = None


# ----------------------------- C3 definitions ---------------------------------

C3_DEFS: List[MetricDef] = [
    # C3/F1 — Food‑delivery domain hits (daily)
    MetricDef(
        id="C3/F1",
        label="Food‑delivery domain hits (daily)",
        dist_col="C3_F1_FoodDeliveryHits",
        value_ref="FUNC:FoodDeliveryHitsDay",
        ok=1.0, higher_is_worse=True,
        fmt_tpl="{v:.0f}",
        latex_formula=r"\mathrm{FDH}=\sum_i \mathbf{1}[\mathrm{SLD}_i\in \mathcal{P}_{\text{food}}]",
        latex_numbers_tpl=r"{value:.0f}",
        explanation_md=(
            "This metric counts how many requests in a day target well‑known food‑delivery services "
            "(for example, domains such as 'ubereats.*' or 'deliveroo.*'). A rising daily count can "
            "indicate increased reliance on take‑away food, which often accompanies appetite change or "
            "reduced motivation to cook. Because it is a simple event count, it is not biased by "
            "connection speeds or video traffic and it avoids reading any sensitive content. "
            "Short‑term spikes can be caused by parties or travel; trends are most informative when they "
            "persist across several days. Interpreting this feature together with late‑night timing and "
            "inter‑order intervals gives a more complete picture. The dashboard compares your daily values "
            "to your own historical distribution rather than a universal norm."
        ),
        missing_md="Requires SLD extraction; if SLD is missing, the metric is N/A.",
    ),

    # C3/F2 — Late‑night delivery ratio (22:00–06:00)
    MetricDef(
        id="C3/F2",
        label="Late‑night delivery ratio (22:00–06:00)",
        dist_col="C3_F2_LateNightDeliveryRatio",
        value_ref="FUNC:LateNightDeliveryRatio",
        ok=0.10, higher_is_worse=True,
        fmt_tpl="{v:.2f}",
        latex_formula=r"R_{\mathrm{night}}=\frac{\text{orders}_{22{:}00\text{–}06{:}00}}{\text{orders}_{24\text{h}}}",
        latex_numbers_tpl=r"\frac{{{orders_night}}}{{{orders_total}}}={value:.2f}",
        explanation_md=(
            "This ratio looks at when food orders happen and flags a shift towards late‑night eating. "
            "Night‑time ordering can reflect disrupted circadian rhythms or comfort‑eating patterns that "
            "correlate with weight gain. The denominator is the total number of food‑delivery hits for "
            "the day, so regular daytime ordering keeps the ratio low. An isolated late event is not "
            "necessarily meaningful; repeated nights with a high ratio are more relevant. Combine this "
            "with sleep‑related features to differentiate genuine schedule changes from occasional late dinners. "
            "The metric is robust to different vendors because it uses a curated set of delivery domains "
            "rather than specific APIs."
        ),
        missing_md="If there are no delivery hits on a day, the ratio is undefined and shown as N/A.",
    ),

    # C3/F3 — Mean inter‑order interval (days, within day)
    MetricDef(
        id="C3/F3",
        label="Mean inter‑order interval (days, within day)",
        dist_col="C3_F3_MeanInterOrderDays",
        value_ref="FUNC:MeanInterOrderDaysDay",
        ok=3.0, higher_is_worse=False,
        fmt_tpl="{v:.2f} d",
        latex_formula=r"\overline{\Delta t}_{\mathrm{orders}}=\frac{1}{K-1}\sum_{j=1}^{K-1}\frac{t_{j+1}-t_j}{86400}\ \text{days}",
        latex_numbers_tpl=r"{value:.2f}\ \mathrm{d}",
        explanation_md=(
            "This measures the average time gap between successive food‑delivery events within the same day. "
            "Shorter gaps indicate multiple orders clustered together, which can signal elevated appetite or "
            "low energy for meal preparation. Using a day‑level window keeps the calculation simple and "
            "privacy‑preserving while still detecting unusual bursts. When tracked over many days, the "
            "distribution shows whether your typical spacing is shrinking or stable. Because a single order "
            "yields no interval, days with one or zero orders will be N/A here. For multi‑day trends, consult "
            "the time‑series view, which reflects how today compares to your baseline."
        ),
        missing_md="Needs at least two delivery events in a day; otherwise N/A.",
    ),

    # C3/F4 — Diet / nutrition site visits
    MetricDef(
        id="C3/F4",
        label="Diet / nutrition site visits",
        dist_col="C3_F4_DietSiteVisits",
        value_ref="FUNC:DietSiteVisitsDay",
        ok=1.0, higher_is_worse=False,
        fmt_tpl="{v:.0f}",
        latex_formula=r"\mathrm{DietHits}=\sum_i \mathbf{1}[\mathrm{SLD}_i\in \mathcal{P}_{\text{diet}}]",
        latex_numbers_tpl=r"{value:.0f}",
        explanation_md=(
            "This counts visits to well‑known diet, calorie, or nutrition services (for example "
            "'myfitnesspal.com' or 'weightwatchers.com'). An increase can reflect active self‑management "
            "after noticing appetite or weight change. Because it is a simple count of destinations, "
            "the metric avoids content inspection and remains privacy‑respectful. Occasional visits are "
            "normal; look for sustained changes relative to your baseline. In isolation, this measure does "
            "not decide whether weight is going up or down—it signals attention to nutrition. Combined with "
            "delivery patterns, it helps distinguish healthy adjustments from passive habits."
        ),
        missing_md="Requires SLD extraction; without SLD the metric is N/A.",
    ),

    # C3/F5 — Calorie‑tracker API bursts
    MetricDef(
        id="C3/F5",
        label="Calorie‑tracker API bursts",
        dist_col="C3_F5_TrackerBurstCount",
        value_ref="FUNC:CalorieTrackerBurstCount",
        ok=2.0, higher_is_worse=True,
        fmt_tpl="{v:.0f}",
        latex_formula=r"B=\sum_{w}\mathbf{1}\Big[\mathrm{count}_{w}\ge \tau\Big],\ \ \tau\ \text{(burst threshold)}",
        latex_numbers_tpl=r"{value:.0f}",
        explanation_md=(
            "This feature detects concentrated activity windows to popular tracker services (e.g., repeated "
            "syncs or log submissions). Bursts are counted in short bins when the number of tracker requests "
            "exceeds a threshold, capturing periods of focused self‑tracking. A sudden rise in bursts can "
            "accompany changes in appetite or weight as people start logging more carefully. Because only "
            "timing and destinations are used, the metric reveals behavior without exposing personal entries. "
            "Device re‑installs and firmware updates can also create bursts; check whether the pattern persists "
            "across days. Interpret together with smart‑scale usage and delivery patterns for context."
        ),
        missing_md="Needs Timestamp and SLD; if missing, or if traffic is too sparse to form a window, the value is N/A.",
    ),

    # C3/F6 — Smart‑scale upload events
    MetricDef(
        id="C3/F6",
        label="Smart‑scale upload events",
        dist_col="C3_F6_SmartScaleUploads",
        value_ref="FUNC:SmartScaleUploadEvents",
        ok=1.0, higher_is_worse=False,
        fmt_tpl="{v:.0f}",
        latex_formula=r"\mathrm{ScaleUploads}=\#\{\text{outbound sessions to smart‑scale vendors}\}",
        latex_numbers_tpl=r"{value:.0f}",
        explanation_md=(
            "This metric counts how many times your device appears to upload weight data to smart‑scale providers "
            "(e.g., Garmin, Withings). More uploads typically indicate frequent weigh‑ins, which often happen when "
            "weight is changing or being tracked more closely. We infer events by grouping connections to vendor "
            "domains into sessions, without reading any payload. Some apps may sync old measurements in bulk; that "
            "looks like a single long session rather than many uploads. As with all metrics, short one‑day variations "
            "matter less than repeated patterns over several days. If you do not own a smart scale or keep it offline, "
            "this metric will remain N/A or zero."
        ),
        missing_md="Requires SLD and IP direction to identify outbound sessions; missing columns → N/A.",
    ),

    # C3/F7 — Weigh‑in time variability (minutes)
    MetricDef(
        id="C3/F7",
        label="Weigh‑in time variability (minutes)",
        dist_col="C3_F7_WeighInTimeVarMin",
        value_ref="FUNC:WeighInTimeVarMinDay",
        ok=120.0, higher_is_worse=True,
        fmt_tpl="{v:.0f} min",
        latex_formula=r"S=\mathrm{circ\_std}(\{\text{weigh‑in times}\})\ \text{in minutes}",
        latex_numbers_tpl=r"{value:.0f}\ \mathrm{min}\ (\text{events}={n_events})",
        explanation_md=(
            "This estimates how consistent your weigh‑in time is by computing the circular standard deviation "
            "of weigh‑in timestamps. Higher values mean weigh‑ins happen at very different times of day, suggesting "
            "a more irregular routine. Rising variability may occur when weight changes quickly or when daily structure "
            "becomes disrupted. Circular statistics avoid artifacts at midnight by wrapping the 24‑hour clock. The "
            "metric needs several events to be meaningful; it will be N/A on days with very few uploads. Viewed across "
            "weeks, it shows whether your weighing routine stabilizes or becomes erratic."
        ),
        missing_md="Needs multiple smart‑scale events to estimate variability; with fewer than 3 events the value is N/A.",
    ),
]


# ------------------------------ internals -------------------------------------

def _safe_resolve(value_ref: str,
                  df_day: pd.DataFrame,
                  today: dict,
                  aux_ctx: dict,
                  today_row: dict) -> Tuple[Any, Dict[str, Any]]:
    """
    Defensive wrapper around resolve_value that *always* returns (value, extras).
    """
    try:
        out = resolve_value(value_ref=value_ref, df_day=df_day,
                            today=today, aux_ctx=aux_ctx, today_row=today_row)
        if isinstance(out, tuple) and len(out) == 2:
            val, extras = out
            return val, (extras or {})
        return out, {}
    except Exception:
        return np.nan, {}


# ------------------------------ main class ------------------------------------

class Criterion3:
    """Appetite / weight change — C3 (self-contained, no Excel)."""

    def compute(self,
                df_day: pd.DataFrame,
                today: dict,           # today_base from base_features.compute_daily_base_record
                aux_ctx: dict,         # per-day extras (ALL_DAILY, counts, etc.)
                ALL_DAILY: pd.DataFrame) -> List[Dict[str, Any]]:

        # Expose ALL_DAILY for any FUNC resolvers that need baselines (future-proof)
        aux_ctx = dict(aux_ctx or {})
        aux_ctx.setdefault("ALL_DAILY", ALL_DAILY)
        today_row = aux_ctx.get("today_row") or {}

        out: List[Dict[str, Any]] = []

        for m in C3_DEFS:
            pre_val = today_row.get(m.dist_col)
            if pre_val is not None and (not isinstance(pre_val, float) or not np.isnan(pre_val)):
                val, extras = pre_val, {}
            else:
                val, extras = _safe_resolve(m.value_ref, df_day, today, aux_ctx, today_row)

            # 2) status & ranges text
            if (val is None) or (isinstance(val, float) and (np.isnan(val) or not np.isfinite(val))) or (m.ok is None):
                status_tuple = ("N/A", "blue")
                ranges_str = "" if (m.ok is None) else thresholds_text(m.ok, higher_is_worse=m.higher_is_worse)
            else:
                v = float(val)
                is_ok = (v <= m.ok) if m.higher_is_worse else (v >= m.ok)
                status_tuple = ("OK", "green") if is_ok else ("Caution", "orange")
                ranges_str = thresholds_text(m.ok, higher_is_worse=m.higher_is_worse)

            # 3) human formatter and LaTeX numbers
            fmt = fmt_from_template(m.fmt_tpl)

            latex_numbers = None
            if isinstance(m.latex_numbers_tpl, str) and m.latex_numbers_tpl.strip():
                ctx = update_extras_with_value(extras, val)
                latex_numbers = latex_fill_from_template(m.latex_numbers_tpl, ctx)

            # 4) assemble dashboard metric dict (consumed by 02_DSM5_Dashboard)
            out.append(
                make_metric(
                    label=m.label,
                    value=val,
                    fmt=fmt,
                    status_tuple=status_tuple,
                    ranges_str=ranges_str,
                    latex_formula=m.latex_formula,
                    latex_numbers=latex_numbers,
                    heuristic_md=m.explanation_md,
                    missing_md=m.missing_md,
                    dist_col=m.dist_col,
                    range_cfg=({"ok": m.ok, "higher_is_worse": m.higher_is_worse}
                               if m.ok is not None else None),
                )
            )

        return out
