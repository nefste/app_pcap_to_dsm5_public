from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import make_metric, thresholds_text
from .common import (
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


C8_DEFS: List[MetricDef] = [
    MetricDef(
        id="C8/F1",
        label="Median page‑dwell (s)",
        dist_col="C8_F1_MedianPageDwellSec",
        value_ref="FUNC:C8_PageDwellMedianSec",
        ok=10.0, higher_is_worse=False, fmt_tpl="{v:.1f} s",
        latex_formula=r"\tilde{t}_{dwell}=\operatorname{median}(\Delta t_{\text{page}})",
        latex_numbers_tpl=r"\tilde{t}_{dwell}={value:.1f}\,\text{s}",
        explanation_md=(
            "Represents the typical time spent on a page before moving to the next within a session. "
            "Longer dwell times imply sustained attention, while very short times suggest rapid switching. "
            "Because it uses the median, a few long reads do not overly influence the value. "
            "Compare with the person’s usual level to judge concentration."
        ),
    ),
    MetricDef(
        id="C8/F2",
        label="DNS burst‑rate / hour",
        dist_col="C8_F2_DNSBurstRatePerHour",
        value_ref="FUNC:DNSBurstRatePerHour",
        ok=4.0, higher_is_worse=True, fmt_tpl="{v:.2f} / h",
        latex_formula=r"\lambda=\frac{\#\text{bursts}}{\text{active hours}}",
        latex_numbers_tpl=r"\lambda={value:.1f}\,/h",
        explanation_md=(
            "Counts bursts where several different websites are contacted within a minute. "
            "Frequent bursts show quick hopping between topics, often seen when attention is scattered. "
            "A low rate indicates steadier browsing. "
            "Short spikes can occur during research and should be considered in context."
        ),
    ),
    MetricDef(
        id="C8/F3",
        label="Notification micro‑sessions / day",
        dist_col="C8_F3_NotificationMicroSessionsCount",
        value_ref="FUNC:NotificationMicroSessionsCount",
        ok=2.0, higher_is_worse=True, fmt_tpl="{v:.0f}",
        latex_formula=r"N=\#\{\text{push burst + 1 quick fetch }(<30s)\}",
        latex_numbers_tpl=r"N={value:.0f}",
        explanation_md=(
            "Measures short two-step interactions triggered by a push notification followed by a quick check. "
            "Many of these micro-sessions mean the person keeps interrupting their activity to glance at the device. "
            "A few per day are normal, but high counts point to constant distraction. "
            "Combine with page-dwell metrics for a fuller picture of focus."
        ),
    ),
    MetricDef(
        id="C8/F4",
        label="Repeated‑query ratio",
        dist_col="C8_F4_RepeatedQueryRatio60m",
        value_ref="FUNC:RepeatedQueryRatio60m",
        ok=0.15, higher_is_worse=True, fmt_tpl="{v:.3f}",
        latex_formula=r"R=\frac{\#\text{identical queries (≤60 min)}}{\#\text{queries}}",
        latex_numbers_tpl=r"R={value:.2f}",
        explanation_md=(
            "Looks for identical DNS or SNI queries repeated within an hour. "
            "A high ratio can mean someone forgets they already looked something up or repeatedly checks the same site. "
            "Low ratios reflect more decisive browsing. "
            "Temporary increases may happen when a site fails to load and the user retries."
        ),
    ),
    MetricDef(
        id="C8/F5",
        label="Query‑reformulation chain length",
        dist_col="C8_F5_QueryReformulationMax",
        value_ref="FUNC:C8_QueryReformulationMax",
        ok=3.0, higher_is_worse=True, fmt_tpl="{v:.0f}",
        latex_formula=r"L=\max \text{edit-distance ≤3 chain before click}",
        latex_numbers_tpl=r"L={value:.0f}",
        explanation_md=(
            "Tracks the longest run of small edits to a search query before finally clicking a result. "
            "Long chains suggest indecision or difficulty finding the right words. "
            "Short chains indicate clearer goals or quick satisfaction. "
            "Complex research tasks naturally produce longer chains, so consider context."
        ),
    ),
    MetricDef(
        id="C8/F6",
        label="Back‑navigation percentage",
        dist_col="C8_F6_BackNavShare",
        value_ref="FUNC:C8_BackNavShare",
        ok=0.25, higher_is_worse=True, fmt_tpl="{v:.2f}",
        latex_formula=r"S=\frac{\#\text{HTTP 302/304 or identical GETs within 30s}}{\#\text{GETs}}",
        latex_numbers_tpl=r"S={value:.2f}",
        explanation_md=(
            "Calculates how often the user quickly goes back to a previous page. "
            "High shares reflect back-and-forth scanning or uncertainty about which link to choose. "
            "Lower shares show more linear progression through pages. "
            "Peaks can also arise from technical errors like broken links."
        ),
    ),
    MetricDef(
        id="C8/F7",
        label="SERP time‑to‑first‑click (s)",
        dist_col="C8_F7_SERPTimeToFirstClickSec",
        value_ref="FUNC:C8_SERPTimeToFirstClickSec",
        ok=30.0, higher_is_worse=True, fmt_tpl="{v:.0f} s",
        latex_formula=r"\Delta t = t_{\text{first link}}-t_{\text{SERP}}",
        latex_numbers_tpl=r"\Delta t={value:.0f}\,\text{s}",
        explanation_md=(
            "Measures the delay between loading a search results page and clicking the first link. "
            "Long delays can mean the person is unsure which result to follow. "
            "Short delays show quick decisions. "
            "This metric is most meaningful when a day contains several searches."
        ),
    ),
    MetricDef(
        id="C8/F8",
        label="Median inter‑keystroke gap (s)",
        dist_col="C8_F8_MedianIKSsec",
        value_ref="FUNC:MedianIKSsec",
        ok=0.4, higher_is_worse=True, fmt_tpl="{v:.2f} s",
        latex_formula=r"\tilde{\Delta t}_{key}=\operatorname{median}(\Delta t_{key})",
        latex_numbers_tpl=r"\tilde{\Delta t}_{key}={value:.2f}\,\text{s}",
        explanation_md=(
            "Captures the typical pause between keystrokes while typing. "
            "Longer gaps suggest slower thought or distractions, whereas short gaps indicate fluent typing. "
            "The median makes the metric robust to a few long pauses. "
            "Use alongside other attention metrics for context."
        ),
    ),
]


class Criterion8:
    """Difficulty concentrating / indecisiveness — C8."""

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

        out: List[Dict[str, Any]] = []
        for mdef in C8_DEFS:
            val = today.get(mdef.dist_col)
            extras: Dict[str, Any] = {}
            if (val is None) or (isinstance(val, float) and np.isnan(val)):
                val, extras = resolve_value(mdef.value_ref, df_day, today, aux, today)
                today.setdefault(mdef.dist_col, val)
            elif df_day is not None and not df_day.empty:
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
                    range_cfg={"ok": mdef.ok, "higher_is_worse": mdef.higher_is_worse}
                    if mdef.ok is not None
                    else None,
                )
            )

        return out

