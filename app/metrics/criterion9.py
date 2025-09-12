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


C9_DEFS: List[MetricDef] = [
    MetricDef(
        id="C9/F1",
        label="Crisis‑line domain hits",
        dist_col="C9_F1_CrisisLineHits",
        value_ref="FUNC:CrisisLineHits",
        ok=0.0, higher_is_worse=True, fmt_tpl="{v:.0f}",
        latex_formula=r"N=\sum \mathbf{1}[\text{eTLD+1}\in\mathcal{H}_{crisis}]",
        latex_numbers_tpl=r"N={value:.0f}",
        explanation_md=(
            "Counts visits to known crisis-line websites or APIs. "
            "Such visits often occur when someone is exploring immediate help options. "
            "A single hit can come from an article or shared link, but repeated hits in a short time deserve attention. "
            "Always follow up with empathy rather than alarm."
        ),
    ),
    MetricDef(
        id="C9/F2",
        label="Suicide‑method query ratio",
        dist_col="C9_F2_SuicideMethodQueryRatio",
        value_ref="FUNC:SuicideMethodQueryRatio",
        ok=0.005, higher_is_worse=True, fmt_tpl="{v:.3f}",
        latex_formula=r"R=\frac{\#\text{method queries}}{\#\text{queries}}",
        latex_numbers_tpl=r"R={value:.4f}",
        explanation_md=(
            "Looks for search queries that mention specific self-harm methods and divides them by all queries for the day. "
            "Even a small rise can be meaningful because most people never search these terms. "
            "Context matters—some searches may stem from news or academic interest. "
            "Persistent elevation is a strong warning sign."
        ),
    ),
    MetricDef(
        id="C9/F3",
        label="Therapy‑booking page visits",
        dist_col="C9_F3_TherapyBookingVisits",
        value_ref="FUNC:C9_TherapyBookingVisits",
        ok=0.0, higher_is_worse=True, fmt_tpl="{v:.0f}",
        latex_formula=r"N=\sum \mathbf{1}[\text{path}\in\{/find\_therapist, /book\_psy\}]",
        latex_numbers_tpl=r"N={value:.0f}",
        explanation_md=(
            "Counts visits to pages where therapy sessions can be booked or therapists located. "
            "These hits may indicate active help-seeking behavior. "
            "Occasional visits can also happen for family or coursework. "
            "Combine with other metrics to understand intent."
        ),
    ),
    MetricDef(
        id="C9/F4",
        label="Self‑harm forum visits",
        dist_col="C9_F4_SelfHarmForumVisits",
        value_ref="FUNC:SelfHarmForumVisits",
        ok=0.0, higher_is_worse=True, fmt_tpl="{v:.0f}",
        latex_formula=r"N=\sum \mathbf{1}[\text{domain}\in\mathcal{H}_{selfharm}]",
        latex_numbers_tpl=r"N={value:.0f}",
        explanation_md=(
            "Measures visits to online communities that discuss self-harm. "
            "Regular access to these spaces can reinforce harmful ideas or, in some cases, seek support. "
            "A sudden increase after a quiet period is especially noteworthy. "
            "Verify context with the individual when possible."
        ),
    ),
    MetricDef(
        id="C9/F5",
        label="Upstream bytes to self‑harm forums (MB)",
        dist_col="C9_F5_SelfHarmForumUpBytes",
        value_ref="FUNC:SelfHarmForumUpBytes",
        ok=0.0, higher_is_worse=True, fmt_tpl="{v:.0f} B",
        latex_formula=r"B_{\uparrow}=\sum \text{bytes}_{\uparrow}(\mathcal{H}_{selfharm})/10^6",
        latex_numbers_tpl=r"B_{\uparrow}={value:.1f}\,\text{MB}",
        explanation_md=(
            "Totals data uploaded to self-harm forums, which can signal posting messages or sharing files. "
            "Higher volumes imply active participation rather than passive reading. "
            "Automatic media uploads can inflate numbers, so compare with visit counts. "
            "Large spikes should prompt a check for accompanying text or images."
        ),
    ),
    MetricDef(
        id="C9/F6",
        label="Avg. session length on self‑harm forums (min)",
        dist_col="C9_F6_SelfHarmForumMeanSessLenSec",
        value_ref="FUNC:SelfHarmForumMeanSessionLenSec",
        ok=3.0, higher_is_worse=True, fmt_tpl="{v:.0f} s",
        latex_formula=r"\bar{L}=\frac{1}{K}\sum L_k/60",
        latex_numbers_tpl=r"\bar{L}={value:.1f}\,\text{min}",
        explanation_md=(
            "Averages how long each visit to self-harm forums lasts. "
            "Long sessions may reflect deep engagement or searching for guidance, while short ones could be quick checks. "
            "Use together with upload volume and visit counts for context. "
            "Sudden increases warrant attention."
        ),
    ),
    MetricDef(
        id="C9/F7",
        label="Will / insurance doc downloads",
        dist_col="C9_F7_WillInsuranceDownloads",
        value_ref="FUNC:WillInsuranceDownloads",
        ok=0.0, higher_is_worse=True, fmt_tpl="{v:.0f}",
        latex_formula=r"N=\#\{\text{PDF downloads from estate/insurance portals}\}",
        latex_numbers_tpl=r"N={value:.0f}",
        explanation_md=(
            "Counts downloads of documents related to wills or life insurance. "
            "Many people update these papers routinely, but unexpected activity can signal preparation for self-harm. "
            "Look for clustering with other concerning metrics. "
            "Always ask about legitimate reasons before drawing conclusions."
        ),
    ),
    MetricDef(
        id="C9/F8",
        label="Cloud‑backup surge (MB vs baseline)",
        dist_col="C9_F8_CloudBackupUpBytesToday",
        value_ref="FUNC:CloudBackupUpBytesToday",
        ok=200.0, higher_is_worse=True, fmt_tpl="{v:.0f} B",
        latex_formula=r"\Delta B_{\uparrow}^{cloud}=B_{\uparrow}^{today}-\text{median}_{28d}(B_{\uparrow})",
        latex_numbers_tpl=r"\Delta={value:.1f}\,\text{MB}",
        explanation_md=(
            "Compares today’s uploads to cloud-backup services with the 28-day median. "
            "A big jump may indicate someone is rapidly saving files, possibly putting affairs in order. "
            "Small changes are normal due to automatic syncs. "
            "Check known device backups to avoid false alarms."
        ),
    ),
    MetricDef(
        id="C9/F9",
        label="Account‑deletion requests",
        dist_col="C9_F9_AccountDeletionRequestsCount",
        value_ref="FUNC:AccountDeletionRequestsCount",
        ok=0.0, higher_is_worse=True, fmt_tpl="{v:.0f}",
        latex_formula=r"N=\#\{\text{POST/DELETE to }/delete\_account\}",
        latex_numbers_tpl=r"N={value:.0f}",
        explanation_md=(
            "Tallies requests sent to delete accounts across services. "
            "Clearing accounts can be a normal privacy step, but a flurry of deletions may suggest disengagement or finality. "
            "Some platforms issue background delete calls, so review which services are affected. "
            "Use in combination with other signs of withdrawal."
        ),
    ),
    MetricDef(
        id="C9/F10",
        label="Night‑time suicide‑query bursts",
        dist_col="C9_F10_NightSuicideQueryBursts",
        value_ref="FUNC:NightSuicideQueryBursts",
        ok=0.0, higher_is_worse=True, fmt_tpl="{v:.0f}",
        latex_formula=r"N=\#\{\text{SI keyword clusters in 00–05h}\}",
        latex_numbers_tpl=r"N={value:.0f}",
        explanation_md=(
            "Counts clusters of suicide-related searches made between midnight and 05:00. "
            "Night-time rumination is strongly associated with distress and impulsivity. "
            "Occasional late-night searches about other topics do not affect this metric. "
            "Multiple bursts across nights should prompt careful follow-up."
        ),
    ),
    MetricDef(
        id="C9/F11",
        label="Night‑time negative‑search ratio",
        dist_col="C9_F11_NightNegativeSearchRatio",
        value_ref="FUNC:NightNegativeSearchRatio",
        ok=0.05, higher_is_worse=True, fmt_tpl="{v:.3f}",
        latex_formula=r"R=\frac{\#\text{negative/SI queries}_{00-05}}{\#\text{queries}_{00-05}}",
        latex_numbers_tpl=r"R={value:.3f}",
        explanation_md=(
            "Calculates the share of negative or self-injury related queries among all searches made at night. "
            "A high ratio shows most late-night searching centers on troubling topics. "
            "Lower ratios imply more neutral night-time activity. "
            "Track over several nights to see if patterns persist."
        ),
    ),
]


class Criterion9:
    """Suicidality — C9."""

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
        for mdef in C9_DEFS:
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

