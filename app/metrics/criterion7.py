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


C7_DEFS: List[MetricDef] = [
    MetricDef(
        id="C7/F1",
        label="Mental‑health site visits / day",
        dist_col="C7_F1_MentalHealthSiteVisitsDay",
        value_ref="FUNC:MentalHealthSiteVisitsDay",
        ok=0.0, higher_is_worse=True, fmt_tpl="{v:.0f}",
        latex_formula=r"\text{MHVisits}=\sum\mathbf{1}[\text{eTLD+1}\in\mathcal{H}_{MH}]",
        latex_numbers_tpl=r"\text{MHVisits}={value:.0f}",
        explanation_md=(
            "Counts visits to websites or services tagged as mental-health resources. "
            "A few visits can reflect curiosity or routine check-ins, but repeated hits in one day "
            "may mean the person is actively searching for support or information. "
            "Look for changes from the usual pattern before drawing conclusions."
        ),
    ),
    MetricDef(
        id="C7/F2",
        label="“Negative‑self” search‑query ratio",
        dist_col="C7_F2_NegativeSelfSearchRatio",
        value_ref="FUNC:NegativeSelfSearchRatio",
        ok=0.02, higher_is_worse=True, fmt_tpl="{v:.3f}",
        latex_formula=r"R=\frac{Q^{-}_{self}}{Q_{\text{all}}}",
        latex_numbers_tpl=r"R=\frac{Q^{-}_{self}}{Q_{\text{all}}}={value:.3f}",
        explanation_md=(
            "Looks for search phrases such as *am I worthless* and divides them by all queries. "
            "A higher ratio suggests more of the day’s searching centers on negative self-talk. "
            "Very small ratios are normal because many routine searches contain no emotional content. "
            "Consider context like news events or school projects when interpreting spikes."
        ),
    ),
    MetricDef(
        id="C7/F3",
        label="Time on self‑assessment pages (s)",
        dist_col="C7_F3_SelfAssessmentDwellSec",
        value_ref="FUNC:SettingsPrivacyDwellSec",
        ok=60.0, higher_is_worse=True, fmt_tpl="{v:.0f} s",
        latex_formula=r"T=\sum \text{dur}(\text{/phq9|/dass21|/self\_test})",
        latex_numbers_tpl=r"T={value:.0f}\,\text{s}",
        explanation_md=(
            "Measures how long someone stays on online self-assessment pages such as the PHQ-9 or DASS-21. "
            "Spending more time there can signal worry about one’s mental state or a search for confirmation. "
            "Brief visits may occur from curiosity or accidental clicks. "
            "Use together with other metrics to avoid over-interpreting a single day."
        ),
    ),
    MetricDef(
        id="C7/F4",
        label="Help‑line / therapy look‑ups",
        dist_col="C7_F4_HelpTherapyLookupHits",
        value_ref="FUNC:CrisisLineHits",
        ok=0.0, higher_is_worse=True, fmt_tpl="{v:.0f}",
        latex_formula=r"\text{HelpLookups}=\sum \mathbf{1}[\text{eTLD+1}\in\mathcal{H}_{help}]",
        latex_numbers_tpl=r"\text{HelpLookups}={value:.0f}",
        explanation_md=(
            "Counts visits to crisis hotlines or therapy-finder portals. "
            "Such look-ups often happen when someone is considering professional help. "
            "One or two hits can come from news articles or assignments, so watch for repeated patterns. "
            "A sudden surge may warrant a compassionate check-in."
        ),
    ),
    MetricDef(
        id="C7/F5",
        label="Account‑delete / unsubscribe requests",
        dist_col="C7_F5_AccountDeleteUnsubCount",
        value_ref="FUNC:AccountDeleteUnsubscribeCount",
        ok=0.0, higher_is_worse=True, fmt_tpl="{v:.0f}",
        latex_formula=r"N=\#\{\text{POST/DELETE to }/delete\_account|/unsubscribe\}",
        latex_numbers_tpl=r"N={value:.0f}",
        explanation_md=(
            "Tracks requests sent to delete or unsubscribe from online accounts. "
            "Routine cleanup is normal, but a burst of deletions may reflect withdrawal or feelings of not deserving services. "
            "Some platforms send background requests automatically, so review which sites are involved. "
            "Combine with other signals before inferring intent."
        ),
    ),
    MetricDef(
        id="C7/F6",
        label="Settings / privacy‑page dwell (s)",
        dist_col="C7_F6_SettingsPrivacyDwellSec",
        value_ref="FUNC:SettingsPrivacyDwellSec",
        ok=120.0, higher_is_worse=True, fmt_tpl="{v:.0f} s",
        latex_formula=r"T=\sum \text{dur}(\text{/settings|/privacy})",
        latex_numbers_tpl=r"T={value:.0f}\,\text{s}",
        explanation_md=(
            "Sums the time spent on account settings or privacy pages. "
            "Lingering in these areas can mean concern about what others see or attempts to tidy accounts out of guilt. "
            "Occasional visits after app updates are common. "
            "Persistent high values alongside other withdrawal signs merit attention."
        ),
    ),
    MetricDef(
        id="C7/F7",
        label="Outgoing‑post share (social)",
        dist_col="C7_F7_SocialOutgoingShareUp",
        value_ref="FUNC:SocialOutgoingShareUpstream",
        ok=0.25, higher_is_worse=False, fmt_tpl="{v:.2f}",
        latex_formula=r"S=\frac{B_{\uparrow}}{B_{\uparrow}+B_{\downarrow}}",
        latex_numbers_tpl=r"S={value:.2f}",
        explanation_md=(
            "Compares data sent versus received on social networks. "
            "A low share shows mostly reading and little posting, which can happen when someone feels unworthy to engage. "
            "Higher shares indicate active contribution. "
            "Watch for sudden drops relative to the person’s usual level."
        ),
    ),
    MetricDef(
        id="C7/F8",
        label="Cloud‑upload delta (MB vs baseline)",
        dist_col="C7_F8_CloudUploadBytesToday",
        value_ref="FUNC:CloudUploadBytesToday",
        ok=0.0, higher_is_worse=True, fmt_tpl="{v:.0f} B",
        latex_formula=r"\Delta B_{\uparrow}^{cloud}=B_{\uparrow}^{today}-\text{median}_{28d}(B_{\uparrow})",
        latex_numbers_tpl=r"\Delta={value:.1f}\,\text{MB}",
        explanation_md=(
            "Looks at how much was uploaded to cloud-storage providers compared with the 28-day median. "
            "Large positive deltas suggest a burst of uploading files or photos, possibly a digital clean-up. "
            "Small fluctuations are expected because many apps sync quietly. "
            "Interpret big changes in light of known device backups or project work."
        ),
    ),
]


class Criterion7:
    """Worthlessness / guilt — C7 (self-contained)."""

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
        for mdef in C7_DEFS:
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

