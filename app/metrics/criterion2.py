from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import make_metric, thresholds_text
from .common import (
    SOCIAL_SLDS,
    PRODUCTIVITY_SLDS,
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


C2_DEFS: List[MetricDef] = [
    MetricDef(
        id="C2/F1",
        label="Unique domains (eTLD+1)",
        dist_col="C2_F1_UniqueSLD",
        ok=15.0, higher_is_worse=False, fmt_tpl="{v:.0f}",
        latex_formula=r"\mathrm{UniqueSLD}=|\{\text{SLD}\}|",
        latex_numbers_tpl=r"|\{\text{SLD}\}|={value_int}",
        explanation_md=(
            "This metric counts how many different registrable website roots ('eTLD+1', such as 'google.com') you visited across the day, ignoring subdomains like 'mail.google.com'. A shrinking number over several days can signal less exploration and a narrower set of online activities, which is consistent with reduced interest or pleasure. The count is simple and robust: it does not require sensitive content, and it is not dominated by how long you stay on each site. It is normal for the number to fluctuate with weekdays, weekends, or travel, so always compare against your own baseline rather than a universal target. Taken together with other features on this tab, a persistent reduction strengthens evidence for anhedonia."
        ),
    ),
    MetricDef(
        id="C2/F2",
        label="Median reply latency (s, chat)",
        dist_col="C2_F2_MedianReplyLatencySec",
        ok=45.0, higher_is_worse=True, fmt_tpl="{v:.0f} s",
        latex_formula=r"\tilde{L}=\mathrm{median}\{\Delta t(\text{inbound}\rightarrow\text{outbound})\},\ \Delta t\le 120s",
        latex_numbers_tpl=r"\tilde{L}={value:.0f}\ \mathrm{s}",
        explanation_md=(
            "Reply latency estimates how long it typically takes you to respond after a message arrives. We measure the time gap between an inbound packet and the next outbound packet in a chat flow, up to a 120‑second cap to focus on real turn‑taking. A higher median can indicate slowed initiation or reduced motivation to participate. We use the median (not the mean) so that a few long silences do not dominate the result. Network conditions and notification settings can introduce some noise, so interpret this together with session count and activity. Sustained increases over several days are more informative than isolated spikes."
        ),
    ),
    MetricDef(
        id="C2/F3",
        label="Chat session count (5‑min gap)",
        dist_col="C2_F3_ChatSessionCount",
        ok=3.0, higher_is_worse=False, fmt_tpl="{v:.0f}",
        latex_formula=r"\#\text{sessions}=\mathrm{segments}(\Delta t>300\ \mathrm{s})",
        explanation_md=(
            "This feature counts the number of messaging sessions per day. Sessions are formed by merging messages that are no more than five minutes apart; a longer pause starts a new session. A falling session count suggests fewer spontaneous or planned conversations, which can reflect reduced social drive. It focuses on interactions with known chat domains and ports and ignores passive media traffic. Short jumps up or down can be normal; look for multi‑day trends rather than single‑day changes. Combine with reply latency and upstream rate to understand not only how often but also how actively you engage."
        ),
    ),
    MetricDef(
        id="C2/F4",
        label="Mean upstream chat rate (bps)",
        dist_col="C2_F4_MeanUpstreamRateBps",
        ok=100.0, higher_is_worse=False, fmt_tpl="{v:.0f} bps",
        latex_formula=r"\bar{r}_{\uparrow}=\frac{1}{K}\sum_{k=1}^K \frac{B_{\uparrow,k}}{T_k}",
        latex_numbers_tpl=r"\bar{r}_{\uparrow}={value:.0f}\ \mathrm{bps}",
        explanation_md=(
            "This feature captures how much you actively transmit during chat sessions, in bytes per second. Lower values over time suggest more 'lurking'—reading without posting—whereas higher values indicate richer participation. By averaging within sessions, the metric down‑weights long idle periods between sessions. It is not affected by downstream streaming, background downloads, or page loads that do not involve messaging. Privacy is preserved: we measure volumes and timing, not content. As with all metrics, compare trends against your own baseline and typical weekday/weekend patterns."
        ),
    ),
    MetricDef(
        id="C2/F5",
        label="Passive/Active byte ratio",
        dist_col="C2_F5_PassiveActiveByteRatio",
        ok=8.0, higher_is_worse=True, fmt_tpl="{v:.2f}",
        latex_formula=r"\mathrm{R}=\frac{B_{\text{passive}}}{B_{\text{active}}}=\frac{B_{\downarrow,\ \text{stream}}}{B_{\uparrow,\ \text{chat}}}",
        latex_numbers_tpl=r"\mathrm{R}=\frac{{Bp}}{{Ba}}={value:.2f}",
        explanation_md=(
            "This ratio compares how much data you receive from passive media streams versus how much you actively send in chats. When the active component shrinks, the ratio rises, showing a shift from producing to consuming. If there is almost no active data, the ratio can become extremely large or mathematically infinite; that is a meaningful signal of near‑zero engagement. Short‑term spikes can happen on movie nights or travel days, so watch for multi‑day elevation. A sustained move upward, especially alongside fewer chat sessions and slower replies, strongly suggests reduced interest. Interpret against your personal baseline and consider day‑of‑week routines."
        ),
    ),
    MetricDef(
        id="C2/F6",
        label="Productivity‑site hits",
        dist_col="C2_F6_ProductivityHits",
        ok=5.0, higher_is_worse=False, fmt_tpl="{v:.0f}",
        latex_formula=r"\mathrm{Hits}_{\text{prod}}=\sum \mathbf{1}[\text{SLD}\in\mathcal{P}]",
        latex_numbers_tpl=r"\mathrm{Hits}_{\text{prod}}={value_int}",
        explanation_md=(
            "This counts visits to a curated set of planning and productivity services such as online calendars, note apps, and task managers. A noticeable drop may reflect reduced goal‑setting, planning, or follow‑through—all core aspects of anhedonia. Context matters: vacations, sick days, or weekends can naturally reduce these hits. Because the list is curated, you can adapt it to your tools to avoid false negatives. We count occurrences rather than time spent to keep the measure straightforward and comparable day‑to‑day. Use this together with the passive/active ratio to see whether active planning gives way to passive browsing."
        ),
    ),
    MetricDef(
        id="C2/F7",
        label="Social outgoing share (upstream)",
        dist_col="C2_F7_SocialOutgoingShareUp",
        ok=0.25, higher_is_worse=False, fmt_tpl="{v:.2f}",
        latex_formula=r"\mathrm{Share}_{\uparrow}=\frac{B_{\uparrow,\ \text{social}}}{B_{\uparrow,\ \text{social}}+B_{\downarrow,\ \text{social}}}",
        latex_numbers_tpl=r"\mathrm{Share}_{\uparrow}=\frac{{bytes_up}}{{bytes_up + bytes_down}}={value:.2f}",
        explanation_md=(
            "Within **social** sites, fraction of **upstream** bytes. "
            "Higher share suggests more posting/messaging vs scrolling."
        ),
    ),
]

# ----------------------------- helpers ----------------------------------------

def _unique_sld_count(df: pd.DataFrame) -> float:
    if df is None or df.empty or "SLD" not in df.columns:
        return np.nan
    return float(df["SLD"].dropna().nunique())

def _median_reply_latency_sec(df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    req = ["Timestamp", "Source IP", "Destination IP"]
    if df is None or df.empty or not all(c in df.columns for c in req):
        return (np.nan, {})
    d = df.loc[chat_mask(df), req].dropna().sort_values("Timestamp")
    if d.empty:
        return (np.nan, {})
    d["inb"]  = d.apply(lambda r: is_inbound(r["Source IP"], r["Destination IP"]), axis=1)
    d["outb"] = d.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1)
    ts = d["Timestamp"].values; inn = d["inb"].values; out = d["outb"].values
    gaps = []
    for i in range(len(d) - 1):
        if inn[i] and out[i + 1]:
            dt = (ts[i + 1] - ts[i]).astype("timedelta64[s]").astype(int)
            if 0 < dt <= 120:
                gaps.append(dt)
    if not gaps:
        return (np.nan, {"reply_pairs": 0})
    med = float(np.median(gaps))
    return (med, {"reply_pairs": len(gaps), "reply_median_s": med})

def _chat_session_count(df: pd.DataFrame, gap_sec: int = 300) -> float:
    d = df.loc[chat_mask(df)] if df is not None and not df.empty else pd.DataFrame()
    if d.empty or "Timestamp" not in d.columns:
        return np.nan
    sess = sessions_from_timestamps(d, gap_sec=gap_sec)
    return float(len(sess)) if sess else 0.0

def _mean_upstream_rate_bps(df: pd.DataFrame, gap_sec: int = 300) -> Tuple[float, Dict[str, Any]]:
    req = {"Timestamp", "Source IP", "Destination IP", "Length"}
    if df is None or df.empty or not req.issubset(df.columns):
        return (np.nan, {})
    chat = df.loc[chat_mask(df)].copy()
    if chat.empty:
        return (np.nan, {})
    sess = sessions_from_timestamps(chat, gap_sec=gap_sec)
    rates = []
    for a, b in sess:
        win = chat[(chat["Timestamp"] >= a) & (chat["Timestamp"] <= b)]
        if win.empty:
            continue
        dur = max(1.0, (b - a).total_seconds())
        up_b = float(win.loc[win.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1), "Length"].sum())
        rates.append(up_b / dur)
    return (float(np.mean(rates)) if rates else np.nan, {"sessions_n": len(sess)})

def _passive_active_byte_ratio(df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    req = {"Source IP", "Destination IP", "Length"}
    if df is None or df.empty or not req.issubset(df.columns):
        return (np.nan, {})
    # Passive: inbound streaming bytes
    Bp = float(df.loc[streaming_inbound_mask(df), "Length"].sum())
    # Active: outbound chat bytes
    chat = df.loc[chat_mask(df)]
    Ba = float(chat.loc[chat.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1), "Length"].sum()) if not chat.empty else 0.0
    if Ba > 0.0:
        return (Bp / Ba, {"Bp": int(Bp), "Ba": int(Ba)})
    return (np.inf if Bp > 0.0 else np.nan, {"Bp": int(Bp), "Ba": int(Ba)})

def _productivity_hits(df: pd.DataFrame) -> float:
    if df is None or df.empty or "SLD" not in df.columns:
        return np.nan
    return float(df["SLD"].isin(PRODUCTIVITY_SLDS).sum())

def _social_outgoing_share_up(df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    if df is None or df.empty or "SLD" not in df.columns:
        return (np.nan, {})
    soc = df[df["SLD"].isin(SOCIAL_SLDS)].copy()
    if soc.empty:
        # fallback: treat chat subset as "social"
        soc = df.loc[chat_mask(df)].copy()
        if soc.empty:
            return (np.nan, {})
    up = 0.0; down = 0.0
    if {"Source IP", "Destination IP", "Length"}.issubset(soc.columns):
        up   = float(soc.loc[soc.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1), "Length"].sum())
        down = float(soc.loc[soc.apply(lambda r: is_inbound(r["Source IP"], r["Destination IP"]), axis=1), "Length"].sum())
    else:
        # if bytes are missing, fall back to counts
        up   = float(len(soc))  # proxy
        down = 0.0
    den = up + down
    if den <= 0:
        return (np.nan, {"bytes_up": int(up), "bytes_down": int(down)})
    return (up / den, {"bytes_up": int(up), "bytes_down": int(down)})

# ------------------------------ main class ------------------------------------

class Criterion2:
    """Loss of interest / anhedonia — C2 (self-contained)."""

    def _compute_today_fields(self, df_day: pd.DataFrame) -> Dict[str, Any]:
        rec: Dict[str, Any] = {}
        rec["C2_F1_UniqueSLD"] = _unique_sld_count(df_day)
        rec["C2_F2_MedianReplyLatencySec"], _ = _median_reply_latency_sec(df_day)
        rec["C2_F3_ChatSessionCount"] = _chat_session_count(df_day, gap_sec=300)
        rec["C2_F4_MeanUpstreamRateBps"], _ = _mean_upstream_rate_bps(df_day, gap_sec=300)
        rec["C2_F5_PassiveActiveByteRatio"], rec["_EXTRAS_Ratio"] = _passive_active_byte_ratio(df_day)
        rec["C2_F6_ProductivityHits"] = _productivity_hits(df_day)
        rec["C2_F7_SocialOutgoingShareUp"], rec["_EXTRAS_Share"] = _social_outgoing_share_up(df_day)
        return rec

    def _latex_ctx(self, mdef: MetricDef, value: Any, today_fields: Dict[str, Any]) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {}
        ctx = update_extras_with_value(ctx, value)
        if mdef.id == "C2/F5":
            ctx.update(today_fields.get("_EXTRAS_Ratio", {}))
        if mdef.id == "C2/F7":
            ctx.update(today_fields.get("_EXTRAS_Share", {}))
        return ctx

    def compute(
        self,
        df_day: pd.DataFrame,
        today: dict,         # today_base (not used here, but kept for parity)
        aux_ctx: dict,       # per-day extras (not used here)
        ALL_DAILY: pd.DataFrame
    ) -> List[Dict[str, Any]]:

        today = dict(today or {})
        need_calc = any(
            (m.dist_col not in today) or pd.isna(today.get(m.dist_col)) for m in C2_DEFS
        )
        today_fields = (
            self._compute_today_fields(df_day) if need_calc and df_day is not None and not df_day.empty else {}
        )
        for k, v in today_fields.items():
            today.setdefault(k, v)

        out: List[Dict[str, Any]] = []

        for mdef in C2_DEFS:
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

            # Format & LaTeX numbers
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
