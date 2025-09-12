# metrics/base_features.py

from __future__ import annotations
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from .common import (
    SOCIAL_SLDS,
    PRODUCTIVITY_SLDS,
    FOOD_DELIVERY_SLDS,
    DIET_SLDS,
    SMART_SCALE_SLDS,
    MSG_PORTS,
    chat_mask,
    sessions_from_timestamps,
    is_outbound,
    is_inbound,
    is_private_ip,
)
# Optional: circular std uses minute-of-day; available via common if needed
from .common import circular_std_minutes  # safe import; used for C3_F7

# ----------------------------- small helpers ----------------------------------

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

def _compute_is_iv(per_min: pd.Series) -> tuple[float, float]:
    """Interdaily Stability (IS) and Intradaily Variability (IV) from a per-minute activity series."""
    if per_min is None or per_min.empty:
        return (float("nan"), float("nan"))
    s = per_min.astype(float).fillna(0.0)
    N = int(len(s))
    if N < 2:
        return (float("nan"), float("nan"))
    x = s.values
    xbar = float(x.mean())
    ssd = float(np.sum((x - xbar) ** 2))
    diffs = np.diff(x); sum_diffs2 = float(np.sum(diffs ** 2))
    IV = (N * sum_diffs2) / ((N - 1) * ssd) if ssd > 0 and (N - 1) > 0 else float("nan")

    # Hourly means across 24h
    df_tmp = s.to_frame("x"); df_tmp["hour"] = df_tmp.index.hour
    hourly_mean = df_tmp.groupby("hour")["x"].mean().reindex(range(24), fill_value=0.0).values
    num_IS = N * float(np.sum((hourly_mean - xbar) ** 2))
    den_IS = 24 * ssd
    IS = num_IS / den_IS if den_IS > 0 else float("nan")
    return (float(IS), float(IV))

def _chat_outbound_ts(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "Timestamp" not in df.columns:
        return pd.Series([], dtype="datetime64[ns]")
    chat = df.loc[chat_mask(df)]
    if chat.empty or (not {"Source IP","Destination IP"}.issubset(chat.columns)):
        return pd.Series([], dtype="datetime64[ns]")
    chat = chat.loc[chat.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1)]
    return chat["Timestamp"].sort_values()

# ------------------------------ main builder ----------------------------------

def compute_daily_base_record(df_d: pd.DataFrame) -> dict:

    rec: Dict[str, Any] = {}

    # Initialize all DistCols we plan to fill (ensures stable columns even on empty days)
    init_cols = [
        # C2
        "C2_F1_UniqueSLD", "C2_F3_ChatSessionCount", "C2_F6_ProductivityHits",
        # C3
        "C3_F1_FoodDeliveryHits", "C3_F2_LateNightDeliveryRatio", "C3_F3_MeanInterOrderDays",
        "C3_F4_DietSiteVisits", "C3_F5_TrackerBurstCount", "C3_F6_SmartScaleUploads",
        "C3_F7_WeighInTimeVarMin",
        # C5
        "C5_F1_DhcpPerHour", "C5_F2_WifiDwellMin", "C5_F3_MedianIKS",
        "C5_F4_IKSStd", "C5_F5_Sub30sSessions", "C5_F6_SessionFano",
        # C6
        "C6_F1_LongestMiddayIdleMin", "C6_F2_DayActiveSessionCount", "C6_F3_DayIdleRatio_09_18",
        "C6_F4_UpstreamBpsPerActiveHour", "C6_F5_PostChatCount", "C6_F6_MedianInterReqSec",
        "C6_F7_InterReqFano", "C6_F8_FirstActivityMin", "C6_F9_ActivationDelayVsBase28d",
        # C7
        "C7_F1_MentalHealthSiteVisitsDay", "C7_F2_NegativeSelfSearchRatio",
        "C7_F3_SelfAssessmentDwellSec", "C7_F4_HelpTherapyLookupHits",
        "C7_F5_AccountDeleteUnsubCount", "C7_F6_SettingsPrivacyDwellSec",
        "C7_F7_SocialOutgoingShareUp", "C7_F8_CloudUploadBytesToday",
        # C8
        "C8_F1_MedianPageDwellSec", "C8_F2_DNSBurstRatePerHour",
        "C8_F3_NotificationMicroSessionsCount", "C8_F4_RepeatedQueryRatio60m",
        "C8_F5_QueryReformulationMax", "C8_F6_BackNavShare",
        "C8_F7_SERPTimeToFirstClickSec", "C8_F8_MedianIKSsec",
        # C9
        "C9_F1_CrisisLineHits", "C9_F2_SuicideMethodQueryRatio", "C9_F3_TherapyBookingVisits",
        "C9_F4_SelfHarmForumVisits", "C9_F5_SelfHarmForumUpBytes",
        "C9_F6_SelfHarmForumMeanSessLenSec", "C9_F7_WillInsuranceDownloads",
        "C9_F8_CloudBackupUpBytesToday", "C9_F9_AccountDeletionRequestsCount",
        "C9_F10_NightSuicideQueryBursts", "C9_F11_NightNegativeSearchRatio",
    ]
    for c in init_cols:
        rec[c] = float("nan")

    if df_d is None or df_d.empty:
        # Minimal TODAY fields (still NaN) to keep schema stable
        rec["LateNightShare"] = float("nan")
        rec["LongestInactivityHours"] = float("nan")
        rec["ActiveNightMinutes"] = float("nan")
        rec["ActiveDayMinutes"] = float("nan")
        rec["ND_Ratio"] = float("nan")
        rec["IS"] = float("nan")
        rec["IV"] = float("nan")
        rec["F1_DistinctSocial"] = float("nan")
        rec["F2_MeanSocialDurSec"] = float("nan")
        return rec

    # --- canonicalize time & convenience columns
    d = df_d.copy()
    d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="coerce")
    d = d.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    d["Hour"] = d["Timestamp"].dt.hour

    total_packets = int(d.shape[0])
    # Store basic packet counts for later reuse (aux_ctx / LaTeX numbers)
    rec["n_total_packets_today"] = float(total_packets)
    rec["n_night_packets_today"] = float(d["Hour"].isin(range(0, 6)).sum())
    rec["night_pkts_today"] = float(d["Hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).sum())
    rec["day_pkts_today"] = float(d["Hour"].isin(range(6, 22)).sum())

    # Store direction-approximate byte counters for downstream/upstream
    if {"Length", "Destination IP", "Source IP"}.issubset(d.columns):
        try:
            down_bytes = d[
                d["Destination IP"].apply(lambda x: is_private_ip(x) if pd.notna(x) else False)
                & (~d["Source IP"].apply(lambda x: is_private_ip(x) if pd.notna(x) else False))
            ]["Length"].sum()
            up_bytes = d[
                d["Source IP"].apply(lambda x: is_private_ip(x) if pd.notna(x) else False)
                & (~d["Destination IP"].apply(lambda x: is_private_ip(x) if pd.notna(x) else False))
            ]["Length"].sum()
        except Exception:
            down_bytes = np.nan
            up_bytes = np.nan
    else:
        down_bytes = np.nan
        up_bytes = np.nan
    rec["down_bytes_today"] = float(down_bytes) if pd.notna(down_bytes) else float("nan")
    rec["up_bytes_today"] = float(up_bytes) if pd.notna(up_bytes) else float("nan")

    # ── TODAY: LateNightShare (00–05)
    if total_packets > 0:
        night_packets = int(d["Hour"].isin([0, 1, 2, 3, 4, 5]).sum())
        rec["LateNightShare"] = night_packets / float(total_packets)
    else:
        rec["LateNightShare"] = float("nan")

    # ── TODAY: LongestInactivityHours
    if total_packets >= 2:
        diffs = np.diff(d["Timestamp"].values).astype("timedelta64[s]").astype(int)
        rec["LongestInactivityHours"] = float(diffs.max() / 3600.0) if diffs.size else float("nan")
    else:
        rec["LongestInactivityHours"] = float("nan")

    # ── TODAY: per-minute activity + rhythm metrics
    per_min = d.set_index("Timestamp").assign(cnt=1)["cnt"].resample("1Min").sum()
    active_min = (per_min > 0).astype(int)

    night_idx = active_min.index.hour.isin([0, 1, 2, 3, 4, 5])
    day_idx   = active_min.index.hour.isin(range(8, 21))
    night_active = int(active_min.loc[night_idx].sum()) if len(active_min) else 0
    day_active   = int(active_min.loc[day_idx].sum()) if len(active_min) else 0
    rec["ActiveNightMinutes"] = float(night_active)
    rec["ActiveDayMinutes"] = float(day_active)
    rec["ND_Ratio"] = (night_active / float(day_active)) if day_active > 0 else float("nan")

    IS_val, IV_val = _compute_is_iv((per_min > 0).astype(float)) if len(per_min) > 0 else (float("nan"), float("nan"))
    rec["IS"] = IS_val
    rec["IV"] = IV_val

    # ── TR: social / chat / productivity basics (C2 helpers)
    if "SLD" in d.columns:
        soc_mask = d["SLD"].isin(SOCIAL_SLDS) | (d["Destination Port"].isin(MSG_PORTS) if "Destination Port" in d.columns else False)
        rec["F1_DistinctSocial"] = float(d.loc[soc_mask, "SLD"].dropna().nunique())
    else:
        rec["F1_DistinctSocial"] = float("nan")

    d_social = d.loc[chat_mask(d)].copy()
    sess_social = sessions_from_timestamps(d_social, gap_sec=300) if not d_social.empty else []
    if sess_social:
        secs = [(b - a).total_seconds() for a, b in sess_social]
        rec["F2_MeanSocialDurSec"] = float(np.mean(secs)) if secs else float("nan")
    else:
        rec["F2_MeanSocialDurSec"] = float("nan")

    rec["C2_F1_UniqueSLD"] = float(d["SLD"].dropna().nunique()) if "SLD" in d.columns else float("nan")
    rec["C2_F3_ChatSessionCount"] = float(len(sess_social)) if sess_social else float("nan")
    rec["C2_F6_ProductivityHits"] = float(d[d["SLD"].isin(PRODUCTIVITY_SLDS)].shape[0]) if "SLD" in d.columns else float("nan")

    # ── C3: Appetite / Weight --------------------------------------------------
    if "SLD" in d.columns:
        # F1: delivery sessions (5min gap)
        rows_deliv = d[d["SLD"].isin(FOOD_DELIVERY_SLDS)]
        if not rows_deliv.empty:
            sess_deliv = sessions_from_timestamps(rows_deliv, gap_sec=300)
            rec["C3_F1_FoodDeliveryHits"] = float(len(sess_deliv))
            # F2: late-night ratio (22:00–06:00)
            starts = [a for (a, _) in sess_deliv]
            if starts:
                night = sum(1 for t in starts if pd.to_datetime(t).hour >= 22 or pd.to_datetime(t).hour < 6)
                rec["C3_F2_LateNightDeliveryRatio"] = (night / float(len(starts))) if len(starts) > 0 else float("nan")
            # F3: mean inter-order interval (days, within day) — needs ≥2
            if len(sess_deliv) >= 2:
                starts_sorted = sorted([pd.to_datetime(a) for (a, _) in sess_deliv])
                diffs_days = np.diff(np.array(starts_sorted, dtype="datetime64[s]")).astype("timedelta64[s]").astype(int) / 86400.0
                rec["C3_F3_MeanInterOrderDays"] = float(np.mean(diffs_days)) if diffs_days.size else float("nan")
        # F4: diet/nutrition site visits
        rec["C3_F4_DietSiteVisits"] = float(d[d["SLD"].isin(DIET_SLDS)].shape[0])

        # F5: calorie-tracker bursts (10‑min bins, threshold 3)
        if not d.empty and d["SLD"].notna().any():
            rows_diet = d[d["SLD"].isin(DIET_SLDS)].copy()
            if not rows_diet.empty and "Timestamp" in rows_diet.columns:
                rows_diet["bin"] = pd.to_datetime(rows_diet["Timestamp"]).dt.floor("10min")
                counts = rows_diet.groupby(["SLD", "bin"]).size().reset_index(name="cnt")
                bursts = int((counts["cnt"] >= 3).sum())
                rec["C3_F5_TrackerBurstCount"] = float(bursts)

        # F6: smart-scale uploads (outbound sessions, 10‑min gap)
        rows_scale = d[d["SLD"].isin(SMART_SCALE_SLDS)].copy()
        if not rows_scale.empty:
            if {"Source IP","Destination IP"}.issubset(rows_scale.columns):
                rows_scale = rows_scale[rows_scale.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1)]
            if not rows_scale.empty:
                sess_scale = sessions_from_timestamps(rows_scale, gap_sec=600)
                rec["C3_F6_SmartScaleUploads"] = float(len(sess_scale))
            # F7: weigh-in time variability (circular SD, minutes)
            ts_list = pd.to_datetime(rows_scale["Timestamp"]).tolist()
            if len(ts_list) >= 3:
                val = circular_std_minutes(ts_list)
                rec["C3_F7_WeighInTimeVarMin"] = float(val) if val is not None else float("nan")

    # ── C5: Psychomotor agitation / retardation --------------------------------
    # F1: DHCP per hour
    if "Timestamp" in d.columns:
        d_dhcp = d.loc[d.apply(_is_dhcp_row, axis=1)].sort_values("Timestamp")
        events_total = int(d_dhcp.shape[0])
        if d["Timestamp"].notna().any():
            span_h = (d["Timestamp"].max() - d["Timestamp"].min()).total_seconds() / 3600.0
            hours_considered = float(min(24.0, max(1.0, span_h)))
        else:
            hours_considered = 24.0
        rec["C5_F1_DhcpPerHour"] = (events_total / hours_considered) if hours_considered > 0 else float("nan")

        # F2: mean dwell minutes (needs ≥2)
        if events_total >= 2:
            gaps_min = (d_dhcp["Timestamp"].diff().dt.total_seconds().dropna() / 60.0).to_numpy()
            rec["C5_F2_WifiDwellMin"] = float(np.mean(gaps_min)) if gaps_min.size else float("nan")

    # F3/F4: inter-keystroke gaps from outbound chat timestamps
    ts_out = _chat_outbound_ts(d)
    if not ts_out.empty:
        gaps = ts_out.diff().dt.total_seconds().dropna()
        gaps = gaps[(gaps > 0) & (gaps <= 3.0)]
        if not gaps.empty:
            rec["C5_F3_MedianIKS"] = float(gaps.median())
            if gaps.shape[0] >= 3:
                rec["C5_F4_IKSStd"] = float(gaps.std(ddof=1))

    # F5: sub‑30s sessions with 5‑min idle segmentation
    if "Timestamp" in d.columns:
        sess_all = sessions_from_timestamps(d.sort_values("Timestamp"), gap_sec=300)
        if sess_all:
            durations = np.array([(b - a).total_seconds() for a, b in sess_all], dtype=float)
            rec["C5_F5_Sub30sSessions"] = float((durations < 30.0).sum())
        else:
            rec["C5_F5_Sub30sSessions"] = 0.0  # explicit zero when no sessions form

    return rec
