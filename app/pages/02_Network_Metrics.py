# pages/02_Network_Metrics.py

from __future__ import annotations

import os
import sys
import re
import hashlib
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pyarrow as pa
import pyarrow.parquet as pq


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ---- criterion implementations ----
from metrics.criterion1 import Criterion1
from metrics.criterion2 import Criterion2
from metrics.criterion3 import Criterion3
from metrics.criterion4 import Criterion4
from metrics.criterion5 import Criterion5
from metrics.criterion6 import Criterion6
from metrics.criterion7 import Criterion7
from metrics.criterion8 import Criterion8
from metrics.criterion9 import Criterion9

from metrics.common import (
    enrich_with_hostnames,
    is_private_ip,
    is_outbound,
    is_inbound,
    streaming_inbound_mask,
    chat_mask,
    sessions_from_timestamps,
    MSG_PORTS,
    SOCIAL_SLDS,
    STREAMING_SLDS,
    PRODUCTIVITY_SLDS,
    FOOD_DELIVERY_SLDS,
    DIET_SLDS,
    SMART_SCALE_SLDS,
    MENTAL_HEALTH_SLDS,
    CRISIS_SLDS,
    THERAPY_SLDS,
    CLOUD_STORAGE_SLDS,
    SELF_HARM_FORUM_PATTERNS,
    SUICIDE_QUERY_PATTERNS,
    TRACKER_BURST_THRESHOLD,
)
from metrics.base_features import compute_daily_base_record

# Status ordering used for sorting KPI tiles in the grid (lower=earlier)
STATUS_ORDER = {"OK": 0, "Caution": 1, "N/A": 2}

# =============================== Page/UI ======================================

st.set_page_config(
    page_title="CareNet - Nef, Stephan",
    page_icon="https://upload.wikimedia.org/wikipedia/de/thumb/7/77/Uni_St_Gallen_Logo.svg/2048px-Uni_St_Gallen_Logo.svg.png",
    layout="wide",
)

# Streamlit added st.logo recently; fall back to st.image if not available
try:
    st.logo(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/HSG_Logo_DE_RGB.svg/1024px-HSG_Logo_DE_RGB.svg.png",
        link="https://www.unisg.ch/de/",
    )
except Exception:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/HSG_Logo_DE_RGB.svg/1024px-HSG_Logo_DE_RGB.svg.png",
        use_column_width=False,
    )



# =============================== Auth =========================================

@st.dialog("Login")
def login():
    try:
        _logo_path = Path(__file__).resolve().parents[1] / "utils" / "logo.svg"
        st.image(str(_logo_path), use_container_width=True)
    except Exception:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/HSG_Logo_DE_RGB.svg/1024px-HSG_Logo_DE_RGB.svg.png", use_container_width=True)
    st.subheader("👋🏻 welcome - please login")
    username = st.text_input("Username", placeholder="nef")
    password = st.text_input("Password", type="password")
    st.info("ℹ️ if you need access please reach out to stephan.nef@student.unisg.ch")
    if username and password:
        if username == st.secrets["username"] and password == st.secrets["password"]:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.session_state.logged_in = False
            st.error("Invalid login data!")
    else:
        st.session_state.logged_in = False

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login()
    st.stop()


col1, col2 = st.columns([7, 2])
with col1:
    st.title("Network Traffic Metrics mapped to DSM‑5 Indicators")
    st.caption(
        """
This dashboard aggregates daily PCAP‑derived features (e.g., session structure,
bytes directionality, DNS/SNI patterns) and maps them to proxy indicators for
DSM‑5 criteria. Select datasets and a day to compute and visualize per‑criterion
KPIs, their status, and time‑series context.
        """
    )
with col2:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/HSG_Logo_DE_RGB.svg/1024px-HSG_Logo_DE_RGB.svg.png"
    )


# =============================== Paths / Caching ===============================

APP_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = APP_DIR / "processed_parquet"
os.makedirs(PROCESSED_DIR, exist_ok=True)

FEATURE_CACHE_DIR = APP_DIR / "feature_cache"
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

# ---------- Robust Parquet read (skips corrupted row groups) ----------
def _safe_read_parquet(fp: str):
    try:
        return pd.read_parquet(fp)
    except Exception:
        try:
            pf = pq.ParquetFile(fp)
            subs = []
            for i in range(pf.num_row_groups):
                try:
                    t = pf.read_row_group(i)
                    subs.append(t)
                except Exception:
                    continue
            if not subs:
                return None
            table = pa.concat_tables(subs, promote=True)
            try:
                return table.to_pandas(types_mapper=pd.ArrowDtype)
            except Exception:
                return table.to_pandas()
        except Exception:
            return None

@st.cache_data(show_spinner=False)
def list_partition_files_cached(base_name: str) -> list[str]:
    """Return all parquet partition files for a base dataset name."""
    dataset_dir = os.path.join(PROCESSED_DIR, base_name)
    if not os.path.isdir(dataset_dir):
        return []
    return sorted(
        os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".parquet")
    )

def partition_file_to_start_dt(path: str) -> datetime | None:
    """Extract the start datetime from a filename pattern '__YYYYMMDD_HHMM.parquet'."""
    m = re.search(r"__(\d{8})_(\d{4})\.parquet$", os.path.basename(path))
    if not m:
        return None
    datestr, timestr = m.groups()
    try:
        return datetime.strptime(datestr + timestr, "%Y%m%d%H%M")
    except Exception:
        return None

def dataset_type(name: str) -> str:
    n = name.lower()
    if "onu" in n:
        return "ONU"
    if "bras" in n:
        return "BRAS"
    return "Other"

def group_prefix(name: str) -> str:
    """Strip trailing numeric suffixes for grouping similarly named datasets."""
    return re.sub(r"([_-]?\d+)$", "", name)

def group_token_from_prefix(prefix: str) -> str:
    s = os.path.basename(prefix).lower()
    s = re.sub(r"^(onu_|bras_|other_)", "", s)
    s = re.sub(r"^capture_", "", s)
    s = re.sub(r"^[_-]+", "", s)
    return s

@st.cache_data(show_spinner=False)
def partition_counts_by_date(base_names: list[str]) -> dict[pd.Timestamp, int]:
    from collections import defaultdict
    counts = defaultdict(int)
    for bn in base_names:
        for fp in list_partition_files_cached(bn):
            dt = partition_file_to_start_dt(fp)
            if dt:
                counts[pd.to_datetime(dt.date())] += 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))

@st.cache_data(show_spinner=False)
def load_day_dataframe(base_name: str, day) -> pd.DataFrame:
    day = pd.to_datetime(day).normalize()
    next_day = day + pd.Timedelta(days=1)
    files = list_partition_files_cached(base_name)
    chosen = []
    for p in files:
        dt = partition_file_to_start_dt(p)
        if dt and (day <= pd.to_datetime(dt) < next_day):
            chosen.append(p)
    if not chosen:
        return pd.DataFrame(
            columns=[
                "Timestamp",
                "Date",
                "Hour",
                "Protocol",
                "Source IP",
                "Destination IP",
                "Source Port",
                "Destination Port",
                "Length",
                "IsDNS",
                "DNS_QNAME",
                "DNS_ANS_NAME",
                "DNS_ANS_IPS",
            ]
        )
    

    dfs = []
    for fp in chosen:
        dfp = _safe_read_parquet(fp)
        if dfp is None:
            continue
        if "Timestamp" not in dfp.columns and {"Date", "Hour"}.issubset(dfp.columns):
            dfp["Timestamp"] = pd.to_datetime(dfp["Date"].astype(str)) + pd.to_timedelta(dfp["Hour"], unit="h")
        dfp["Timestamp"] = pd.to_datetime(dfp["Timestamp"], errors="coerce")
        dfs.append(dfp)
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["Timestamp"] >= day) & (df["Timestamp"] < next_day)].copy()
    df["Date"] = df["Timestamp"].dt.date
    df["Hour"] = df["Timestamp"].dt.hour
    df["Dataset"] = base_name
    return df

def cache_key_for_selection(base_names: list[str]) -> str:
    return hashlib.md5("|".join(sorted(base_names)).encode("utf-8")).hexdigest()

def cache_path_for_selection(base_names: list[str]) -> str:
    return os.path.join(FEATURE_CACHE_DIR, f"features_{cache_key_for_selection(base_names)}.csv")

def fingerprint_for_day(base_names: list[str], day) -> str:
    """Fingerprint a calendar day from all 5‑min parquet files (name|size|mtime)."""
    day = pd.to_datetime(day).normalize()
    next_day = day + pd.Timedelta(days=1)
    parts: list[str] = []
    for bn in base_names:
        for p in list_partition_files_cached(bn):
            dt = partition_file_to_start_dt(p)
            if dt and (day <= pd.to_datetime(dt) < next_day):
                try:
                    st_stat = os.stat(p)
                    parts.append(f"{os.path.basename(p)}|{st_stat.st_size}|{int(st_stat.st_mtime)}")
                except OSError:
                    continue
    return hashlib.md5("|".join(sorted(parts)).encode("utf-8")).hexdigest()

@st.cache_data(show_spinner=False)
def compute_or_load_all_days_features(
    base_names: list[str], all_days_list: list[pd.Timestamp], force: bool = False
) -> pd.DataFrame:
    cpath = cache_path_for_selection(base_names)
    existing: dict[datetime, pd.Series] = {}
    if os.path.isfile(cpath):
        try:
            prev = pd.read_csv(cpath, parse_dates=["Date"])
            existing = {pd.to_datetime(r.Date).date(): r for _, r in prev.iterrows()}
        except Exception:
            prev = pd.DataFrame()
    else:
        prev = pd.DataFrame()

    # If cache exists (regardless of Fingerprint), and we are not forcing recompute,
    # and the cache covers the requested days, load directly from CSV.
    if (not force) and ("prev" in locals()) and (not prev.empty):
        df_loaded = prev.copy()
        if all_days_list:
            wanted = set(pd.to_datetime(all_days_list).normalize())
            have = set(df_loaded["Date"].dt.normalize())
            if wanted.issubset(have):
                with st.status("Loading all-days metrics from CSV cache…", expanded=False) as s:
                    df_loaded = df_loaded[df_loaded["Date"].dt.normalize().isin(wanted)]
                    df_loaded = df_loaded.sort_values("Date")
                    s.update(label=f"Loaded {len(df_loaded)} day(s) from cache.", state="complete")
                return df_loaded

    # Determine if any day actually needs recomputation
    fingerprints: dict[date, str] = {}
    needs_compute = False
    for d in sorted(all_days_list):
        day_date = pd.to_datetime(d).date()
        fp = fingerprint_for_day(base_names, d)
        fingerprints[day_date] = fp
        if force or (day_date not in existing) or (str(existing[day_date].get("Fingerprint", "")) != fp):
            needs_compute = True

    if not needs_compute:
        with st.status("Loading all-days metrics from CSV cache…", expanded=False) as s:
            rows = [existing[pd.to_datetime(d).date()].to_dict() for d in sorted(all_days_list) if pd.to_datetime(d).date() in existing]
            df_loaded = pd.DataFrame(rows).sort_values("Date") if rows else pd.DataFrame(columns=["Date"])
            s.update(label=f"Loaded {len(df_loaded)} day(s) from cache.", state="complete")
            return df_loaded

    rows: list[dict] = []
    with st.status(
        f"Building all-days metrics cache for {len(all_days_list)} day(s)…", expanded=False
    ) as cache_stat:
        for d in sorted(all_days_list):
            day_date = pd.to_datetime(d).date()
            fp = fingerprints[day_date]
            if (not force) and (day_date in existing) and (str(existing[day_date].get("Fingerprint", "")) == fp):
                rows.append(existing[day_date].to_dict())
                continue

            cache_stat.write(f"Computing base features: {pd.to_datetime(d).date()}")
            day_frames = []
            for bn in base_names:
                df_b = load_day_dataframe(bn, d)
                if not df_b.empty:
                    day_frames.append(df_b)
            if not day_frames:
                continue
            df_day_full = pd.concat(day_frames, ignore_index=True)
            df_day_full["Timestamp"] = pd.to_datetime(df_day_full["Timestamp"], errors="coerce")
            df_day_full = enrich_with_hostnames(df_day_full)

            rec = compute_daily_base_record(df_day_full)
            rec["Date"] = pd.to_datetime(d)
            rec["Fingerprint"] = fp

            aux_ctx = dict(today_row=rec, ALL_DAILY=pd.DataFrame(rows))
            crit_instances = [
                Criterion1(),
                Criterion2(),
                Criterion3(),
                Criterion4(),
                Criterion5(),
                Criterion6(),
                Criterion7(),
                Criterion8(),
                Criterion9(),
            ]
            for inst in crit_instances:
                try:
                    metrics = inst.compute(df_day_full, rec, aux_ctx, pd.DataFrame(rows))
                    for m in metrics:
                        dc = m.get("dist_col")
                        if dc:
                            rec[dc] = m.get("value")
                except Exception:
                    continue

            rows.append(rec)

        cache_stat.update(label="Writing CSV cache…", state="running")
        if rows:
            df_all = pd.DataFrame(rows).sort_values("Date")
            df_all.to_csv(cpath, index=False)
            cache_stat.update(label=f"All-days metrics cached → {os.path.basename(cpath)}", state="complete")
            return df_all
        else:
            cache_stat.update(label="No rows computed for cache.", state="error")
            return pd.DataFrame(columns=["Date"])

# =============================== Sidebar: selection ===========================

with st.spinner("Scanning available datasets…"):
    all_datasets = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]
    all_datasets = sorted(set(all_datasets))

with st.sidebar:
    st.header("Data selection")
    selected_types = st.multiselect(
        "Filter by dataset type",
        options=["ONU", "BRAS", "Other"],
        default=["ONU", "BRAS", "Other"],
        key="filter_types",
    )

def type_filter(name: str) -> bool:
    return dataset_type(name) in selected_types

filtered_datasets = [d for d in all_datasets if type_filter(d)]

# Build group mapping from filtered datasets (by common prefix)
token_to_dsets: dict[str, set[str]] = {}
for name in filtered_datasets:
    pref = group_prefix(name)
    tok = group_token_from_prefix(pref)
    token_to_dsets.setdefault(tok, set()).add(name)

token_options = sorted(token_to_dsets.keys())
quick_picks = ["[ALL]", "[ALL ONU]", "[ALL BRAS]", "[ALL OTHER]"]
group_display_options = quick_picks + token_options

with st.sidebar:
    selected_group_tokens = st.multiselect(
        "Select dataset groups (prefix match)",
        options=group_display_options,
        default=["[ALL OTHER]"],
        key="group_tokens",
    )

auto_selected_from_groups: set[str] = set()
if "[ALL]" in selected_group_tokens:
    auto_selected_from_groups |= set(filtered_datasets)
if "[ALL ONU]" in selected_group_tokens:
    auto_selected_from_groups |= {d for d in filtered_datasets if dataset_type(d) == "ONU"}
if "[ALL BRAS]" in selected_group_tokens:
    auto_selected_from_groups |= {d for d in filtered_datasets if dataset_type(d) == "BRAS"}
if "[ALL OTHER]" in selected_group_tokens:
    auto_selected_from_groups |= {d for d in filtered_datasets if dataset_type(d) == "Other"}
for tok in selected_group_tokens:
    if tok in quick_picks:
        continue
    auto_selected_from_groups |= token_to_dsets.get(tok, set())

with st.sidebar:
    selected_individual = st.multiselect(
        "Additionally select individual datasets",
        options=filtered_datasets,
        default=sorted(auto_selected_from_groups),
        key="individual_datasets",
    )

selected_base_names = sorted(set(selected_individual) | set(auto_selected_from_groups))

if not selected_base_names:
    st.info(
        "Use the **sidebar** to pick dataset type(s), groups or individual datasets. "
        "Once selected, choose a day to load the dashboards."
    )
    st.stop()

# =============================== Days / calendar ==============================

def _render_calendar_heatmap(counts_by_date: dict[pd.Timestamp, int]):
    if not counts_by_date:
        return
    dates = pd.date_range(min(counts_by_date.keys()), max(counts_by_date.keys()), freq="D")
    key_map = {pd.to_datetime(k).normalize(): v for k, v in counts_by_date.items()}
    df = pd.DataFrame({"date": dates})
    df["count"] = df["date"].map(key_map).fillna(0).astype(int)
    df["weekday"] = df["date"].dt.weekday  # 0..6
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.isocalendar().year.astype(int)
    df["year_week"] = df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2)
    pivot = df.pivot(index="weekday", columns="year_week", values="count").reindex(index=[0, 1, 2, 3, 4, 5, 6])
    y_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    colorscale = [[0.0, "#ecfdf5"], [1.0, "#16a34a"]]  # green scale
    fig = go.Figure(go.Heatmap(z=pivot.values, x=list(pivot.columns), y=y_labels, colorscale=colorscale, showscale=False))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10))
    cal_key = "cal_" + hashlib.md5(
        "|".join([str(pd.to_datetime(k).date()) for k in sorted(counts_by_date.keys())]).encode()
    ).hexdigest()[:8]
    st.plotly_chart(fig, use_container_width=True, key=cal_key)

with st.status("Indexing available days for the selected datasets…", expanded=False) as idx_stat:
    counts_by_date = partition_counts_by_date(selected_base_names)
    available_days = list(counts_by_date.keys())
    if not available_days:
        idx_stat.update(label="No 5‑minute partitions found for the current selection.", state="error")
        st.stop()
    idx_stat.update(label=f"Found {len(available_days)} day(s).", state="complete")
    _render_calendar_heatmap(counts_by_date)

with st.sidebar:
    chosen_day = st.selectbox(
        "Select day",
        options=available_days,
        index=len(available_days) - 1,
        format_func=lambda d: pd.to_datetime(d).strftime("%Y-%m-%d"),
        key="chosen_day",
    )
    force_cache_refresh = st.checkbox("Recompute all-days metric cache", value=False, key="force_cache")
    live_recompute = st.checkbox("Recompute selected day from raw data", value=False, key="live_recompute")

# Build / load ALL_DAILY cache once per selection unless user forces refresh
st.session_state.setdefault("__ALL_DAILY_CACHE__", {})
_days_sig = "|".join([str(pd.to_datetime(d).date()) for d in sorted(available_days)])
_sel_sig = "|".join(sorted(selected_base_names))
ALL_DAILY_KEY = hashlib.md5(f"{_sel_sig}||{_days_sig}".encode()).hexdigest()

if (not force_cache_refresh) and (ALL_DAILY_KEY in st.session_state["__ALL_DAILY_CACHE__"]):
    ALL_DAILY = st.session_state["__ALL_DAILY_CACHE__"][ALL_DAILY_KEY]
else:
    ALL_DAILY = compute_or_load_all_days_features(selected_base_names, available_days, force=force_cache_refresh)
    st.session_state["__ALL_DAILY_CACHE__"][ALL_DAILY_KEY] = ALL_DAILY

# =============================== Load & enrich selected day ===================

def selection_key(base_names: list[str], day) -> str:
    return hashlib.md5(("|".join(base_names) + "||" + str(pd.to_datetime(day).date())).encode("utf-8")).hexdigest()

SEL_KEY = selection_key(selected_base_names, chosen_day)
st.session_state.setdefault("__day_cache__", {})
st.session_state.setdefault("__metrics_cache__", {})
st.session_state.setdefault("__range_overrides__", {})  # per-metric overrides keyed by "<label>|<dist_col>"

# Reuse day + metrics if already computed for this selection
if (not live_recompute) and SEL_KEY in st.session_state["__day_cache__"]:
    df_day = st.session_state["__day_cache__"][SEL_KEY]["df_day"]
    today_base = st.session_state["__day_cache__"][SEL_KEY]["today_base"]
    aux_ctx = st.session_state["__day_cache__"][SEL_KEY]["aux_ctx"]
else:
    today_row = (
        ALL_DAILY[ALL_DAILY["Date"].dt.date == pd.to_datetime(chosen_day).date()].iloc[-1].to_dict()
        if (not ALL_DAILY.empty and "Date" in ALL_DAILY.columns
            and (ALL_DAILY["Date"].dt.date == pd.to_datetime(chosen_day).date()).any())
        else {}
    )

    if live_recompute:
        with st.status("Loading & enriching selected day…", expanded=False) as status:
            frames = []
            for bn in selected_base_names:
                df_b = load_day_dataframe(bn, chosen_day)
                if not df_b.empty:
                    frames.append(df_b)
            if not frames:
                status.update(label="No traffic for selected day.", state="error")
                st.stop()
            df_day = pd.concat(frames, ignore_index=True)
            df_day["Timestamp"] = pd.to_datetime(df_day["Timestamp"], errors="coerce")
            df_day = enrich_with_hostnames(df_day)
            status.update(label="Selected day ready.", state="complete")

        with st.status("Computing per-minute activity, auxiliary features, and today's base record…", expanded=False):
            REQUIRED_COLS = [
                "Timestamp",
                "Date",
                "Hour",
                "Protocol",
                "Source IP",
                "Destination IP",
                "Source Port",
                "Destination Port",
                "Length",
            ]
            missing_cols = [c for c in REQUIRED_COLS if c not in df_day.columns]
            for c in missing_cols:
                df_day[c] = np.nan

            if missing_cols:
                st.info(
                    "Some expected columns are missing for this day: "
                    f"{', '.join(missing_cols)}. Metrics that depend on these (e.g., direction‑specific bytes) "
                    "will be shown as N/A."
                )

            per_min = df_day.set_index("Timestamp").assign(cnt=1)["cnt"].resample("1Min").sum()
            active_min_series = (per_min > 0).astype(float)

            def compute_is_iv(series: pd.Series) -> tuple[float, float, dict]:
                s = series.copy().astype(float).fillna(0.0)
                N = int(len(s))
                if N < 2:
                    return (np.nan, np.nan, {})
                x = s.values
                xbar = float(x.mean())
                ssd = float(np.sum((x - xbar) ** 2))
                diffs = np.diff(x)
                sum_diffs2 = float(np.sum(diffs ** 2))
                IV = (N * sum_diffs2) / ((N - 1) * ssd) if ssd > 0 and (N - 1) > 0 else np.nan
                df_tmp = s.to_frame("x")
                df_tmp["hour"] = df_tmp.index.hour
                hourly_mean = df_tmp.groupby("hour")["x"].mean().reindex(range(24), fill_value=0.0).values
                num_IS = N * float(np.sum((hourly_mean - xbar) ** 2))
                den_IS = 24 * ssd
                IS = num_IS / den_IS if den_IS > 0 else np.nan
                return (float(IS), float(IV), dict(N=N, xbar=xbar))

            IS_val, IV_val, _ = compute_is_iv(active_min_series) if active_min_series.size >= 60 else (np.nan, np.nan, {})
            night_mask = active_min_series.index.hour.isin(range(0, 6))
            day_mask = active_min_series.index.hour.isin(range(8, 21))
            night_active_mins = int(active_min_series.loc[night_mask].sum())
            day_active_mins = int(active_min_series.loc[day_mask].sum())
            nd_ratio = (night_active_mins / day_active_mins) if day_active_mins > 0 else np.nan

            night_pkts_today = int(df_day[df_day["Hour"].isin([22, 23, 0, 1, 2, 3, 4, 5])].shape[0])
            day_pkts_today = int(df_day[df_day["Hour"].isin(list(range(6, 22)))].shape[0])

            if {"Length", "Destination IP", "Source IP"}.issubset(df_day.columns):
                try:
                    down_bytes_today = df_day[
                        (df_day["Destination IP"].apply(lambda x: is_private_ip(x) if pd.notna(x) else False))
                        & (~df_day["Source IP"].apply(lambda x: is_private_ip(x) if pd.notna(x) else False))
                    ]["Length"].sum()
                    up_bytes_today = df_day[
                        (df_day["Source IP"].apply(lambda x: is_private_ip(x) if pd.notna(x) else False))
                        & (~df_day["Destination IP"].apply(lambda x: is_private_ip(x) if pd.notna(x) else False))
                    ]["Length"].sum()
                except Exception:
                    down_bytes_today = np.nan
                    up_bytes_today = np.nan
            else:
                down_bytes_today = np.nan
                up_bytes_today = np.nan

            aux_ctx = dict(
                IS_val=IS_val,
                IV_val=IV_val,
                night_active_mins=night_active_mins,
                day_active_mins=day_active_mins,
                nd_ratio=nd_ratio,
                n_total_packets_today=len(df_day),
                n_night_packets_today=int((df_day["Hour"].isin(range(0, 6))).sum()),
                night_pkts_today=night_pkts_today,
                day_pkts_today=day_pkts_today,
                down_bytes_today=down_bytes_today,
                up_bytes_today=up_bytes_today,
                today_row=today_row,
                ALL_DAILY=ALL_DAILY,
            )

            today_base = compute_daily_base_record(df_day)
    else:
        df_day = pd.DataFrame()
        aux_ctx = dict(
            IS_val=today_row.get("IS"),
            IV_val=today_row.get("IV"),
            night_active_mins=today_row.get("ActiveNightMinutes"),
            day_active_mins=today_row.get("ActiveDayMinutes"),
            nd_ratio=today_row.get("ND_Ratio"),
            n_total_packets_today=today_row.get("n_total_packets_today"),
            n_night_packets_today=today_row.get("n_night_packets_today"),
            night_pkts_today=today_row.get("night_pkts_today"),
            day_pkts_today=today_row.get("day_pkts_today"),
            down_bytes_today=today_row.get("down_bytes_today"),
            up_bytes_today=today_row.get("up_bytes_today"),
            today_row=today_row,
            ALL_DAILY=ALL_DAILY,
        )
        today_base = today_row

    with st.status("Computing all criterion metrics…", expanded=False) as mstat:
        crit_instances = [
            Criterion1(),
            Criterion2(),
            Criterion3(),
            Criterion4(),
            Criterion5(),
            Criterion6(),
            Criterion7(),
            Criterion8(),
            Criterion9(),
        ]
        metrics_by_tab: list[list[dict]] = []
        for i, inst in enumerate(crit_instances, start=1):
            mstat.write(f"Criterion {i}: computing metrics…")
            metrics = inst.compute(df_day, today_base, aux_ctx, ALL_DAILY)
            metrics_by_tab.append(metrics)
        st.session_state["__metrics_cache__"][SEL_KEY] = metrics_by_tab
        st.session_state["__day_cache__"][SEL_KEY] = dict(df_day=df_day, today_base=today_base, aux_ctx=aux_ctx)
        mstat.update(label="All metrics computed.", state="complete")

# If cached, fetch the computed metrics
crit_instances = [
    Criterion1(),
    Criterion2(),
    Criterion3(),
    Criterion4(),
    Criterion5(),
    Criterion6(),
    Criterion7(),
    Criterion8(),
    Criterion9(),
]
if SEL_KEY not in st.session_state["__metrics_cache__"]:
    with st.status("Computing all criterion metrics…", expanded=False) as mstat:
        metrics_by_tab = [inst.compute(df_day, today_base, aux_ctx, ALL_DAILY) for inst in crit_instances]
        st.session_state["__metrics_cache__"][SEL_KEY] = metrics_by_tab
        mstat.update(label="All metrics computed.", state="complete")
else:
    metrics_by_tab = st.session_state["__metrics_cache__"][SEL_KEY]

# =============================== UI helpers ===================================

def badge(label: str, color: str = "gray", icon: str | None = None):
    """Tiny wrapper for a status pill; falls back if st.badge is unavailable."""
    try:
        st.badge(label, color=color, icon=icon)
    except Exception:
        colors = {"green": "#16a34a", "orange": "#f59e0b", "red": "#dc2626", "gray": "#6b7280", "blue": "#2563eb"}
        st.markdown(
            f"<span style='background:{colors.get(color, '#6b7280')};color:white;"
            f"padding:4px 8px;border-radius:999px;font-size:0.8rem;display:inline-block;'>{label}</span>",
            unsafe_allow_html=True,
        )

def metric_filter_ui(tab_key: str) -> set[str]:
    return set(
        st.multiselect(
            "Filter metrics by status",
            options=["OK", "Caution", "N/A"],
            default=["OK", "Caution"],
            key=f"{tab_key}_status_filter",
            help="Show only KPIs with these statuses. Order: OK → Caution → N/A.",
        )
    )

def get_effective_range_cfg(label: str, dist_col: str | None, base_cfg: dict | None):
    effective = dict(base_cfg or {})
    key = f"{label}|{dist_col}"
    override = st.session_state.get("__range_overrides__", {}).get(key)
    if override:
        effective.update({k: v for k, v in override.items() if v is not None})
    return effective

def status_from_value(value, range_cfg: dict | None, default_status: str) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or not np.isfinite(value))):
        return "N/A"
    if range_cfg and ("ok" in range_cfg):
        ok_thr = float(range_cfg["ok"])
        higher_is_worse = bool(range_cfg.get("higher_is_worse", True))
        try:
            v = float(value)
        except Exception:
            return "N/A"
        if higher_is_worse:
            return "OK" if v <= ok_thr else "Caution"
        else:
            return "OK" if v >= ok_thr else "Caution"
    return default_status

@st.dialog("Metric details", width="medium")
def _show_metric_dialog():
    payload = st.session_state.get("__metric_dialog_payload__", {})
    if not payload:
        st.info("No metric selected.")
        return

    label = payload.get("label", "Metric")
    ranges_str = payload.get("ranges_str")
    latex_formula = payload.get("latex_formula")
    latex_numbers = payload.get("latex_numbers")
    explanation = payload.get("explanation_md")
    dist_col = payload.get("dist_col")
    base_cfg = payload.get("range_cfg") or {}
    current_value = payload.get("current_value")
    ts_df = payload.get("ts_df")

    effective_cfg = get_effective_range_cfg(label, dist_col, base_cfg)

    st.markdown(f"### {label}")

    # Curated inputs used by metrics (domains/ports/patterns/thresholds)
    def _curated_for_metric(dist_col: str | None, label: str | None):
        d = str(dist_col or "")
        items: list[tuple[str, object, str]] = []

        def add(name: str, values: object, why: str):
            items.append((name, values, why))

        # Criterion 2 (social / chat / productivity)
        if d in {"C2_F3_ChatSessionCount", "C2_F4_MeanUpstreamRateBps", "C2_F5_PassiveActiveByteRatio", "C2_F7_SocialOutgoingShareUp"}:
            add("SOCIAL_SLDS", SOCIAL_SLDS, "Known social/chat services used to identify social traffic.")
            add("MSG_PORTS", MSG_PORTS, "Chat/messaging ports: 5222 (XMPP), 5223 (legacy TLS), 443 (HTTPS/TLS; many apps tunnel chat here).")
        if d == "C2_F5_PassiveActiveByteRatio":
            add("STREAMING_SLDS", STREAMING_SLDS, "Inbound streaming sites treated as passive consumption in the ratio.")
        if d == "C2_F6_ProductivityHits":
            add("PRODUCTIVITY_SLDS", PRODUCTIVITY_SLDS, "Productivity/office/tool domains counted as productivity hits.")

        # Criterion 3 (appetite/weight)
        if d.startswith("C3_F1") or d.startswith("C3_F2") or d.startswith("C3_F3"):
            add("FOOD_DELIVERY_SLDS", FOOD_DELIVERY_SLDS, "Food delivery services used to detect order sessions.")
        if d.startswith("C3_F4") or d.startswith("C3_F5"):
            add("DIET_SLDS", DIET_SLDS, "Diet/calorie tracker sites.")
            add("TRACKER_BURST_THRESHOLD", TRACKER_BURST_THRESHOLD, "Minimum events per 10‑minute bin counted as a ‘burst’ (default 3).")
        if d.startswith("C3_F6") or d.startswith("C3_F7"):
            add("SMART_SCALE_SLDS", SMART_SCALE_SLDS, "Smart scale vendors (used to detect weigh‑in uploads).")

        # Criterion 5 (psychomotor): inter‑keystroke from chat timestamps
        if d in {"C5_F3_MedianIKS", "C5_F4_IKSStd"}:
            add("SOCIAL_SLDS", SOCIAL_SLDS, "Chat/social subset used to approximate typing activity.")
            add("MSG_PORTS", MSG_PORTS, "Messaging ports used to detect chat flows for keystroke gaps.")

        # Criterion 7 (worthlessness/guilt)
        if d == "C7_F1_MentalHealthSiteVisitsDay":
            add("MENTAL_HEALTH_SLDS", MENTAL_HEALTH_SLDS, "Mental‑health resource sites.")
        if d == "C7_F4_HelpTherapyLookupHits":
            add("CRISIS_SLDS", CRISIS_SLDS, "Crisis hotline sites.")
            add("THERAPY_SLDS", THERAPY_SLDS, "Therapy‑finder portals.")
        if d == "C7_F7_SocialOutgoingShareUp":
            add("SOCIAL_SLDS", SOCIAL_SLDS, "Social domains considered for outgoing share.")
            add("MSG_PORTS", MSG_PORTS, "Messaging ports for social flows.")
        if d == "C7_F8_CloudUploadBytesToday":
            add("CLOUD_STORAGE_SLDS", CLOUD_STORAGE_SLDS, "Cloud storage providers used to attribute upload bytes.")

        # Criterion 9 (suicidality)
        if d == "C9_F1_CrisisLineHits":
            add("CRISIS_SLDS", CRISIS_SLDS, "Crisis hotline sites.")
        if d in {"C9_F2_SuicideMethodQueryRatio", "C9_F10_NightSuicideQueryBursts", "C9_F11_NightNegativeSearchRatio"}:
            add("SUICIDE_QUERY_PATTERNS", SUICIDE_QUERY_PATTERNS, "Keyword patterns matched in queries indicative of self‑harm methods or negative self‑talk.")
        if d in {"C9_F4_SelfHarmForumVisits", "C9_F5_SelfHarmForumUpBytes", "C9_F6_SelfHarmForumMeanSessLenSec"}:
            add("SELF_HARM_FORUM_PATTERNS", SELF_HARM_FORUM_PATTERNS, "Forum and subreddit path patterns used to detect self‑harm communities.")
        if d in {"C9_F8_CloudBackupUpBytesToday"}:
            add("CLOUD_STORAGE_SLDS", CLOUD_STORAGE_SLDS, "Cloud backup/storage providers used to attribute uploads.")

        return items

    # Time series: show three tabs inside a bordered container
    if isinstance(ts_df, pd.DataFrame) and dist_col in ts_df.columns and not ts_df.empty and "Date" in ts_df.columns:
        ts_plot = ts_df.dropna(subset=[dist_col]).copy()
        ts_plot["Date"] = pd.to_datetime(ts_plot["Date"])
        if not ts_plot.empty:
            y_min = float(ts_plot[dist_col].min())
            y_max = float(ts_plot[dist_col].max())
            if y_min == y_max:
                y_min -= 1.0
                y_max += 1.0

            weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday_dates = pd.date_range("2024-01-01", periods=7, freq="D")

            with st.container(border=True):
                # Order requested: Over time, Weekday totals, Weekly, Distribution
                tab_dates, tab_totals, tab_weekly, tab_box = st.tabs(["Over time", "Weekday totals", "Weekly", "Distribution"])

                # Tab 1: Weekly (one line per week across weekdays)
                with tab_weekly:
                    fig_w = go.Figure()
                    ts_plot["WeekStart"] = ts_plot["Date"].dt.to_period("W").dt.start_time
                    ts_plot["Weekday"] = ts_plot["Date"].dt.day_name()
                    for wk, sub in ts_plot.groupby("WeekStart"):
                        sub = sub.set_index("Weekday")[dist_col].reindex(weekday_names)
                        fig_w.add_trace(
                            go.Scatter(
                                x=weekday_dates,
                                y=sub.values,
                                mode="lines+markers",
                                name=str(pd.to_datetime(wk).date()),
                            )
                        )

                    if "ok" in effective_cfg:
                        ok_thr = float(effective_cfg["ok"])
                        higher_is_worse = bool(effective_cfg.get("higher_is_worse", True))
                        if higher_is_worse:
                            fig_w.add_hrect(y0=y_min, y1=ok_thr, line_width=0, fillcolor="rgba(34,197,94,0.15)", layer="below")
                            fig_w.add_hrect(y0=ok_thr, y1=y_max, line_width=0, fillcolor="rgba(239,68,68,0.15)", layer="below")
                        else:
                            fig_w.add_hrect(y0=y_min, y1=ok_thr, line_width=0, fillcolor="rgba(239,68,68,0.15)", layer="below")
                            fig_w.add_hrect(y0=ok_thr, y1=y_max, line_width=0, fillcolor="rgba(34,197,94,0.15)", layer="below")

                    try:
                        x_day = pd.to_datetime(pd.to_datetime(chosen_day).date())
                        x_val = weekday_dates[x_day.dayofweek]
                        fig_w.add_vline(x=x_val, line_dash="dot", line_color="gray")
                    except Exception:
                        pass

                    if current_value is not None and isinstance(current_value, (int, float)) and np.isfinite(current_value):
                        try:
                            fig_w.add_hline(y=float(current_value), line_dash="dash", line_color="red")
                        except Exception:
                            pass

                    key_w = "ts_weekly_" + hashlib.md5((label + "|" + str(dist_col)).encode()).hexdigest()[:8]
                    fig_w.update_layout(
                        height=380,
                        margin=dict(l=10, r=10, t=10, b=60),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                        xaxis_title="Weekday",
                        yaxis_title=dist_col,
                    )
                    fig_w.update_xaxes(
                        tickmode="array",
                        tickvals=weekday_dates,
                        ticktext=weekday_names,
                        range=[weekday_dates[0], weekday_dates[-1]],
                    )
                    st.plotly_chart(fig_w, use_container_width=True, key=key_w)

                # Tab 2: Weekday totals (sum Monday..Sunday)
                with tab_totals:
                    ts_plot["Weekday"] = ts_plot["Date"].dt.day_name()
                    agg = ts_plot.groupby("Weekday")[dist_col].sum().reindex(weekday_names)
                    fig_t = go.Figure(
                        data=[go.Scatter(x=weekday_dates, y=agg.values, mode="lines+markers", name="Total")]
                    )

                    if "ok" in effective_cfg:
                        ok_thr = float(effective_cfg["ok"])
                        higher_is_worse = bool(effective_cfg.get("higher_is_worse", True))
                        if higher_is_worse:
                            fig_t.add_hrect(y0=y_min, y1=ok_thr, line_width=0, fillcolor="rgba(34,197,94,0.15)", layer="below")
                            fig_t.add_hrect(y0=ok_thr, y1=y_max, line_width=0, fillcolor="rgba(239,68,68,0.15)", layer="below")
                        else:
                            fig_t.add_hrect(y0=y_min, y1=ok_thr, line_width=0, fillcolor="rgba(239,68,68,0.15)", layer="below")
                            fig_t.add_hrect(y0=ok_thr, y1=y_max, line_width=0, fillcolor="rgba(34,197,94,0.15)", layer="below")

                    try:
                        x_day = pd.to_datetime(pd.to_datetime(chosen_day).date())
                        x_val = weekday_dates[x_day.dayofweek]
                        fig_t.add_vline(x=x_val, line_dash="dot", line_color="gray")
                    except Exception:
                        pass

                    if current_value is not None and isinstance(current_value, (int, float)) and np.isfinite(current_value):
                        try:
                            fig_t.add_hline(y=float(current_value), line_dash="dash", line_color="red")
                        except Exception:
                            pass

                    key_t = "ts_totals_" + hashlib.md5((label + "|" + str(dist_col)).encode()).hexdigest()[:8]
                    fig_t.update_layout(
                        height=380,
                        margin=dict(l=10, r=10, t=10, b=60),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                        xaxis_title="Weekday",
                        yaxis_title=dist_col,
                    )
                    fig_t.update_xaxes(
                        tickmode="array",
                        tickvals=weekday_dates,
                        ticktext=weekday_names,
                        range=[weekday_dates[0], weekday_dates[-1]],
                    )
                    st.plotly_chart(fig_t, use_container_width=True, key=key_t)

                # Tab 3: Over time (Date on X-axis)
                with tab_dates:
                    fig_d = go.Figure(
                        data=[
                            go.Scatter(
                                x=ts_plot["Date"].sort_values(),
                                y=ts_plot.set_index("Date")[dist_col].sort_index().values,
                                mode="lines+markers",
                                name=label,
                            )
                        ]
                    )

                    if "ok" in effective_cfg:
                        ok_thr = float(effective_cfg["ok"])
                        higher_is_worse = bool(effective_cfg.get("higher_is_worse", True))
                        if higher_is_worse:
                            fig_d.add_hrect(y0=y_min, y1=ok_thr, line_width=0, fillcolor="rgba(34,197,94,0.15)", layer="below")
                            fig_d.add_hrect(y0=ok_thr, y1=y_max, line_width=0, fillcolor="rgba(239,68,68,0.15)", layer="below")
                        else:
                            fig_d.add_hrect(y0=y_min, y1=ok_thr, line_width=0, fillcolor="rgba(239,68,68,0.15)", layer="below")
                            fig_d.add_hrect(y0=ok_thr, y1=y_max, line_width=0, fillcolor="rgba(34,197,94,0.15)", layer="below")

                    try:
                        x_day = pd.to_datetime(pd.to_datetime(chosen_day).date())
                        fig_d.add_vline(x=x_day, line_dash="dot", line_color="gray")
                    except Exception:
                        pass

                    if current_value is not None and isinstance(current_value, (int, float)) and np.isfinite(current_value):
                        try:
                            fig_d.add_hline(y=float(current_value), line_dash="dash", line_color="red")
                        except Exception:
                            pass

                    key_d = "ts_dates_" + hashlib.md5((label + "|" + str(dist_col)).encode()).hexdigest()[:8]
                    fig_d.update_layout(
                        height=380,
                        margin=dict(l=10, r=10, t=10, b=60),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                        xaxis_title="Date",
                        yaxis_title=dist_col,
                    )
                    st.plotly_chart(fig_d, use_container_width=True, key=key_d)

                # Tab 4: Distribution (boxplot over all days)
                with tab_box:
                    df_box = ts_plot[[dist_col]].replace([np.inf, -np.inf], np.nan).dropna()
                    if df_box.empty:
                        st.info("No all-days data available for boxplot.")
                    else:
                        fig_box = px.box(df_box, y=dist_col, points="all")
                        series = df_box[dist_col]
                        yb_min = float(series.min())
                        yb_max = float(series.max())
                        if "ok" in effective_cfg:
                            ok_thr = float(effective_cfg["ok"])
                            higher_is_worse = bool(effective_cfg.get("higher_is_worse", True))
                            if higher_is_worse:
                                if yb_min < ok_thr:
                                    fig_box.add_hrect(y0=yb_min, y1=ok_thr, line_width=0, fillcolor="rgba(34,197,94,0.15)", layer="below")
                                if ok_thr < yb_max:
                                    fig_box.add_hrect(y0=ok_thr, y1=yb_max, line_width=0, fillcolor="rgba(239,68,68,0.15)", layer="below")
                            else:
                                if yb_min < ok_thr:
                                    fig_box.add_hrect(y0=yb_min, y1=ok_thr, line_width=0, fillcolor="rgba(239,68,68,0.15)", layer="below")
                                if ok_thr < yb_max:
                                    fig_box.add_hrect(y0=ok_thr, y1=yb_max, line_width=0, fillcolor="rgba(34,197,94,0.15)", layer="below")

                        if (current_value is not None) and isinstance(current_value, (int, float)) and np.isfinite(current_value):
                            try:
                                fig_box.add_hline(y=float(current_value), line_dash="dash", line_color="red")
                            except Exception:
                                pass

                        key_box = "ts_box_" + hashlib.md5((label + "|" + str(dist_col)).encode()).hexdigest()[:8]
                        fig_box.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                        fig_box.update_xaxes(visible=False)
                        fig_box.update_yaxes(title=None)
                        st.plotly_chart(fig_box, use_container_width=True, key=key_box)
    else:
        st.info("No historical time series available for this metric.")

    # Curated inputs popover (non-technical explanation + lists)
    curated = _curated_for_metric(dist_col, label)
    if curated:
        with st.popover("Curated inputs used", use_container_width=True):
            st.caption("These lists and thresholds are coded into the metric to classify traffic consistently. They are transparent and adjustable.")
            for name, values, why in curated:
                st.markdown(f"**{name}**")
                st.caption(why)
                try:
                    if isinstance(values, (set, list, tuple)):
                        df_vals = pd.DataFrame(sorted(list(values)), columns=[name])
                        st.dataframe(df_vals, use_container_width=True, hide_index=True)
                    else:
                        st.write(values)
                except Exception:
                    st.write(str(values))

    # Metric-specific popovers with concrete examples/lists for today's selection
    def _load_today_df_for_details():
        try:
            frames = []
            for bn in selected_base_names:
                dfi = load_day_dataframe(bn, chosen_day)
                if not dfi.empty:
                    frames.append(dfi)
            if not frames:
                return pd.DataFrame()
            d = pd.concat(frames, ignore_index=True)
            d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="coerce")
            d = enrich_with_hostnames(d)
            return d
        except Exception:
            return pd.DataFrame()

    if dist_col in {"C2_F1_UniqueSLD", "F1_DistinctSocial", "F2_MeanSocialDurSec",
                    "C2_F2_MedianReplyLatencySec","C2_F3_ChatSessionCount","C2_F4_MeanUpstreamRateBps",
                    "C2_F5_PassiveActiveByteRatio","C2_F6_ProductivityHits","C2_F7_SocialOutgoingShareUp",
                    "C3_F1_FoodDeliveryHits","C3_F2_LateNightDeliveryRatio","C3_F3_MeanInterOrderDays",
                    "C3_F4_DietSiteVisits","C3_F5_TrackerBurstCount","C3_F6_SmartScaleUploads","C3_F7_WeighInTimeVarMin",
                    "C5_F1_DhcpPerHour","C5_F3_MedianIKS","C5_F4_IKSStd","C5_F5_Sub30sSessions",
                    "C7_F1_MentalHealthSiteVisitsDay","C7_F4_HelpTherapyLookupHits",
                    "C7_F8_CloudUploadBytesToday"}:
        d_today = _load_today_df_for_details()
        if not d_today.empty:
            if dist_col == "C2_F1_UniqueSLD":
                with st.popover("Today's unique domains (eTLD+1)", use_container_width=True):
                    slds = sorted([s for s in d_today.get("SLD", pd.Series(dtype=str)).dropna().astype(str).unique()])
                    st.caption("Domains are simplified to their registrable roots (e.g., google.com).")
                    st.dataframe(pd.DataFrame(slds, columns=["SLD"]), use_container_width=True, hide_index=True)
            elif dist_col == "F1_DistinctSocial":
                with st.popover("Today's social domains", use_container_width=True):
                    soc = d_today.get("SLD", pd.Series(dtype=str)).dropna().astype(str)
                    slds = sorted(set([s for s in soc if s in SOCIAL_SLDS]))
                    st.caption("Social/chat platforms observed today (matched against the curated list).")
                    st.dataframe(pd.DataFrame(slds, columns=["Social SLD"]), use_container_width=True, hide_index=True)
            elif dist_col == "F2_MeanSocialDurSec":
                with st.popover("What is a social session?", use_container_width=True):
                    st.markdown(
                        "A social session groups messages close in time. We join packets that are at most 5 minutes apart; a longer pause starts a new session. The value shown is the average session duration for today."
                    )
                    try:
                        chat = d_today.loc[chat_mask(d_today)].copy()
                        sess = sessions_from_timestamps(chat, gap_sec=300) if not chat.empty else []
                        st.caption(f"Detected {len(sess)} social session(s) today.")
                        # Show a short preview of session durations
                        rows = []
                        for a, b in sess[:10]:
                            dur = (pd.to_datetime(b) - pd.to_datetime(a)).total_seconds()
                            rows.append({"Start": pd.to_datetime(a), "End": pd.to_datetime(b), "Duration (s)": int(dur)})
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    except Exception:
                        pass
            elif dist_col == "C2_F2_MedianReplyLatencySec":
                with st.popover("How reply latency is defined", use_container_width=True):
                    st.markdown("Time from an inbound chat packet to the very next outbound chat packet, capped at 120 seconds, measured within chat flows.")
                    try:
                        d = d_today.loc[chat_mask(d_today)].copy()
                        d = d.sort_values("Timestamp")
                        d["inb"] = d.apply(lambda r: is_inbound(r.get("Source IP"), r.get("Destination IP")), axis=1)
                        d["outb"] = d.apply(lambda r: is_outbound(r.get("Source IP"), r.get("Destination IP")), axis=1)
                        ts = pd.to_datetime(d["Timestamp"]).values; inn = d["inb"].values; out = d["outb"].values
                        gaps = []
                        for i in range(len(d)-1):
                            if inn[i] and out[i+1]:
                                dt = (ts[i+1] - ts[i]).astype("timedelta64[s]").astype(int)
                                if 0 < dt <= 120:
                                    gaps.append(int(dt))
                        st.caption(f"Reply pairs found: {len(gaps)}")
                        if gaps:
                            st.dataframe(pd.DataFrame(gaps[:20], columns=["Latency (s)"]), use_container_width=True, hide_index=True)
                    except Exception:
                        pass
            elif dist_col == "C2_F3_ChatSessionCount":
                with st.popover("Chat sessions (definition & sample)", use_container_width=True):
                    st.markdown("5‑minute idle gap segmentation on chat traffic (social domains/ports). A new session begins after >5 minutes pause.")
                    try:
                        chat = d_today.loc[chat_mask(d_today)].copy()
                        sess = sessions_from_timestamps(chat, gap_sec=300) if not chat.empty else []
                        st.caption(f"Detected {len(sess)} chat session(s) today.")
                        rows = [{"Start": pd.to_datetime(a), "End": pd.to_datetime(b), "Duration (s)": int((pd.to_datetime(b)-pd.to_datetime(a)).total_seconds())} for a,b in sess[:10]]
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    except Exception:
                        pass
            elif dist_col == "C2_F4_MeanUpstreamRateBps":
                with st.popover("Per‑session upstream rate (bps)", use_container_width=True):
                    st.markdown("Within each chat session, divide total outbound bytes by session duration (seconds), then average across sessions.")
                    try:
                        chat = d_today.loc[chat_mask(d_today)].copy()
                        sess = sessions_from_timestamps(chat, gap_sec=300) if not chat.empty else []
                        rows = []
                        for a,b in sess[:10]:
                            win = chat[(chat["Timestamp"]>=a) & (chat["Timestamp"]<=b)]
                            if win.empty: continue
                            dur = max(1.0, (pd.to_datetime(b)-pd.to_datetime(a)).total_seconds())
                            up = float(win.loc[win.apply(lambda r: is_outbound(r.get("Source IP"), r.get("Destination IP")), axis=1), "Length"].sum())
                            rows.append({"Start": pd.to_datetime(a), "End": pd.to_datetime(b), "Up bytes": int(up), "Dur (s)": int(dur), "Rate (bps)": int(up/dur)})
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    except Exception:
                        pass
            elif dist_col == "C2_F5_PassiveActiveByteRatio":
                with st.popover("Passive vs active bytes", use_container_width=True):
                    st.markdown("Passive = inbound streaming bytes on known streaming domains; Active = outbound chat bytes. Ratio = Passive / Active.")
                    try:
                        Bp = float(d_today.loc[streaming_inbound_mask(d_today), "Length"].sum()) if "Length" in d_today.columns else float("nan")
                        chat = d_today.loc[chat_mask(d_today)]
                        Ba = float(chat.loc[chat.apply(lambda r: is_outbound(r.get("Source IP"), r.get("Destination IP")), axis=1), "Length"].sum()) if (not chat.empty and "Length" in chat.columns) else 0.0
                        st.write({"Passive (bytes)": int(Bp) if pd.notna(Bp) else None, "Active (bytes)": int(Ba)})
                    except Exception:
                        pass
            elif dist_col == "C2_F6_ProductivityHits":
                with st.popover("Productivity domains today", use_container_width=True):
                    slds = d_today.get("SLD", pd.Series(dtype=str)).dropna().astype(str)
                    hits = sorted(set([s for s in slds if s in PRODUCTIVITY_SLDS]))
                    st.dataframe(pd.DataFrame(hits, columns=["Productivity SLD"]), use_container_width=True, hide_index=True)
            elif dist_col == "C2_F7_SocialOutgoingShareUp":
                with st.popover("Social up/down bytes", use_container_width=True):
                    soc = d_today.loc[d_today.get("SLD", pd.Series(dtype=str)).isin(SOCIAL_SLDS)].copy()
                    if soc.empty:
                        soc = d_today.loc[chat_mask(d_today)].copy()
                    if not soc.empty and {"Source IP","Destination IP","Length"}.issubset(soc.columns):
                        up = float(soc.loc[soc.apply(lambda r: is_outbound(r.get("Source IP"), r.get("Destination IP")), axis=1), "Length"].sum())
                        down = float(soc.loc[soc.apply(lambda r: is_inbound(r.get("Source IP"), r.get("Destination IP")), axis=1), "Length"].sum())
                        st.write({"Up bytes": int(up), "Down bytes": int(down)})
            elif dist_col in {"C3_F1_FoodDeliveryHits","C3_F2_LateNightDeliveryRatio","C3_F3_MeanInterOrderDays"}:
                with st.popover("Food‑delivery sessions (5‑min gap)", use_container_width=True):
                    rows = d_today.loc[d_today.get("SLD", pd.Series(dtype=str)).isin(FOOD_DELIVERY_SLDS)].copy()
                    sess = sessions_from_timestamps(rows, gap_sec=300) if not rows.empty else []
                    st.caption(f"Detected {len(sess)} order session(s) today.")
                    if sess:
                        data = []
                        for a,b in sess:
                            data.append({"Start": pd.to_datetime(a), "End": pd.to_datetime(b), "Start hour": pd.to_datetime(a).hour})
                        df_s = pd.DataFrame(data)
                        if dist_col == "C3_F2_LateNightDeliveryRatio":
                            night = int(((df_s["Start hour"] >= 22) | (df_s["Start hour"] < 6)).sum())
                            st.write({"Night sessions": night, "Total": len(sess)})
                        st.dataframe(df_s.head(10), use_container_width=True, hide_index=True)
            elif dist_col == "C3_F4_DietSiteVisits":
                with st.popover("Diet/fitness domains today", use_container_width=True):
                    slds = d_today.get("SLD", pd.Series(dtype=str)).dropna().astype(str)
                    hits = sorted(set([s for s in slds if s in DIET_SLDS]))
                    st.dataframe(pd.DataFrame(hits, columns=["Diet SLD"]), use_container_width=True, hide_index=True)
            elif dist_col == "C3_F5_TrackerBurstCount":
                with st.popover("Calorie‑tracker bursts (10‑min bins)", use_container_width=True):
                    rows = d_today.loc[d_today.get("SLD", pd.Series(dtype=str)).isin(DIET_SLDS)].copy()
                    if not rows.empty and "Timestamp" in rows.columns:
                        rows["bin"] = pd.to_datetime(rows["Timestamp"]).dt.floor("10min")
                        counts = rows.groupby(["SLD","bin"]).size().reset_index(name="cnt")
                        bursts = counts[counts["cnt"] >= TRACKER_BURST_THRESHOLD].sort_values(["bin","cnt"], ascending=[True,False])
                        st.caption(f"Threshold: {TRACKER_BURST_THRESHOLD} events per 10‑min bin")
                        if not bursts.empty:
                            st.dataframe(bursts.head(10), use_container_width=True, hide_index=True)
            elif dist_col in {"C3_F6_SmartScaleUploads","C3_F7_WeighInTimeVarMin"}:
                with st.popover("Smart‑scale events", use_container_width=True):
                    rows = d_today.loc[d_today.get("SLD", pd.Series(dtype=str)).isin(SMART_SCALE_SLDS)].copy()
                    if {"Source IP","Destination IP","Timestamp"}.issubset(rows.columns):
                        rows = rows[rows.apply(lambda r: is_outbound(r.get("Source IP"), r.get("Destination IP")), axis=1)]
                    st.dataframe(rows[["Timestamp","SLD"]].head(10) if not rows.empty else pd.DataFrame([{"Info":"No events"}]), use_container_width=True, hide_index=True)
            elif dist_col == "C5_F1_DhcpPerHour":
                with st.popover("How DHCP is detected", use_container_width=True):
                    st.markdown("UDP packets on ports 67/68 (IPv4) or 546/547 (IPv6). We compute events per hour over the day span.")
            elif dist_col in {"C5_F3_MedianIKS","C5_F4_IKSStd"}:
                with st.popover("Inter‑keystroke gaps (proxy)", use_container_width=True):
                    st.markdown("Estimated from chat outbound timestamps: we measure time gaps between consecutive outbound chat packets under 3 seconds.")
                    try:
                        chat = d_today.loc[chat_mask(d_today)].copy()
                        if not chat.empty:
                            chat = chat.sort_values("Timestamp")
                            out_mask = chat.apply(lambda r: is_outbound(r.get("Source IP"), r.get("Destination IP")), axis=1)
                            tt = pd.to_datetime(chat.loc[out_mask, "Timestamp"]).diff().dt.total_seconds().dropna()
                            tt = tt[(tt > 0) & (tt <= 3.0)]
                            st.dataframe(tt.head(20).to_frame("Gap (s)"), use_container_width=True, hide_index=True)
                    except Exception:
                        pass
            elif dist_col == "C5_F5_Sub30sSessions":
                with st.popover("Sessions < 30s (gap 5‑min)", use_container_width=True):
                    try:
                        sess = sessions_from_timestamps(d_today.sort_values("Timestamp"), gap_sec=300)
                        rows = []
                        for a,b in (sess[:20] if sess else []):
                            dur = (pd.to_datetime(b)-pd.to_datetime(a)).total_seconds()
                            if dur < 30.0:
                                rows.append({"Start": pd.to_datetime(a), "End": pd.to_datetime(b), "Dur (s)": int(dur)})
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        else:
                            st.caption("No sub‑30s sessions found in sample.")
                    except Exception:
                        pass
            elif dist_col in {"C7_F1_MentalHealthSiteVisitsDay","C7_F4_HelpTherapyLookupHits"}:
                with st.popover("Matched domains today", use_container_width=True):
                    slds = d_today.get("SLD", pd.Series(dtype=str)).dropna().astype(str)
                    if dist_col == "C7_F1_MentalHealthSiteVisitsDay":
                        hits = sorted(set([s for s in slds if s in MENTAL_HEALTH_SLDS]))
                    else:
                        hits = sorted(set([s for s in slds if (s in CRISIS_SLDS or s in THERAPY_SLDS)]))
                    st.dataframe(pd.DataFrame(hits, columns=["SLD"]), use_container_width=True, hide_index=True)
            elif dist_col == "C7_F8_CloudUploadBytesToday":
                with st.popover("Cloud providers and uploads", use_container_width=True):
                    rows = d_today.loc[d_today.get("SLD", pd.Series(dtype=str)).isin(CLOUD_STORAGE_SLDS)].copy()
                    up = float(rows.loc[rows.apply(lambda r: is_outbound(r.get("Source IP"), r.get("Destination IP")), axis=1), "Length"].sum()) if {"Source IP","Destination IP","Length"}.issubset(rows.columns) else float("nan")
                    st.write({"Matched providers": sorted(rows.get("SLD", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()), "Up bytes": (int(up) if pd.notna(up) else None)})

    if ranges_str:
        st.caption(ranges_str)
    if explanation:
        st.markdown(explanation)
    else:
        st.markdown(
            f"This metric (**{label}**) is derived from network activity and proxies "
            f"behavioral aspects relevant to the DSM‑5 criterion on this tab. "
            f"Interpret it relative to the baseline (green) and caution ranges (orange)."
        )

    if latex_formula:
        st.latex(latex_formula)
    if latex_numbers:
        st.latex(latex_numbers)

    if (current_value is None) or (isinstance(current_value, float) and (np.isnan(current_value) or not np.isfinite(current_value))):
        st.info("Not enough data or missing variables for today's value.")

    st.markdown("---")
    with st.popover("Adjust status ranges", use_container_width=True):
        default_ok = float(effective_cfg.get("ok")) if "ok" in effective_cfg else 0.0
        default_hiw = bool(effective_cfg.get("higher_is_worse", True))
        new_ok = st.number_input("OK threshold", value=default_ok, help="Boundary between OK and Caution.")
        new_hiw = st.checkbox("Higher values are worse", value=default_hiw)

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Apply to this metric"):
                key = f"{label}|{dist_col}"
                overrides = st.session_state.get("__range_overrides__", {})
                overrides[key] = {"ok": float(new_ok), "higher_is_worse": bool(new_hiw)}
                st.session_state["__range_overrides__"] = overrides
                st.rerun()
        with c2:
            if st.button("Reset to default"):
                key = f"{label}|{dist_col}"
                overrides = st.session_state.get("__range_overrides__", {})
                if key in overrides:
                    del overrides[key]
                    st.session_state["__range_overrides__"] = overrides
                st.rerun()

def _summarize_status_counts(metrics: list[dict], selected_metric_labels: list[str]) -> dict:
    counts = {"OK": 0, "Caution": 0, "N/A": 0}
    for m in metrics:
        if selected_metric_labels and (m.get("label") not in selected_metric_labels):
            continue
        eff_cfg = get_effective_range_cfg(m.get("label"), m.get("dist_col"), m.get("range_cfg"))
        stt = status_from_value(m.get("value"), eff_cfg, m["status_tuple"][0])
        if stt not in counts:
            stt = "N/A"
        counts[stt] += 1
    return counts

def _render_gauge(col, value: int, max_value: int, title: str, color_hex: str, key: str):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [0, max_value]},
                "bar": {"color": color_hex},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#e5e7eb",
            },
            number={"valueformat": "d"},
        )
    )
    fig.update_layout(height=180, margin=dict(l=40, r=40, t=40, b=10))
    col.plotly_chart(fig, use_container_width=True, key=key)

def _render_status_gauges(metrics: list[dict], selected_metric_labels: list[str], key_prefix: str):
    counts = _summarize_status_counts(metrics, selected_metric_labels)
    max_value = len(metrics)
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        _render_gauge(st, counts.get("OK", 0), max_value, "OK", "#16a34a", key=f"{key_prefix}_g_ok")
    with c2.container(border=True):
        _render_gauge(st, counts.get("Caution", 0), max_value, "Caution", "#f59e0b", key=f"{key_prefix}_g_caution")
    with c3.container(border=True):
        _render_gauge(st, counts.get("N/A", 0), max_value, "N/A", "#2563eb", key=f"{key_prefix}_g_na")

def render_metric_grid(
    metric_items: list[dict],
    selected_statuses: set[str],
    all_days_df: pd.DataFrame,
    selected_metric_labels: list[str] | None = None,
    key_prefix: str = "",
):
    """Render KPI cards 2 per row with boxplot, status, and a Details dialog."""
    filtered = []
    for m in metric_items:
        label = m.get("label")
        dist_col = m.get("dist_col")
        base_cfg = m.get("range_cfg") or {}
        eff_cfg = get_effective_range_cfg(label, dist_col, base_cfg)
        disp_status = status_from_value(m.get("value"), eff_cfg, m["status_tuple"][0])
        if disp_status in selected_statuses:
            if (selected_metric_labels is None) or (label in selected_metric_labels):
                filtered.append(m)

    filtered.sort(
        key=lambda m: STATUS_ORDER.get(
            status_from_value(m.get("value"), get_effective_range_cfg(m.get("label"), m.get("dist_col"), m.get("range_cfg")), m["status_tuple"][0]),
            99,
        )
    )

    for i in range(0, len(filtered), 2):
        cols = st.columns(2, vertical_alignment="top")
        for j, m in enumerate(filtered[i : i + 2]):
            cont = cols[j].container(border=True)
            render_kpi(
                cont,
                m["label"],
                m["value"],
                m["fmt"],
                m["status_tuple"],
                m["ranges_str"],
                m.get("latex_formula"),
                m.get("latex_numbers"),
                m.get("heuristic_md"),
                m.get("missing_md"),
                dist_df=all_days_df,
                dist_col=m.get("dist_col"),
                current_value=m["value"],
                range_cfg=m.get("range_cfg"),
                key_prefix=f"{key_prefix}_m{i+j}",
            )

def render_kpi(
    col,
    label,
    value,
    fmt,
    original_status_tuple,
    ranges_str,
    latex_formula=None,
    latex_numbers=None,
    heuristic_md=None,
    missing_md=None,
    dist_df: pd.DataFrame | None = None,
    dist_col: str | None = None,
    current_value=None,
    range_cfg: dict | None = None,
    key_prefix: str = "",
):
    with col:
        st.markdown(f"**{label}**")
        inner = st.columns([1, 1], vertical_alignment="top")

        # Determine if we have distribution data
        has_dist = False
        df_plot = None
        if dist_df is not None and dist_col is not None and dist_col in dist_df.columns:
            df_plot = dist_df[[dist_col]].replace([np.inf, -np.inf], np.nan).dropna()
            has_dist = not df_plot.empty

        # Effective range config (with overrides applied)
        effective_cfg = get_effective_range_cfg(label, dist_col, range_cfg or {})
        display_status = status_from_value(value, effective_cfg, original_status_tuple[0])

        with inner[0]:
            if value is None or (isinstance(value, float) and (np.isnan(value) or not np.isfinite(value))):
                st.metric(" ", "N/A", label_visibility="collapsed")
                badge("N/A", color="blue", icon=":material/info:")
            else:
                try:
                    st.metric(" ", fmt(value), label_visibility="collapsed")
                except Exception:
                    st.metric(" ", str(value), label_visibility="collapsed")

                if not has_dist:
                    badge("N/A", color="blue", icon=":material/info:")
                else:
                    if display_status == "OK":
                        badge("OK", color="green", icon=":material/check_circle:")
                    elif display_status == "Caution":
                        badge("Caution", color="orange", icon=":material/priority_high:")
                    else:
                        badge("N/A", color="blue", icon=":material/info:")

            # Details button
            btn_key = f"{key_prefix}_details_{hashlib.md5((label + '|' + str(dist_col)).encode()).hexdigest()[:8]}"
            if st.button("Details", key=btn_key):
                ts_df = None
                if dist_df is not None and dist_col is not None and dist_col in dist_df.columns and "Date" in dist_df.columns:
                    ts_df = dist_df[["Date", dist_col]].copy()
                st.session_state["__metric_dialog_payload__"] = {
                    "label": label,
                    "ranges_str": ranges_str,
                    "latex_formula": latex_formula,
                    "latex_numbers": latex_numbers,
                    "explanation_md": heuristic_md,
                    "dist_col": dist_col,
                    "range_cfg": range_cfg,
                    "current_value": current_value,
                    "ts_df": ts_df,
                }
                _show_metric_dialog()

        with inner[1]:
            if not has_dist:
                st.info("No all-days data.")
            else:
                fig_box = px.box(df_plot, y=dist_col, points="all")
                series = df_plot[dist_col]
                y_min = float(series.min())
                y_max = float(series.max())

                if "ok" in effective_cfg:
                    ok_thr = float(effective_cfg["ok"])
                    higher_is_worse = bool(effective_cfg.get("higher_is_worse", True))
                    if higher_is_worse:
                        if y_min < ok_thr:
                            fig_box.add_hrect(y0=y_min, y1=ok_thr, line_width=0, fillcolor="rgba(34,197,94,0.15)", layer="below")
                        if ok_thr < y_max:
                            fig_box.add_hrect(y0=ok_thr, y1=y_max, line_width=0, fillcolor="rgba(239,68,68,0.15)", layer="below")
                    else:
                        if y_min < ok_thr:
                            fig_box.add_hrect(y0=y_min, y1=ok_thr, line_width=0, fillcolor="rgba(239,68,68,0.15)", layer="below")
                        if ok_thr < y_max:
                            fig_box.add_hrect(y0=ok_thr, y1=y_max, line_width=0, fillcolor="rgba(34,197,94,0.15)", layer="below")

                if (current_value is not None) and isinstance(current_value, (int, float)) and np.isfinite(current_value):
                    try:
                        fig_box.add_hline(y=float(current_value), line_dash="dash", line_color="red")
                    except Exception:
                        pass

                box_key = f"{key_prefix}_box_{hashlib.md5((label + '|' + str(dist_col)).encode()).hexdigest()[:8]}"
                fig_box.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                fig_box.update_xaxes(visible=False)
                fig_box.update_yaxes(title=None)
                st.plotly_chart(fig_box, use_container_width=True, key=box_key)

        if not has_dist:
            st.info(
                "This metric is shown as **N/A** for status because the all‑days cache has no values yet. "
                "Use **Recompute all-days metric cache** if needed."
            )

# =============================== Tabs / Rendering =============================

st.write("---")
st.subheader("Network Traffic Metrics mapped to DSM‑5 Indicators")
tabs = st.tabs([f"Criterion {i}" for i in range(1, 10)])

def compute_and_render(tab_index: int, title: str, caption: str):
    with tabs[tab_index]:
        st.subheader(title)
        st.caption(caption)

        metrics = metrics_by_tab[tab_index]
        key_prefix = f"c{tab_index+1}"

        # Popover: status & metric name filters
        with st.popover("Metric filters", use_container_width=True):
            selected_statuses = metric_filter_ui(key_prefix)
            metric_labels = [m["label"] for m in metrics]
            selected_metric_labels = st.multiselect(
                "Select metrics",
                options=metric_labels,
                default=metric_labels,
                key=f"{key_prefix}_metric_names",
                help="Choose which KPIs to display. By default, all metrics are selected.",
            )

        # Gauges (OK / Caution / N/A)
        counts = {"OK": 0, "Caution": 0, "N/A": 0}
        for m in metrics:
            eff_cfg = get_effective_range_cfg(m.get("label"), m.get("dist_col"), m.get("range_cfg"))
            stt = status_from_value(m.get("value"), eff_cfg, m["status_tuple"][0])
            counts[stt] = counts.get(stt, 0) + 1
        _ = counts  # (left gauges out to reduce clutter, keep logic if you re-add)

        # Metric cards/grid
        render_metric_grid(metrics, selected_statuses, ALL_DAILY, selected_metric_labels, key_prefix=key_prefix)

compute_and_render(0, "Criterion 1 — Sleep disturbance", "Insomnia or hypersomnia, nearly every day.")
compute_and_render(1, "Criterion 2 — Loss of interest / anhedonia", "Markedly diminished interest or pleasure.")
compute_and_render(2, "Criterion 3 — Appetite / weight change", "Significant weight loss/gain or appetite change.")
compute_and_render(3, "Criterion 4 — Sleep timing & duration", "Insomnia or hypersomnia proxies.")
compute_and_render(4, "Criterion 5 — Psychomotor agitation/retardation", "Observable agitation or slowing.")
compute_and_render(5, "Criterion 6 — Fatigue / low energy", "Fatigue or loss of energy, nearly every day.")
compute_and_render(6, "Criterion 7 — Worthlessness / guilt", "Feelings of worthlessness or excessive/inappropriate guilt.")
compute_and_render(7, "Criterion 8 — Difficulty concentrating / indecisiveness", "Diminished ability to think or concentrate; indecisiveness.")
compute_and_render(8, "Criterion 9 — Suicidality", "Recurrent thoughts of death or suicidal ideation.")
