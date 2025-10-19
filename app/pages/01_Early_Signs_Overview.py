# pages/01_FASL_DSM_Gate.py

from __future__ import annotations

import os, re, json, hashlib, copy, math
import html
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from utils.acronyms import render_acronyms_helper_in_sidebar

from metrics.base_features import compute_daily_base_record
from metrics.common import enrich_with_hostnames
from md_explanations import MD_EXPLANATIONS
from utils.auto_tune import auto_tune_for_criterion, AutoTuneResult
from metrics.criterion1 import C1_DEFS, Criterion1
from metrics.criterion2 import C2_DEFS, Criterion2
from metrics.criterion3 import C3_DEFS, Criterion3
from metrics.criterion4 import C4_DEFS, Criterion4
from metrics.criterion5 import C5_DEFS, Criterion5
from metrics.criterion6 import C6_DEFS, Criterion6
from metrics.criterion7 import C7_DEFS, Criterion7
from metrics.criterion8 import C8_DEFS, Criterion8
from metrics.criterion9 import C9_DEFS, Criterion9

CRIT_KEYS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

CRITERION_CLASSES = (
    Criterion1,
    Criterion2,
    Criterion3,
    Criterion4,
    Criterion5,
    Criterion6,
    Criterion7,
    Criterion8,
    Criterion9,
)

EXPECTED_METRIC_COLUMNS: set[str] = {
    d.dist_col
    for defs in (
        C1_DEFS,
        C2_DEFS,
        C3_DEFS,
        C4_DEFS,
        C5_DEFS,
        C6_DEFS,
        C7_DEFS,
        C8_DEFS,
        C9_DEFS,
    )
    for d in defs
}

# ------------------------------ Page header / Auth -----------------------------

st.set_page_config(
    page_title="CareNet - Nef, Stephan",
    page_icon="https://upload.wikimedia.org/wikipedia/de/thumb/7/77/Uni_St_Gallen_Logo.svg/2048px-Uni_St_Gallen_Logo.svg.png",
    layout="wide",
)

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


# Sidebar: helper dialog just below the page selector
render_acronyms_helper_in_sidebar()

col1, col2 = st.columns([7, 2])
with col1:
    st.title("Early Signs Overview")
    st.caption(
        """
        This is a research prototype and not a medical device, Stephan Nef
        """
    )
with col2:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/HSG_Logo_DE_RGB.svg/1024px-HSG_Logo_DE_RGB.svg.png"
    )



# ------------------------------ Basics / Paths --------------------------------

APP_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = APP_DIR / "processed_parquet"
FEATURE_CACHE_DIR = APP_DIR / "feature_cache"
FASL_CONFIG_PATH = APP_DIR / "fasl_config.json"


# ------------------------------ Data loading ----------------------------------

@st.cache_data(show_spinner=False)
def list_partition_files(base_name: str) -> list[str]:
    d = os.path.join(PROCESSED_DIR, base_name)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".parquet"))


def partition_file_to_start_dt(path: str):
    m = re.search(r"__(\d{8})_(\d{4})\.parquet$", os.path.basename(path))
    if not m:
        return None
    datestr, timestr = m.groups()
    try:
        return pd.to_datetime(datestr + timestr, format="%Y%m%d%H%M")
    except Exception:
        return None


def load_day_dataframe(base_name: str, day) -> pd.DataFrame:
    day = pd.to_datetime(day).normalize()
    next_day = day + pd.Timedelta(days=1)
    chosen = []
    for p in list_partition_files(base_name):
        dt = partition_file_to_start_dt(p)
        if dt is not None and (day <= dt) and (dt < next_day):
            chosen.append(p)
    if not chosen:
        return pd.DataFrame(columns=["Timestamp"])
    dfs: List[pd.DataFrame] = []
    for fp in chosen:
        try:
            df = pd.read_parquet(fp)
        except Exception:
            try:
                import pyarrow.parquet as pq, pyarrow as pa

                pf = pq.ParquetFile(fp)
                tabs = []
                for i in range(pf.num_row_groups):
                    try:
                        tabs.append(pf.read_row_group(i))
                    except Exception:
                        pass
                if not tabs:
                    continue
                df = pa.concat_tables(tabs, promote=True).to_pandas()
            except Exception:
                continue
        # derive Timestamp if only Date/Hour present
        if "Timestamp" not in df.columns and {"Date", "Hour"}.issubset(df.columns):
            df["Timestamp"] = pd.to_datetime(df["Date"].astype(str)) + pd.to_timedelta(
                df["Hour"], unit="h"
            )
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["Timestamp"])
    out = pd.concat(dfs, ignore_index=True)
    out = out[(out["Timestamp"] >= day) & (out["Timestamp"] < next_day)].copy()
    out["Date"] = out["Timestamp"].dt.date
    out["Hour"] = out["Timestamp"].dt.hour
    out = enrich_with_hostnames(out)
    return out


@st.cache_data(show_spinner=False)
def available_days_for(base_names: list[str]) -> list[pd.Timestamp]:
    days = set()
    for bn in base_names:
        for p in list_partition_files(bn):
            dt = partition_file_to_start_dt(p)
            if dt is not None:
                days.add(pd.to_datetime(dt.date()))
    return sorted(days)


@st.cache_data(show_spinner=False)
def compute_all_daily(base_names: list[str], days: list[pd.Timestamp]) -> pd.DataFrame:
    if not base_names or not days:
        return pd.DataFrame(columns=["Date"])

    rows: List[dict] = []
    day_list = sorted({pd.to_datetime(d).normalize() for d in days})

    for d in day_list:
        frames = []
        for bn in base_names:
            df_b = load_day_dataframe(bn, d)
            if not df_b.empty:
                frames.append(df_b)
        if not frames:
            continue

        df_day = pd.concat(frames, ignore_index=True)
        df_day["Timestamp"] = pd.to_datetime(df_day["Timestamp"], errors="coerce")
        df_day = df_day.dropna(subset=["Timestamp"]).sort_values("Timestamp")

        today = compute_daily_base_record(df_day)
        today["Date"] = pd.to_datetime(d).normalize()

        history_df = pd.DataFrame(rows).sort_values("Date") if rows else pd.DataFrame(columns=["Date"])
        aux_ctx = {
            "today_row": today,
            "ALL_DAILY": history_df,
            "IS_val": today.get("IS"),
            "IV_val": today.get("IV"),
            "nd_ratio": today.get("ND_Ratio"),
            "n_total_packets_today": today.get("n_total_packets_today"),
            "n_night_packets_today": today.get("n_night_packets_today"),
            "night_pkts_today": today.get("night_pkts_today"),
            "day_pkts_today": today.get("day_pkts_today"),
            "down_bytes_today": today.get("down_bytes_today"),
            "up_bytes_today": today.get("up_bytes_today"),
        }

        for crit_cls in CRITERION_CLASSES:
            try:
                metrics = crit_cls().compute(df_day, today, aux_ctx, history_df)
            except Exception:
                continue
            if not metrics:
                continue
            for m in metrics:
                dist_col = m.get("dist_col")
                if dist_col:
                    today[dist_col] = m.get("value")

        rows.append(today)

    if not rows:
        return pd.DataFrame(columns=["Date"])
    df_out = pd.DataFrame(rows).sort_values("Date")
    return df_out


# --------------------------- Membership functions -----------------------------


def mf_tri(
    x: float | None, lo: float, mid: float, hi: float, invert: bool = False
) -> float:
    if x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
        return 0.0
    # Support infinite endpoints to emulate shoulder shapes
    if lo == mid == hi:
        return 1.0
    if not np.isfinite(lo) and not np.isfinite(hi):
        val = 1.0  # everywhere
    elif not np.isfinite(lo):
        # Left-open: 1 up to mid, then down to 0 at hi
        if x <= mid:
            val = 1.0
        elif x >= hi:
            val = 0.0
        else:
            val = (hi - x) / max(1e-9, (hi - mid))
    elif not np.isfinite(hi):
        # Right-open: 0 up to lo, then up to 1 at mid, 1 afterwards
        if x >= mid:
            val = 1.0
        elif x <= lo:
            val = 0.0
        else:
            val = (x - lo) / max(1e-9, (mid - lo))
    else:
        if x <= lo or x >= hi:
            val = 0.0
        elif x == mid:
            val = 1.0
        elif x < mid:
            val = (x - lo) / max(1e-9, (mid - lo))
        else:
            val = (hi - x) / max(1e-9, (hi - mid))
    return float(1.0 - val if invert else val)


def mf_clip01(x: float | None) -> float:
    try:
        return float(np.clip(float(x), 0.0, 1.0))
    except Exception:
        return 0.0


def fasl_score(values: Dict[str, float], spec: Dict[str, Any]) -> float:
    # spec: { metric_key: {w, mf: {type: 'tri', lo, mid, hi, invert}} }
    total = 0.0
    wsum = 0.0
    for k, cfg in (spec or {}).items():
        w = float(cfg.get("w", 0.0))
        mf_cfg = cfg.get("mf", {})
        mft = (mf_cfg.get("type") or "tri").lower()
        invert = bool(mf_cfg.get("invert", False))
        x = values.get(k)
        if mft == "tri":
            lo = float(mf_cfg.get("lo", 0.0))
            mid = float(mf_cfg.get("mid", 0.0))
            hi = float(mf_cfg.get("hi", 0.0))
            mu = mf_tri(x, lo, mid, hi, invert=invert)
        else:
            mu = mf_clip01(x)
        total += w * mu
        wsum += w
    return float(np.clip(total if wsum <= 0 else total, 0.0, 1.0))


def gate_present(series: pd.Series, theta: float, need_days: int, window: int) -> bool:
    s = pd.Series(series).dropna().astype(float).tail(window)
    if s.empty:
        return False
    return int((s >= float(theta)).sum()) >= int(need_days)


# ------------------------------- UI / Defaults -------------------------------

with st.expander("🤓 How it works", expanded=False):
    how_md_path = Path(__file__).with_name("how_it_works_en.md")
    _md = how_md_path.read_text(encoding="utf-8")
    try:
        _md = re.sub(r"^> \*\*Data quality \(DQI\)\*\*:[^\n]*\n", "", _md, flags=re.M)
        # Remove the entire 'Second example: Anhedonia' section up to the next horizontal rule
        _md = _md.replace("$\\color{#ff7f0e}{L_k}", "\\color{#ff7f0e}{L_k}")
        _md = _md.replace(")$\n$$", ")\n$$")
    except Exception:
        pass
    st.markdown(_md, unsafe_allow_html=True)


def dataset_type(name: str) -> str:
    n = name.lower()
    if "onu" in n:
        return "ONU"
    if "bras" in n:
        return "BRAS"
    return "Other"


def group_prefix(name: str) -> str:
    return re.sub(r"([_-]?\d+)$", "", name)


def group_token_from_prefix(prefix: str) -> str:
    s = os.path.basename(prefix).lower()
    s = re.sub(r"^(onu_|bras_|other_)", "", s)
    s = re.sub(r"^capture_", "", s)
    s = re.sub(r"^[_-]+", "", s)
    return s


# Initialize configuration state early (used by sidebar gate controls)
cfg_state = st.session_state.setdefault("fasl_cfg", {})
cfg_state.setdefault("M", 14)
cfg_state.setdefault("N", 10)
cfg_state.setdefault("theta", 0.7)
cfg_state.setdefault("core_symptoms", ["C2"])

# Auto-load default FASL config once per session when opening this page
try:
    if not st.session_state.get("fasl_config_autoload_done", False):
        default_cfg_path = (Path(__file__).resolve().parent.parent / "utils" / "fasl_config_20250912_0946.json")
        if default_cfg_path.exists():
            raw_cfg = json.loads(default_cfg_path.read_text(encoding="utf-8"))
            # Use the same normalization as the uploader to align schema
            # _normalize_uploaded_config is defined later; try to call it if available
            norm_fn = globals().get("_normalize_uploaded_config")
            loaded_cfg = norm_fn(raw_cfg) if callable(norm_fn) else raw_cfg

            # Overwrite current state with loaded config
            cfg_state.clear()
            if isinstance(loaded_cfg, dict):
                cfg_state.update(loaded_cfg)
            # Ensure gate defaults exist even if not present in file
            cfg_state.setdefault("M", 14)
            cfg_state.setdefault("N", 10)
            cfg_state.setdefault("theta", 0.7)
            cfg_state.setdefault("core_symptoms", ["C2"])
            # Seed widget state so sidebar uses loaded values immediately
            st.session_state["fasl_gate_M"] = int(cfg_state.get("M", 14))
            st.session_state["fasl_gate_N"] = int(cfg_state.get("N", 10))
            st.session_state["fasl_gate_theta"] = float(cfg_state.get("theta", 0.7))
            st.session_state["fasl_gate_core"] = list(cfg_state.get("core_symptoms", ["C2"]))
            # Do not set per-metric widget keys here; their 'value=' params read from cfg_state on first render
            try:
                st.toast("Loaded default FASL Config. You can adjust it in the Configuration or also upload your own config.", icon="✅")
            except Exception:
                pass
        st.session_state["fasl_config_autoload_done"] = True
except Exception as _e:
    # Fail silently; fall back to defaults
    st.session_state["fasl_config_autoload_done"] = True

# Dataset selection (same style as DSM5 dashboard)
with st.sidebar:
    st.header("Datasets")
    with st.status("Scanning datasets…", expanded=False):
        all_datasets = [
            d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))
        ]
        all_datasets = sorted(set(all_datasets))
    selected_types = st.multiselect(
        "Filter by dataset type",
        options=["ONU", "BRAS", "Other"],
        default=["ONU", "BRAS", "Other"],
        key="fasl_filter_types",
    )


def _type_filter(name: str) -> bool:
    return dataset_type(name) in selected_types


filtered = [d for d in all_datasets if _type_filter(d)]

# Build group mapping by common prefix
token_to_dsets: dict[str, set[str]] = {}
for name in filtered:
    pref = group_prefix(name)
    tok = group_token_from_prefix(pref)
    token_to_dsets.setdefault(tok, set()).add(name)

token_options = sorted(token_to_dsets.keys())
quick = ["[ALL]", "[ALL ONU]", "[ALL BRAS]", "[ALL OTHER]"]
group_display_options = quick + token_options

with st.sidebar:
    use_feature_cache = st.toggle(
        "Select feature cache",
        value=bool(st.session_state.get("fasl_use_feature_cache", False)),
        key="fasl_use_feature_cache",
        help="Switch between loading precomputed feature cache snapshots or rebuilding from raw datasets.",
    )
    try:
        cache_dir = FEATURE_CACHE_DIR
        cache_files = sorted(
            p.name for p in cache_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"
        )
    except Exception:
        cache_files = []
    selected_cache_file = ""
    selected_group_tokens: list[str] = []
    if use_feature_cache:
        selected_cache_file = st.selectbox(
            "Feature cache snapshot",
            options=cache_files,
            index=cache_files.index(st.session_state.get("fasl_feature_cache_file", "")) if st.session_state.get("fasl_feature_cache_file", "") in cache_files else 0 if cache_files else -1,
            key="fasl_feature_cache_file",
            help="Load a precomputed ALL_DAILY snapshot from feature_cache.",
        ) if cache_files else ""
        if not cache_files:
            st.caption("No feature cache snapshots found.")
        selected_cache_files = [selected_cache_file] if selected_cache_file else []
    else:
        selected_group_tokens = st.multiselect(
            "Select dataset groups (prefix match)",
            options=group_display_options,
            key="fasl_group_tokens",
            default=["[ALL OTHER]"],
        )
        selected_cache_files = []

auto_selected_from_groups: set[str] = set()
if not use_feature_cache:
    if "[ALL]" in selected_group_tokens:
        auto_selected_from_groups |= set(filtered)
    if "[ALL ONU]" in selected_group_tokens:
        auto_selected_from_groups |= {d for d in filtered if dataset_type(d) == "ONU"}
    if "[ALL BRAS]" in selected_group_tokens:
        auto_selected_from_groups |= {d for d in filtered if dataset_type(d) == "BRAS"}
    if "[ALL OTHER]" in selected_group_tokens:
        auto_selected_from_groups |= {d for d in filtered if dataset_type(d) == "Other"}
    for tok in selected_group_tokens:
        if tok in quick:
            continue
        auto_selected_from_groups |= token_to_dsets.get(tok, set())

selected_base_names = sorted(auto_selected_from_groups)
if not use_feature_cache and not selected_base_names:
    st.info("No datasets selected. Choose dataset type(s) and group(s) in the sidebar.")
    st.stop()
if use_feature_cache and not selected_cache_files:
    st.info("Select a feature cache snapshot to continue.")
    st.stop()

refresh_requested = False
with st.sidebar:
    st.header("Window & Gate")
    gate_window = st.number_input(
        "M: rolling window (days)",
        min_value=7,
        max_value=60,
        value=int(cfg_state.get("M", 14)),
        key="fasl_gate_M",
        step=1,
        help="Look-back horizon in days to evaluate each criterion's daily likelihood L_k.",
    )
    gate_need = st.number_input(
        "N: days ≥ θ",
        min_value=1,
        max_value=60,
        value=int(cfg_state.get("N", 10)),
        key="fasl_gate_N",
        step=1,
        help="Minimum number of days within the last M days where L_k ≥ θ to mark a criterion as present.",
    )
    theta_default = st.slider(
        "θ: criterion present threshold",
        min_value=0.0,
        max_value=0.95,
        value=float(cfg_state.get("theta", 0.7)),
        key="fasl_gate_theta",
        step=0.05,
        help="Daily likelihood threshold. If L_k ≥ θ on N days within M, the criterion is present.",
    )
    core_criteria = st.multiselect(
        "Core symptoms",
        options=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
        default=cfg_state.get("core_symptoms", ["C2"]),
        key="fasl_gate_core",
        help="Select which criteria count as core symptoms; at least one must be present for an episode.",
    )
    cfg_state["M"] = int(gate_window)
    cfg_state["N"] = int(gate_need)
    cfg_state["theta"] = float(theta_default)
    cfg_state["core_symptoms"] = core_criteria
    refresh_requested = st.button(
        "Refresh cached day metrics",
        key="fasl_refresh_cache",
        help="Drop cached ALL_DAILY data and rebuild from source files.",
    )

def _cache_key_for_selection(items: list[str]) -> str:
    return hashlib.md5("|".join(sorted(items)).encode("utf-8")).hexdigest()


def _cache_path_for_selection(base_names: list[str]) -> str:
    return os.path.join(
        FEATURE_CACHE_DIR, f"features_{_cache_key_for_selection(base_names)}.csv"
    )


if refresh_requested:
    st.session_state["fasl_force_refresh"] = True

selection_signature = selected_base_names + [f"[cache]{name}" for name in selected_cache_files]
selection_key = _cache_key_for_selection(selection_signature)

days_cache: dict[str, list[pd.Timestamp]] = st.session_state.setdefault("fasl_days_cache", {})
daily_cache: dict[str, pd.DataFrame] = st.session_state.setdefault("fasl_all_daily_cache", {})
source_map: dict[str, dict[str, Any]] = st.session_state.setdefault("fasl_daily_source_meta", {})

if st.session_state.pop("fasl_force_refresh", False):
    days_cache.pop(selection_key, None)
    daily_cache.pop(selection_key, None)
    source_map.pop(selection_key, None)

cache_load_warnings: list[str] = []
if selected_cache_files:
    cache_frames: list[pd.DataFrame] = []
    for fname in selected_cache_files:
        fpath = FEATURE_CACHE_DIR / fname
        if not fpath.exists():
            cache_load_warnings.append(f"Feature cache '{fname}' not found; skipped.")
            continue
        try:
            df_cache = pd.read_csv(fpath, parse_dates=["Date"])
        except Exception:
            try:
                df_cache = pd.read_csv(fpath)
            except Exception:
                cache_load_warnings.append(f"Failed to load feature cache '{fname}'.")
                continue
            if "Date" in df_cache.columns:
                df_cache["Date"] = pd.to_datetime(df_cache["Date"], errors="coerce")
        if "Date" not in df_cache.columns:
            cache_load_warnings.append(f"Feature cache '{fname}' has no Date column; skipped.")
            continue
        df_cache["Date"] = pd.to_datetime(df_cache["Date"], errors="coerce")
        df_cache = df_cache.dropna(subset=["Date"])
        if df_cache.empty:
            cache_load_warnings.append(f"Feature cache '{fname}' contains no dated rows; skipped.")
            continue
        cache_frames.append(df_cache)
    if cache_frames:
        merged_cache = (
            pd.concat(cache_frames, ignore_index=True)
            .sort_values("Date")
            .drop_duplicates(subset=["Date"], keep="last")
        )
        merged_cache["Date"] = pd.to_datetime(merged_cache["Date"], errors="coerce")
        merged_cache = merged_cache.dropna(subset=["Date"]).sort_values("Date")
        if not merged_cache.empty:
            day_series = merged_cache["Date"].dt.normalize()
            days_cache[selection_key] = day_series.drop_duplicates().tolist()
            daily_cache[selection_key] = merged_cache.copy()
            source_map[selection_key] = {"type": "feature_cache", "files": list(selected_cache_files)}
    for msg in cache_load_warnings:
        st.sidebar.warning(msg)

# All-days aufbauen (leichtgewichtige Rekonstruktion)
col_s_left, col_s_right = st.columns(2)

days: list[pd.Timestamp] = []
with col_s_left:
    cached_days = days_cache.get(selection_key)
    if cached_days:
        days = [pd.to_datetime(d).normalize() for d in cached_days]
        st.success(f"Found {len(days)} day(s). (session cache)")
    else:
        with st.status("Indexing available days for the current selection…", expanded=False) as s1:
            days = available_days_for(selected_base_names)
            if not days:
                s1.update(
                    label="No 5-minute partitions found for the current selection.", state="error"
                )
                st.stop()
            s1.update(label=f"Found {len(days)} day(s).", state="complete")
        days_cache[selection_key] = [pd.to_datetime(d).normalize() for d in days]

if not days:
    st.error("No daily partitions available for the current selection.")
    st.stop()

ALL_DAILY = pd.DataFrame()
with col_s_right:
    cached_df = daily_cache.get(selection_key)
    if cached_df is not None and not cached_df.empty:
        ALL_DAILY = cached_df.copy()
        st.success(f"DONE: {len(ALL_DAILY)} day(s) ready (session cache).")
    else:
        with st.status(
            "Building per-day base features (ALL_DAILY)…", expanded=False
        ) as s2:
            cpath = _cache_path_for_selection(selected_base_names)
            loaded = pd.DataFrame()
            if os.path.isfile(cpath):
                try:
                    loaded = pd.read_csv(cpath, parse_dates=["Date"])
                except Exception:
                    loaded = pd.DataFrame()
            want = set(pd.to_datetime(days).normalize())
            have = (
                set(pd.to_datetime(loaded["Date"]).dt.normalize())
                if (not loaded.empty and "Date" in loaded.columns)
                else set()
            )
            missing = sorted(list(want - have))
            missing_metric_cols = (
                [col for col in EXPECTED_METRIC_COLUMNS if col not in loaded.columns]
                if not loaded.empty
                else list(EXPECTED_METRIC_COLUMNS)
            )

            if loaded.empty or missing or missing_metric_cols:
                ALL_DAILY = compute_all_daily(selected_base_names, days)
                try:
                    ALL_DAILY.to_csv(cpath, index=False)
                except Exception:
                    pass
            else:
                ALL_DAILY = loaded.sort_values("Date")
            s2.update(
                label=f"DONE: {len(ALL_DAILY)} day(s) ready (cache: {'used' if os.path.isfile(cpath) else 'new'}).",
                state="complete",
            )
        if not ALL_DAILY.empty:
            daily_cache[selection_key] = ALL_DAILY.copy()
            source_map[selection_key] = {"type": "computed"}
            if selection_key not in days_cache:
                day_series = pd.to_datetime(ALL_DAILY["Date"], errors="coerce").dropna().dt.normalize()
                days_cache[selection_key] = day_series.drop_duplicates().tolist()

if ALL_DAILY.empty:
    st.error("No day metrics available")
    st.stop()

source_info = source_map.get(selection_key)
if source_info and source_info.get("type") == "feature_cache":
    files = source_info.get("files") or list(selected_cache_files)
    if files:
        st.caption(f"Feature cache snapshot(s) loaded: {', '.join(files)}")

# Model configuration notice
pass



# -------- Konfiguration: nur die vorgegebenen Metriken --------

DEFAULT_CFG: Dict[str, Any] = {
    "M": 14,
    "N": 10,
    "theta": 0.7,
    "core_symptoms": ["C2"],
    # C1
    "C1": {
        # keys = exact ALL_DAILY columns
        "IS": {
            "w": 0.20,
            "mf": {"type": "tri", "lo": 0.30, "mid": 0.50, "hi": 0.70, "invert": False},
        },
        "IV": {
            "w": 0.20,
            "mf": {"type": "tri", "lo": 0.60, "mid": 1.00, "hi": 1.50, "invert": False},
        },
        "F2_MeanSocialDurSec": {
            "w": 0.10,
            "mf": {"type": "tri", "lo": 60, "mid": 120, "hi": 240, "invert": False},
        },
        "F7_HourlyCV": {
            "w": 0.10,
            "mf": {"type": "tri", "lo": 0.10, "mid": 0.30, "hi": 0.60, "invert": False},
        },
        "F8_NightDayRatioPkts": {
            "w": 0.10,
            "mf": {"type": "tri", "lo": 0.10, "mid": 0.30, "hi": 0.60, "invert": False},
        },
        "LateNightShare": {
            "w": 0.10,
            "mf": {"type": "tri", "lo": 0.10, "mid": 0.20, "hi": 0.30, "invert": False},
        },
        "LongestInactivityHours": {
            "w": 0.05,
            "mf": {"type": "tri", "lo": 2.0, "mid": 4.0, "hi": 8.0, "invert": True},
        },
        "ActiveNightMinutes": {
            "w": 0.05,
            "mf": {"type": "tri", "lo": 10, "mid": 30, "hi": 90, "invert": False},
        },
        "ND_Ratio": {
            "w": 0.05,
            "mf": {"type": "tri", "lo": 0.05, "mid": 0.20, "hi": 0.50, "invert": False},
        },
        "F1_DistinctSocial": {
            "w": 0.025,
            "mf": {"type": "tri", "lo": 2, "mid": 3, "hi": 5, "invert": True},
        },
        "F6_Fano": {
            "w": 0.025,
            "mf": {"type": "tri", "lo": 0.5, "mid": 1.0, "hi": 1.5, "invert": False},
        },
        "F4_DownUpRatio": {
            "w": 0.025,
            "mf": {"type": "tri", "lo": 5, "mid": 10, "hi": 20, "invert": False},
        },
    },
    # C2
    "C2": {
        "C2_F3_ChatSessionCount": {
            "w": 0.35,
            "mf": {"type": "tri", "lo": -50, "mid": -25, "hi": 0, "invert": False},
        },  # use deltas (%) optional
        "C2_F1_UniqueSLD": {
            "w": 0.30,
            "mf": {"type": "tri", "lo": -40, "mid": -20, "hi": 0, "invert": False},
        },
        "C2_F6_ProductivityHits": {
            "w": 0.20,
            "mf": {"type": "tri", "lo": -60, "mid": -30, "hi": 0, "invert": False},
        },
        "C2_F4_MeanUpstreamRateBps": {
            "w": 0.15,
            "mf": {"type": "tri", "lo": -50, "mid": -25, "hi": 0, "invert": False},
        },
    },
    # C4
    "C4": {
        "C4_F4_SleepDurationZAbs30d": {
            "w": 0.50,
            "mf": {"type": "tri", "lo": 0.5, "mid": 1.0, "hi": 1.5, "invert": False},
        },
        "C4_F1_OnsetDelayFrom2200Min": {
            "w": 0.30,
            "mf": {"type": "tri", "lo": 15, "mid": 75, "hi": 180, "invert": False},
        },
        "C4_F2_WakeAfter0400Min": {
            "w": 0.20,
            "mf": {"type": "tri", "lo": 150, "mid": 210, "hi": 270, "invert": False},
        },
    },
    # C5
    "C5": {
        "C5_F5_Sub30sSessions": {
            "w": 0.50,
            "mf": {"type": "tri", "lo": 0, "mid": 5, "hi": 10, "invert": False},
        },
        "C5_F1_DhcpPerHour": {
            "w": 0.30,
            "mf": {"type": "tri", "lo": 1, "mid": 3, "hi": 5, "invert": False},
        },
        "C5_F2_WifiDwellMin": {
            "w": 0.20,
            "mf": {"type": "tri", "lo": 5, "mid": 10, "hi": 20, "invert": True},
        },
    },
    # C6
    "C6": {
        "C6_F8_FirstActivityMin": {
            "w": 0.60,
            "mf": {"type": "tri", "lo": 0, "mid": 45, "hi": 90, "invert": False},
        },
        "C6_F9_ActivationDelayVsBase28d": {
            "w": 0.40,
            "mf": {"type": "tri", "lo": 15, "mid": 45, "hi": 90, "invert": False},
        },
    },
    # C8
    "C8": {
        "C8_F2_DNSBurstRatePerHour": {
            "w": 0.25,
            "mf": {"type": "tri", "lo": 1, "mid": 3, "hi": 5, "invert": False},
        },
        "C8_F4_RepeatedQueryRatio60m": {
            "w": 0.15,
            "mf": {"type": "tri", "lo": 0.05, "mid": 0.10, "hi": 0.20, "invert": False},
        },
        "C8_F8_MedianIKSsec": {
            "w": 0.60,
            "mf": {"type": "tri", "lo": 0.30, "mid": 0.40, "hi": 0.60, "invert": False},
        },
    },
}


CRIT_DISPLAY = {
    "C1": "C1 - Depressed mood",
    "C2": "C2 – Loss of interest / anhedonia",
    "C3": "C3 – Appetite / weight change",
    "C4": "C4 – Sleep timing & duration",
    "C5": "C5 – Psychomotor agitation/retardation",
    "C6": "C6 – Fatigue / low energy",
    "C7": "C7 – Worthlessness / guilt",
    "C8": "C8 – Difficulty concentrating / indecisiveness",
    "C9": "C9 – Suicidality",
}

# Prepare per-day likelihoods per criterion
crit_cols = CRIT_KEYS
vals = {c: [] for c in crit_cols}
dates = []

for _, row in ALL_DAILY.iterrows():
    d = pd.to_datetime(row.get("Date")).normalize()
    dates.append(d)
    for crit in crit_cols:
        spec = cfg_state.get(crit, {})
        # Gather metric values for this criterion
        vmap = {}
        for k in spec.keys():
            vmap[k] = row.get(k)
        vals[crit].append(fasl_score(vmap, spec))

DF_L = pd.DataFrame(
    {"Date": dates, **{f"L_{c}": pd.Series(vals[c]) for c in crit_cols}}
).sort_values("Date")

## Compute DSM‑Gate presence flags for summary metrics
present = {}
for crit in crit_cols:
    s = DF_L[f"L_{crit}"] if f"L_{crit}" in DF_L.columns else pd.Series(dtype=float)
    present[crit] = gate_present(s, theta=theta_default, need_days=gate_need, window=gate_window)

# Episode decision summary
core_ok = any(present.get(c, False) for c in core_criteria)
total_present = int(sum(1 for v in present.values() if v))
episode_likely = core_ok and (total_present >= 5)

st.write("---")
st.subheader("Criterion Likelihoods and DSM-Gate")
st.caption(
    f"""
    The DSM‑Gate uses a rolling window M={int(cfg_state.get('M', 14))} days and requires N={int(cfg_state.get('N', 10))} days ≥ θ to mark a criterion as present.
    A depressive episode is likely when ≥5 criteria are present and at least one core symptom ({', '.join(cfg_state.get('core_symptoms', ['C2']))}) is present.
    """
)

# Place summary cards directly under the subheader with background colors
GREEN_BG = "#d1fae5"
RED_BG = "#fee2e2"
cols_summary = st.columns(3)
with cols_summary[0]:
    bg = RED_BG if episode_likely else GREEN_BG
    st.markdown(
        f"""
        <div style=\"background-color:{bg}; border-radius:8px; padding:10px;\">
          <div style=\"font-size:0.9rem; opacity:0.7;\">Episode likely</div>
          <div style=\"font-size:1.4rem; font-weight:600;\">{'Yes' if episode_likely else 'No'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with cols_summary[1]:
    core_flags = {c: bool(present.get(c, False)) for c in core_criteria}
    selected_names: List[str] = []
    active_names: List[str] = []
    for code in core_criteria:
        label_full = CRIT_DISPLAY.get(code, code)
        label_desc = label_full.split(" – ", 1)[1] if " – " in label_full else label_full
        selected_names.append(label_desc)
        if core_flags.get(code):
            active_names.append(label_desc)
    selected_text = ", ".join(selected_names) if selected_names else "None selected"
    active_text = ", ".join(active_names) if active_names else "None currently"
    bg = RED_BG if core_ok else GREEN_BG
    st.markdown(
        f"""
        <div style=\"background-color:{bg}; border-radius:8px; padding:10px;\">
          <div style=\"font-size:0.9rem; opacity:0.7;\">Core symptom satisfied</div>
          <div style=\"font-size:1.4rem; font-weight:600;\">{'Yes' if core_ok else 'No'}</div>
          <div style=\"font-size:0.75rem; opacity:0.75; margin-top:0.25rem;\">Selected: {html.escape(selected_text)}</div>
          <div style=\"font-size:0.75rem; opacity:0.75;\">Active: {html.escape(active_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with cols_summary[2]:
    bg = GREEN_BG if total_present < 5 else RED_BG
    st.markdown(
        f"""
        <div style=\"background-color:{bg}; border-radius:8px; padding:10px;\">
          <div style=\"font-size:0.9rem; opacity:0.7;\">Criteria present (last M={int(gate_window)} days)</div>
          <div style=\"font-size:1.4rem; font-weight:600;\">{total_present}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("  ")
with st.expander("DSM-5 Diagnostic Reference", expanded=False):
    st.markdown(
        """
    DSM‑5 describes a Major Depressive Episode as having at least 5 symptoms present during the same 2‑week period, and at least one is either depressed mood or markedly diminished interest/pleasure (anhedonia). This page implements a transparent, rule‑based approximation: daily likelihoods (L_k) aggregated over a rolling window M with threshold θ and count N. Tuning M/N/θ adjusts sensitivity while staying faithful to the spirit of the DSM‑5 criteria.
    """
    )
st.write("  ")

# Compute average likelihood per criterion (all days), pick top 6 (shared)
avg_list: list[tuple[str, float]] = []
for c in crit_cols:
    col = f"L_{c}"
    if col in DF_L.columns:
        s = pd.to_numeric(DF_L[col], errors="coerce").dropna()
        if not s.empty:
            m = float(s.mean())
            if np.isfinite(m):
                avg_list.append((c, m))
avg_list.sort(key=lambda x: x[1], reverse=True)
top6_tabs = avg_list[:6]

# Tabs for gauges: average vs selected day
tab_avg, tab_day = st.tabs(["Average over all days", "Selected day"])

def _gauge_plot(label: str, value: float):
    import plotly.graph_objects as go
    val = float(np.clip(value, 0.0, 1.0))
    num_color = ("#ef4444" if val >= float(theta_default) else "#22c55e")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            title={"text": label, "font": {"size": 12}},
            number={"font": {"color": num_color, "size": 28}},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, float(theta_default)], "color": "rgba(34,197,94,0.15)"},
                    {"range": [float(theta_default), 1], "color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "gray", "width": 2},
                    "thickness": 0.75,
                    "value": float(theta_default),
                },
            },
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=16, b=8), height=120)
    return fig

with tab_avg:
    for i in range(0, len(top6_tabs), 3):
        trio = top6_tabs[i : i + 3]
        row_cols = st.columns(len(trio))
        for j, (c, m) in enumerate(trio):
            with row_cols[j].container(border=True):
                label = CRIT_DISPLAY.get(c, c)
                st.markdown(f"**{label}**")
                st.plotly_chart(_gauge_plot(label, m), use_container_width=True, key=f"gauge_avg_{c}")

with tab_day:
    try:
        _ = st.calendar("Select a day", key="fasl_sel_day")
        sel_date = st.session_state.get("fasl_sel_day")
    except Exception:
        default_date = pd.to_datetime(DF_L["Date"].max()).date() if not DF_L.empty else pd.Timestamp.now().date()
        sel_date = st.date_input("Select a day", value=default_date, key="fasl_sel_day_fallback")

    try:
        if isinstance(sel_date, (list, tuple)) and len(sel_date) >= 1:
            _raw = sel_date[0]
        else:
            _raw = sel_date
        day_norm = pd.to_datetime(_raw).normalize() if _raw is not None else None
    except Exception:
        day_norm = None

    if day_norm is None:
        st.info("Please select a valid day with data.")
    else:
        df_day_l = DF_L[DF_L["Date"] == day_norm]
        if df_day_l.empty:
            st.warning("No data for the selected day.")
        else:
            for i in range(0, len(top6_tabs), 3):
                trio = top6_tabs[i : i + 3]
                row_cols = st.columns(len(trio))
                for j, (c, _avg) in enumerate(trio):
                    with row_cols[j].container(border=True):
                        label = CRIT_DISPLAY.get(c, c)
                        val = df_day_l.iloc[0].get(f"L_{c}")
                        try:
                            val = float(val)
                        except Exception:
                            val = float('nan')
                        st.markdown(f"**{label}**")
                        day_key = pd.to_datetime(day_norm).strftime('%Y%m%d') if day_norm is not None else 'NA'
                        st.plotly_chart(
                            _gauge_plot(label, val if np.isfinite(val) else 0.0),
                            use_container_width=True,
                            key=f"gauge_day_{c}_{day_key}"
                        )

# Gauges (top): 3 columns x 2 rows; time series below (full width)
container_placeholder = None  # no columns; gauges use full width

with st.container():
    try:
        import plotly.graph_objects as go

        # Compute average likelihood per criterion (all days), pick top 6
        avg_list: list[tuple[str, float]] = []
        for c in crit_cols:
            col = f"L_{c}"
            if col in DF_L.columns:
                s = pd.to_numeric(DF_L[col], errors="coerce").dropna()
                if not s.empty:
                    m = float(s.mean())
                    if np.isfinite(m):
                        avg_list.append((c, m))
        avg_list.sort(key=lambda x: x[1], reverse=True)
        top6 = []  # disabled (moved to tabs)

        def _gauge(label: str, value: float):
            val = float(np.clip(value, 0.0, 1.0))
            # Tailwind-like colors: red-500 and green-600
            num_color = ("#ef4444" if val >= float(theta_default) else "#22c55e")
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=val,
                    title={"text": label, "font": {"size": 12}},
                    number={"font": {"color": num_color, "size": 18}},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "#1f77b4"},
                        "steps": [
                            {"range": [0, float(theta_default)], "color": "rgba(34,197,94,0.15)"},
                            {"range": [float(theta_default), 1], "color": "rgba(239,68,68,0.15)"},
                        ],
                        "threshold": {
                            "line": {"color": "gray", "width": 2},
                            "thickness": 0.75,
                            "value": float(theta_default),
                        },
                    },
                )
            )
            fig.update_layout(margin=dict(l=10, r=10, t=16, b=8), height=120)
            return fig

        # Render gauges in 2 rows x 3 columns, each inside a bordered container
        for i in range(0, len(top6), 3):
            trio = top6[i : i + 3]
            row_cols = st.columns(len(trio))
            for j, (c, m) in enumerate(trio):
                with row_cols[j].container(border=True):
                    label = c
                    try:
                        label = CRIT_DISPLAY.get(c, c)  # optional mapping if available
                    except Exception:
                        pass
                    st.markdown(f"**{label}**")
                    st.plotly_chart(_gauge(label, m), use_container_width=True)

    except Exception:
        pass



st.write("---")
st.subheader("FASL Configuration")

CRIT_TABS = [
    ("C1", "C1 - Depressed mood"),
    ("C2", "C2 – Loss of interest / anhedonia"),
    ("C3", "C3 – Appetite / weight change"),
    ("C4", "C4 – Sleep timing & duration"),
    ("C5", "C5 – Psychomotor agitation/retardation"),
    ("C6", "C6 – Fatigue / low energy"),
    ("C7", "C7 – Worthlessness / guilt"),
    ("C8", "C8 – Difficulty concentrating / indecisiveness"),
    ("C9", "C9 – Suicidality"),
]
CRIT_KEYS = [c for c, _ in CRIT_TABS]
tabs = st.tabs([label for _, label in CRIT_TABS])

CRIT_DISPLAY = {
    "C1": "C1 - Depressed mood",
    "C2": "C2 – Loss of interest / anhedonia",
    "C3": "C3 – Appetite / weight change",
    "C4": "C4 – Sleep timing & duration",
    "C5": "C5 – Psychomotor agitation/retardation",
    "C6": "C6 – Fatigue / low energy",
    "C7": "C7 – Worthlessness / guilt",
    "C8": "C8 – Difficulty concentrating",
    "C9": "C9 – Suicidality",
}

ALL_METRIC_OPTIONS = {

    "C1": [d.dist_col for d in C1_DEFS],

    "C2": [d.dist_col for d in C2_DEFS],

    "C3": [d.dist_col for d in C3_DEFS],

    "C4": [d.dist_col for d in C4_DEFS],

    "C5": [d.dist_col for d in C5_DEFS],

    "C6": [d.dist_col for d in C6_DEFS],

    "C7": [d.dist_col for d in C7_DEFS],

    "C8": [d.dist_col for d in C8_DEFS],

    "C9": [d.dist_col for d in C9_DEFS],

}



def _normalise_bom_value(raw) -> str:
    if isinstance(raw, str):
        val = raw.strip()
        if not val or val.lower() == "no bom defined":
            return ""
        return val
    return ""

CRIT_BOM_OPTIONS = {
    "C1": [
        "(HOB1) Reduced Social Interaction",
        "(HOB2) Passive Media Binge",
        "(HOB3) Rumination Browsing loops",
        "(HOB4) Flattened diurnal Rhythm",
    ],
    "C2": [
        "(HOB1) Shrinking Domain Diversity",
        "(HOB2) Reduced Interactive Engagement",
        "(HOB3) Drop in Goal-Oriented Browsing",
    ],
    "C3": [
        "(HOB1) Food-ordering Pattern Shift",
        "(HOB2) Calorie / Nutrition Information Seeking",
        "(HOB3) Smart-scale usage Variability",
    ],
    "C4": [
        "(HOB1) Shifted Sleep Timing",
        "(HOB2) Changed Sleep Duration",
        "(HOB3) Sleep Fragmentation",
        "(HOB4) Daytime Hypersomnia / Rhythm Flattening",
    ],
    "C5": [
        "(HOB1) Restless Device Switching (agitation)",
        "(HOB2) Slowed / Variable Typing Dynamics (retardation)",
        "(HOB3) \"Screen-check\" Burstiness",
    ],
    "C6": [
        "(HOB1) Extended Day-time Inactivity",
        "(HOB2) Decline in Effortful Interaction",
        "(HOB3) Slowed Browsing Tempo",
        "(HOB4) Delayed Morning Activation",
    ],
    "C7": [
        "(HOB1) Engagement with self-evaluative or mental-health resources",
        "(HOB2) Digital self-withdrawal / data purge behaviour",
    ],
    "C8": [
        "(HOB1) Fragmented focus / frequent task-switches",
        "(HOB2) Indecisive information seeking",
        "(HOB3) Cognitive sluggishness / slow response",
    ],
    "C9": [
        "(HOB1) Crisis-oriented help seeking",
        "(HOB2) Self-harm community engagement",
        "(HOB3) Farewell / estate preparation",
        "(HOB4) Nocturnal rumination spikes",
    ],
}


def _ensure_bom_field(cfg: dict) -> None:
    for crit_val in cfg.values():
        if isinstance(crit_val, dict):
            for spec in crit_val.values():
                if isinstance(spec, dict) and "w" in spec:
                    spec["bom"] = _normalise_bom_value(spec.get("bom"))

_ensure_bom_field(DEFAULT_CFG)


def _metric_differs_from_default(crit: str, metric: str, spec: dict) -> bool:
    default = DEFAULT_CFG.get(crit, {}).get(metric)
    if default is None:
        return True
    try:
        if not math.isclose(float(spec.get("w", 0.0)), float(default.get("w", 0.0)), rel_tol=1e-9, abs_tol=1e-9):
            return True
    except Exception:
        return True
    if _normalise_bom_value(spec.get("bom")) != _normalise_bom_value(default.get("bom")):
        return True
    mf = spec.get("mf", {}) or {}
    def_mf = default.get("mf", {}) or {}
    if str(mf.get("type", "tri")).lower() != str(def_mf.get("type", "tri")).lower():
        return True
    for key in ("lo", "mid", "hi"):
        try:
            if not math.isclose(float(mf.get(key, 0.0)), float(def_mf.get(key, 0.0)), rel_tol=1e-9, abs_tol=1e-9):
                return True
        except Exception:
            return True
    if bool(mf.get("invert", False)) != bool(def_mf.get("invert", False)):
        return True
    return False


def _metric_sort_key(name: str) -> tuple:
    """Sort F-prefixed metrics numerically first, then others."""
    m = re.match(r"F(\d+)_", name)
    if m:
        return (0, int(m.group(1)), name)
    return (1, name)


def _normalize_uploaded_config(cfg: dict) -> dict:
    """Normalize an uploaded config to the app's internal schema.

    - Accept both "core" and "core_symptoms" and unify as "core_symptoms".
    - Accept per-metric "weight" as alias for "w".
    - Accept MF parameters either nested under "mf" or flattened (lo/mid/hi/invert/type at metric level).
    - Filter metrics to those available in this app per criterion.
    """
    if not isinstance(cfg, dict):
        return {}

    out: dict = {}
    # Gate-level keys
    if "M" in cfg:
        try:
            out["M"] = int(cfg.get("M"))
        except Exception:
            pass
    if "N" in cfg:
        try:
            out["N"] = int(cfg.get("N"))
        except Exception:
            pass
    if "theta" in cfg:
        try:
            out["theta"] = float(cfg.get("theta"))
        except Exception:
            pass
    # Core symptoms (support both keys)
    core_val = cfg.get("core_symptoms", cfg.get("core"))
    if core_val is not None:
        try:
            if isinstance(core_val, str):
                core_candidates = [core_val]
            elif isinstance(core_val, (list, tuple, set)):
                core_candidates = list(core_val)
            else:
                core_candidates = []
            valid = {"C1","C2","C3","C4","C5","C6","C7","C8","C9"}
            core_list = [str(x) for x in core_candidates if str(x) in valid]
            out["core_symptoms"] = core_list
        except Exception:
            pass

    # Per-criterion metrics
    # Support configs where criteria are nested under a "criteria" object
    crit_src = cfg.get("criteria") if isinstance(cfg.get("criteria"), dict) else cfg
    for crit in ["C1","C2","C3","C4","C5","C6","C7","C8","C9"]:
        if crit not in crit_src:
            continue
        crit_in = crit_src[crit]
        # If criterion object wraps metrics under a 'metrics' key, unwrap it
        if isinstance(crit_in, dict) and "metrics" in crit_in:
            inner = crit_in.get("metrics")
            if isinstance(inner, (dict, list)):
                crit_in = inner
        crit_out = {}
        available = set(ALL_METRIC_OPTIONS.get(crit, []))
        # crit_in can be a dict mapping metric->spec, or a list of entries
        if isinstance(crit_in, dict):
            items_iter = crit_in.items()
        elif isinstance(crit_in, list):
            # Convert list entries to (metric, spec) pairs
            tmp = []
            for entry in crit_in:
                if not isinstance(entry, dict):
                    continue
                mname = entry.get("metric") or entry.get("name") or entry.get("key")
                if not mname:
                    continue
                tmp.append((str(mname), entry))
            items_iter = tmp
        else:
            items_iter = []

        for m, spec in items_iter:
            if m not in available:
                # Skip unknown metrics to avoid UI inconsistencies
                continue
            if not isinstance(spec, dict):
                continue
            weight = spec.get("w", spec.get("weight", 0.1))
            try:
                weight = float(weight)
            except Exception:
                weight = 0.1
            mf_in = spec.get("mf", {})
            # Allow flattened parameters
            if not isinstance(mf_in, dict):
                mf_in = {}
            lo = spec.get("lo", mf_in.get("lo", 0.0))
            mid = spec.get("mid", mf_in.get("mid", 0.0))
            hi = spec.get("hi", mf_in.get("hi", 0.0))
            invert = spec.get("invert", mf_in.get("invert", False))
            typ = spec.get("type", mf_in.get("type", "tri"))
            # Robust bool parsing for invert
            if isinstance(invert, str):
                invert = invert.strip().lower() in {"true","1","yes","y","on"}
            else:
                invert = bool(invert)
            try:
                lo = float(lo)
            except Exception:
                lo = 0.0
            try:
                mid = float(mid)
            except Exception:
                mid = 0.0
            try:
                hi = float(hi)
            except Exception:
                hi = 0.0
            crit_out[m] = {
                "w": weight,
                "bom": _normalise_bom_value(spec.get("bom")) if isinstance(spec, dict) else "",
                "mf": {"type": str(typ).lower(), "lo": lo, "mid": mid, "hi": hi, "invert": invert},
            }
        if crit_out:
            out[crit] = crit_out

    _ensure_bom_field(out)
    return out





cfg_state = st.session_state.setdefault("fasl_cfg", {})
for _k in ("M", "N", "theta", "core_symptoms"):
    if _k not in cfg_state:
        _v = DEFAULT_CFG.get(_k)
        cfg_state[_k] = copy.deepcopy(_v) if isinstance(_v, (dict, list)) else _v

_ensure_bom_field(cfg_state)


HIW_MAP = {
    d.dist_col: d.higher_is_worse
    for defs in [
        C1_DEFS,
        C2_DEFS,
        C3_DEFS,
        C4_DEFS,
        C5_DEFS,
        C6_DEFS,
        C7_DEFS,
        C8_DEFS,
        C9_DEFS,
    ]
    for d in defs
}

LABEL_MAP = {
    d.dist_col: getattr(d, 'label', d.dist_col)
    for defs in [
        C1_DEFS,
        C2_DEFS,
        C3_DEFS,
        C4_DEFS,
        C5_DEFS,
        C6_DEFS,
        C7_DEFS,
        C8_DEFS,
        C9_DEFS,
    ]
    for d in defs
}


def _plotly_chart_scaled(fig, width_scale: float = 1.0, **kwargs) -> None:
    """Render Plotly chart with optional width scaling."""
    try:
        scale = float(width_scale)
    except Exception:
        scale = 1.0
    scale = max(0.0, min(scale, 1.0))
    if abs(scale - 1.0) < 1e-6 or scale <= 0.0:
        st.plotly_chart(fig, **kwargs)
        return
    remainder = max(1e-3, 1.0 - scale)
    cols = st.columns([scale, remainder])
    with cols[0]:
        st.plotly_chart(fig, **kwargs)


def _add_tri_background(fig, lo, mid, hi, invert, y_min, y_max):
    import math
    vals = (lo, mid, hi)
    if all((not np.isfinite(v)) or math.isclose(float(v), 0.0, abs_tol=1e-9) for v in vals):
        return None
    if y_min is None or not np.isfinite(y_min):
        y_min = float('-inf')
    if y_max is None or not np.isfinite(y_max):
        y_max = float('inf')
    if not np.isfinite(mid):
        return None
    lo_draw = float(lo) if np.isfinite(lo) else y_min
    hi_draw = float(hi) if np.isfinite(hi) else y_max
    mid_draw = float(mid)
    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
        lo_draw = float(np.clip(lo_draw, y_min, y_max))
        mid_draw = float(np.clip(mid_draw, y_min, y_max))
        hi_draw = float(np.clip(hi_draw, y_min, y_max))
    membership_fill = "rgba(239,68,68,0.18)"
    outside_fill = "rgba(134,239,172,0.18)"
    span_lo = sorted((lo_draw, mid_draw))
    span_hi = sorted((mid_draw, hi_draw))
    if span_lo[0] != span_lo[1]:
        fig.add_hrect(
            y0=span_lo[0],
            y1=span_lo[1],
            line_width=0,
            fillcolor=membership_fill,
            layer="below",
        )
    if span_hi[0] != span_hi[1]:
        fig.add_hrect(
            y0=span_hi[0],
            y1=span_hi[1],
            line_width=0,
            fillcolor=membership_fill,
            layer="below",
        )
    if not invert and np.isfinite(y_min) and np.isfinite(lo_draw) and lo_draw > y_min:
        fig.add_hrect(
            y0=y_min,
            y1=lo_draw,
            line_width=0,
            fillcolor=outside_fill,
            layer="below",
        )
    if invert and np.isfinite(y_max) and np.isfinite(hi_draw) and hi_draw < y_max:
        fig.add_hrect(
            y0=hi_draw,
            y1=y_max,
            line_width=0,
            fillcolor=outside_fill,
            layer="below",
        )
    return mid_draw

def _add_tri_markers(
    fig,
    lo,
    mid,
    hi,
    orientation: str = "h",
    labels: dict[str, str] | None = None,
) -> None:
    entries = (
        ("lo", lo, "#0ea5e9"),
        ("mid", mid, "#f59e0b"),
        ("hi", hi, "#ef4444"),
    )
    processed: list[tuple[str, float, str, str]] = []
    label_map = labels or {}
    if orientation == "v":
        priority = {"lo": 2, "mid": 1, "hi": 2}
        for key, value, color in entries:
            if not np.isfinite(value):
                continue
            val = float(value)
            label_text = label_map.get(key, key)
            replace_idx: int | None = None
            for idx, (existing_key, existing_val, _, _) in enumerate(processed):
                if np.isclose(existing_val, val, rtol=1e-9, atol=1e-9):
                    if priority.get(key, 0) > priority.get(existing_key, 0):
                        replace_idx = idx
                    else:
                        replace_idx = -1  # marker to skip replacement
                    break
            if replace_idx == -1:
                continue
            if replace_idx is None:
                processed.append((key, val, color, label_text))
            else:
                processed[replace_idx] = (key, val, color, label_text)
    else:
        for key, value, color in entries:
            if not np.isfinite(value):
                continue
            val = float(value)
            label_text = label_map.get(key, key)
            processed.append((key, val, color, label_text))

    for _, val, color, label_text in processed:
        if orientation == "h":
            fig.add_hline(y=val, line_dash="dash", line_color=color)
            fig.add_annotation(
                x=1.0,
                y=val,
                xref="paper",
                yref="y",
                text=label_text,
                showarrow=False,
                font=dict(color=color, size=10),
                xanchor="left",
                xshift=-6,
                align="left",
            )
        else:
            fig.add_vline(x=val, line_dash="dash", line_color=color)
            fig.add_annotation(
                x=val,
                y=1.02,
                xref="x",
                yref="paper",
                text=label_text,
                showarrow=False,
                font=dict(color=color, size=10),
                yanchor="bottom",
            )

def _boxplot_with_ranges(
    df: pd.DataFrame,
    col: str,
    lo: float,
    mid: float,
    hi: float,
    invert: bool = False,
    theta: float | None = None,
) -> None:
    import plotly.express as px, plotly.graph_objects as go

    if col not in df.columns or df[col].dropna().empty:
        st.info("No historical values available for this metric.")
        return
    series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    fig = px.box(series, points="all")
    fig.update_layout(title_text="")
    fig.update_xaxes(visible=False)
    axis_label = LABEL_MAP.get(col, col)
    fig.update_yaxes(title=axis_label)
    ymin = float(series.min())
    ymax = float(series.max())
    try:
        _add_tri_background(fig, lo, mid, hi, invert, ymin, ymax)
        _add_tri_markers(
            fig,
            lo,
            mid,
            hi,
            orientation="h",
            labels={"lo": "lo", "mid": "mid", "hi": "hi"},
        )
        theta_val = float(theta) if theta is not None and np.isfinite(theta) else None
        if (
            theta_val is not None
            and np.isfinite(ymin)
            and np.isfinite(ymax)
            and ymin <= theta_val <= ymax
        ):
            fig.add_hline(y=theta_val, line_dash="dot", line_color="#6b7280")
            fig.add_annotation(
                x=1.0,
                y=theta_val,
                xref="paper",
                yref="y",
                text="\u03b8",
                showarrow=False,
                font=dict(color="#6b7280", size=10),
                xanchor="left",
                xshift=-6,
                align="left",
            )
    except Exception:
        pass
    _plotly_chart_scaled(fig, width_scale=0.9, use_container_width=True)



def _boxplot_with_ranges_marks(
    df: pd.DataFrame,
    col: str,
    lo: float,
    mid: float,
    hi: float,
    invert: bool = False,
    centers: tuple[float, float, float] | None = None,
    boundaries: tuple[float, float] | None = None,
    show_overlays: bool = True,
    theta: float | None = None,
):
    import plotly.express as px, plotly.graph_objects as go
    if col not in df.columns or df[col].dropna().empty:
        st.info("No historical values available for this metric.")
        return
    series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    fig = px.box(series, points="all")
    fig.update_layout(title_text="")
    fig.update_xaxes(visible=False)
    axis_label = LABEL_MAP.get(col, col)
    fig.update_yaxes(title=axis_label)
    ymin = float(series.min())
    ymax = float(series.max())
    try:
        _add_tri_background(fig, lo, mid, hi, invert, ymin, ymax)
        _add_tri_markers(
            fig,
            lo,
            mid,
            hi,
            orientation="h",
            labels={"lo": "lo", "mid": "mid", "hi": "hi"},
        )
        theta_val = float(theta) if theta is not None and np.isfinite(theta) else None
        if (
            theta_val is not None
            and np.isfinite(ymin)
            and np.isfinite(ymax)
            and ymin <= theta_val <= ymax
        ):
            fig.add_hline(y=theta_val, line_dash="dot", line_color="#6b7280")
            fig.add_annotation(
                x=1.0,
                y=theta_val,
                xref="paper",
                yref="y",
                text="θ",
                showarrow=False,
                font=dict(color="#6b7280", size=10),
                xanchor="left",
                xshift=-6,
                align="left",
            )
        if show_overlays:
            if boundaries is not None:
                t12, t23 = float(boundaries[0]), float(boundaries[1])
                fig.add_hline(y=t12, line_dash="dashdot", line_color="#9333ea")
                fig.add_hline(y=t23, line_dash="dashdot", line_color="#9333ea")
            if centers is not None:
                c1, c2, c3 = (float(centers[0]), float(centers[1]), float(centers[2]))
                for c, colr in zip((c1, c2, c3), ("#0ea5e9", "#f59e0b", "#ef4444")):
                    fig.add_hline(y=c, line_dash="solid", line_color=colr)
        leg = [
            go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#0ea5e9", width=3), name="Center lo"),
            go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#f59e0b", width=3), name="Center mid"),
            go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#ef4444", width=3), name="Center hi"),
            go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#9333ea", width=3, dash="dashdot"), name="Boundary"),
        ]
        for tr in leg:
            fig.add_trace(tr)
    except Exception:
        pass
    _plotly_chart_scaled(fig, width_scale=0.9, use_container_width=True)



def _boxplot_membership(
    df: pd.DataFrame,
    col: str,
    lo: float,
    mid: float,
    hi: float,
    invert: bool = False,
    theta: float | None = None,
):
    import plotly.express as px
    if col not in df.columns or df[col].dropna().empty:
        st.info("No historical values available for this metric.")
        return
    series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    mu_vals = series.apply(lambda x: mf_tri(float(x), float(lo), float(mid), float(hi), invert=bool(invert)))
    mu_vals = mu_vals.replace([np.inf, -np.inf], np.nan).dropna()
    if mu_vals.empty:
        st.info("No valid values to normalize.")
        return
    axis_label = LABEL_MAP.get(col, col)
    fig = px.box(mu_vals, points="all", range_y=[0, 1])
    fig.update_layout(title_text="")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(title=axis_label, range=[0, 1])
    theta_val = float(theta) if theta is not None and np.isfinite(theta) else None
    if theta_val is not None:
        fill_top = float(np.clip(theta_val, 0.0, 1.0))
        fig.add_hrect(
            y0=0,
            y1=fill_top,
            line_width=0,
            fillcolor="rgba(239,68,68,0.14)",
            layer="below",
        )
        fig.add_hline(y=fill_top, line_dash="dot", line_color="#6b7280")
        fig.add_annotation(
            x=1.0,
            y=fill_top,
            xref="paper",
            yref="y",
            text="θ",
            showarrow=False,
            font=dict(color="#6b7280", size=10),
            xanchor="left",
            xshift=-6,
            align="left",
        )
    _plotly_chart_scaled(fig, width_scale=0.9, use_container_width=True)


def _membership_curve_chart(
    df: pd.DataFrame,
    col: str,
    lo: float,
    mid: float,
    hi: float,
    invert: bool = False,
    theta: float | None = None,
) -> None:
    import numpy as np
    import plotly.graph_objects as go

    axis_label = LABEL_MAP.get(col, col)
    if col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    else:
        series = pd.Series(dtype=float)

    candidates: list[float] = []
    for val in (lo, mid, hi):
        if np.isfinite(val):
            candidates.append(float(val))
    if not series.empty:
        candidates.extend([float(series.min()), float(series.max())])
    if not np.isfinite(lo) and np.isfinite(mid):
        candidates.append(float(mid))
    if not np.isfinite(hi) and np.isfinite(mid):
        candidates.append(float(mid))

    finite_candidates = [c for c in candidates if np.isfinite(c)]
    if finite_candidates:
        x_min = min(finite_candidates)
        x_max = max(finite_candidates)
    else:
        x_min, x_max = -1.0, 1.0

    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0

    span = x_max - x_min
    margin = 0.1 * span if np.isfinite(span) and span > 0 else 1.0
    x_min -= margin
    x_max += margin

    lo_draw = float(lo) if np.isfinite(lo) else x_min
    hi_draw = float(hi) if np.isfinite(hi) else x_max
    if np.isfinite(mid):
        mid_draw = float(mid)
    else:
        mid_draw = float(
            lo_draw + (hi_draw - lo_draw) / 2.0 if np.isfinite(hi_draw) and np.isfinite(lo_draw) else x_min
        )
    lo_draw = float(np.clip(lo_draw, x_min, x_max))
    mid_draw = float(np.clip(mid_draw, x_min, x_max))
    hi_draw = float(np.clip(hi_draw, x_min, x_max))
    lo_draw, mid_draw, hi_draw = sorted((lo_draw, mid_draw, hi_draw))

    x_values = np.linspace(x_min, x_max, 200)
    y_values = [
        mf_tri(float(xv), float(lo), float(mid), float(hi), invert=bool(invert))
        for xv in x_values
    ]

    mu_label = re.sub(r"[^A-Za-z0-9]+", " ", axis_label).strip() or axis_label
    y_axis_title = f"\u03BC_{mu_label}(x)"

    fig = go.Figure()
    membership_fill = "rgba(239,68,68,0.18)"
    outside_fill = "rgba(134,239,172,0.18)"
    theta_val = float(theta) if theta is not None and np.isfinite(theta) else None
    cap = float(np.clip(theta_val, 0.0, 1.0)) if theta_val is not None else 1.0
    cap = max(0.0, float(cap))

    def _add_band(x0: float, x1: float, color: str) -> None:
        if cap <= 0.0:
            return
        if not (np.isfinite(x0) and np.isfinite(x1)):
            return
        start = float(np.clip(min(x0, x1), x_min, x_max))
        end = float(np.clip(max(x0, x1), x_min, x_max))
        if start >= end:
            return
        fig.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=0.0,
            y1=cap,
            xref="x",
            yref="y",
            fillcolor=color,
            line=dict(width=0),
            layer="below",
        )

    if not invert and x_min < lo_draw:
        _add_band(x_min, lo_draw, outside_fill)
    if lo_draw < mid_draw:
        _add_band(lo_draw, mid_draw, membership_fill)
    if mid_draw < hi_draw:
        _add_band(mid_draw, hi_draw, membership_fill)
    if invert and hi_draw < x_max:
        _add_band(hi_draw, x_max, outside_fill)
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            name="Membership",
            line=dict(color="#0ea5e9"),
        )
    )

    scatter_x: list[float] = []
    scatter_y: list[float] = []
    if not series.empty:
        scatter_x = series.tolist()
        scatter_y = [
            mf_tri(float(val), float(lo), float(mid), float(hi), invert=bool(invert))
            for val in series
        ]
        above_x: list[float] = []
        above_y: list[float] = []
        below_x: list[float] = []
        below_y: list[float] = []
        for x_val, y_val in zip(scatter_x, scatter_y):
            if theta_val is not None and y_val > theta_val:
                above_x.append(float(x_val))
                above_y.append(float(y_val))
            else:
                below_x.append(float(x_val))
                below_y.append(float(y_val))
        if below_x:
            fig.add_trace(
                go.Scatter(
                    x=below_x,
                    y=below_y,
                    mode="markers",
                    marker=dict(size=6, color="#1d4ed8", opacity=0.85),
                    showlegend=False,
                )
            )
        if above_x:
            fig.add_trace(
                go.Scatter(
                    x=above_x,
                    y=above_y,
                    mode="markers",
                    marker=dict(size=6, color="#60a5fa", opacity=0.95),
                    showlegend=False,
                )
            )

    _add_tri_markers(fig, lo, mid, hi, orientation="v")

    if theta_val is not None:
        fig.add_hline(y=cap, line_dash="dot", line_color="#6b7280")
        fig.add_annotation(
            x=1.0,
            y=cap,
            xref="paper",
            yref="y",
            text="θ",
            showarrow=False,
            font=dict(color="#6b7280", size=10),
            xanchor="left",
            xshift=-6,
            align="left",
        )
        if scatter_x:
            threshold_x: list[float] = []
            threshold_y: list[float] = []
            for x_val, y_val in zip(scatter_x, scatter_y):
                if y_val > theta_val:
                    fig.add_annotation(
                        x=float(x_val),
                        y=cap,
                        ax=float(x_val),
                        ay=float(y_val),
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=0,
                        arrowsize=0.6,
                        arrowwidth=1,
                        arrowcolor="#6b7280",
                    )
                    threshold_x.append(float(x_val))
                    threshold_y.append(cap)
            if threshold_x:
                fig.add_trace(
                    go.Scatter(
                        x=threshold_x,
                        y=threshold_y,
                        mode="markers",
                        marker=dict(size=5, color="#1d4ed8"),
                        showlegend=False,
                    )
                )

    fig.update_layout(
        title="",
        margin=dict(l=40, r=10, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title=axis_label)
    fig.update_yaxes(title=y_axis_title, range=[0, 1.05])
    _plotly_chart_scaled(fig, width_scale=0.9, use_container_width=True)


def _mf_tri_latex(name: str, lo: float, mid: float, hi: float, invert: bool) -> str:
    """Return LaTeX for a triangular membership with given params."""
    def fmt(x: float) -> str:
        return ("{:.3g}".format(x)).rstrip(".")

    safe = re.sub(r"[^A-Za-z0-9]+", " ", name).strip()
    base = rf"""
\mu_{{\text{{{safe}}}}}(x) = {'1 - ' if invert else ''}\begin{{cases}}
0, & x \le {fmt(lo)}\\
\frac{{x - {fmt(lo)}}}{{{fmt(mid)} - {fmt(lo)}}}, & {fmt(lo)} < x \le {fmt(mid)}\\
\frac{{{fmt(hi)} - x}}{{{fmt(hi)} - {fmt(mid)}}}, & {fmt(mid)} < x < {fmt(hi)}\\
0, & x \ge {fmt(hi)}
\end{{cases}}
"""
    return base

def _metric_sort_key(name: str) -> tuple:
    """Sort F-prefixed metrics numerically first, then others."""
    m = re.match(r"F(\d+)_", name)
    if m:
        return (0, int(m.group(1)), name)
    return (1, name)


def _normalize_uploaded_config(cfg: dict) -> dict:
    """Normalize an uploaded config to the app's internal schema.

    - Accept both "core" and "core_symptoms" and unify as "core_symptoms".
    - Accept per-metric "weight" as alias for "w".
    - Accept MF parameters either nested under "mf" or flattened (lo/mid/hi/invert/type at metric level).
    - Filter metrics to those available in this app per criterion.
    """
    if not isinstance(cfg, dict):
        return {}

    out: dict = {}
    # Gate-level keys
    if "M" in cfg:
        try:
            out["M"] = int(cfg.get("M"))
        except Exception:
            pass
    if "N" in cfg:
        try:
            out["N"] = int(cfg.get("N"))
        except Exception:
            pass
    if "theta" in cfg:
        try:
            out["theta"] = float(cfg.get("theta"))
        except Exception:
            pass
    # Core symptoms (support both keys)
    core_val = cfg.get("core_symptoms", cfg.get("core"))
    if core_val is not None:
        try:
            if isinstance(core_val, str):
                core_candidates = [core_val]
            elif isinstance(core_val, (list, tuple, set)):
                core_candidates = list(core_val)
            else:
                core_candidates = []
            valid = {"C1","C2","C3","C4","C5","C6","C7","C8","C9"}
            core_list = [str(x) for x in core_candidates if str(x) in valid]
            out["core_symptoms"] = core_list
        except Exception:
            pass

    # Per-criterion metrics
    # Support configs where criteria are nested under a "criteria" object
    crit_src = cfg.get("criteria") if isinstance(cfg.get("criteria"), dict) else cfg
    for crit in ["C1","C2","C3","C4","C5","C6","C7","C8","C9"]:
        if crit not in crit_src:
            continue
        crit_in = crit_src[crit]
        # If criterion object wraps metrics under a 'metrics' key, unwrap it
        if isinstance(crit_in, dict) and "metrics" in crit_in:
            inner = crit_in.get("metrics")
            if isinstance(inner, (dict, list)):
                crit_in = inner
        crit_out = {}
        available = set(ALL_METRIC_OPTIONS.get(crit, []))
        # crit_in can be a dict mapping metric->spec, or a list of entries
        if isinstance(crit_in, dict):
            items_iter = crit_in.items()
        elif isinstance(crit_in, list):
            # Convert list entries to (metric, spec) pairs
            tmp = []
            for entry in crit_in:
                if not isinstance(entry, dict):
                    continue
                mname = entry.get("metric") or entry.get("name") or entry.get("key")
                if not mname:
                    continue
                tmp.append((str(mname), entry))
            items_iter = tmp
        else:
            items_iter = []

        for m, spec in items_iter:
            if m not in available:
                # Skip unknown metrics to avoid UI inconsistencies
                continue
            if not isinstance(spec, dict):
                continue
            weight = spec.get("w", spec.get("weight", 0.1))
            try:
                weight = float(weight)
            except Exception:
                weight = 0.1
            mf_in = spec.get("mf", {})
            # Allow flattened parameters
            if not isinstance(mf_in, dict):
                mf_in = {}
            lo = spec.get("lo", mf_in.get("lo", 0.0))
            mid = spec.get("mid", mf_in.get("mid", 0.0))
            hi = spec.get("hi", mf_in.get("hi", 0.0))
            invert = spec.get("invert", mf_in.get("invert", False))
            typ = spec.get("type", mf_in.get("type", "tri"))
            # Robust bool parsing for invert
            if isinstance(invert, str):
                invert = invert.strip().lower() in {"true","1","yes","y","on"}
            else:
                invert = bool(invert)
            try:
                lo = float(lo)
            except Exception:
                lo = 0.0
            try:
                mid = float(mid)
            except Exception:
                mid = 0.0
            try:
                hi = float(hi)
            except Exception:
                hi = 0.0
            crit_out[m] = {
                "w": weight,
                "bom": _normalise_bom_value(spec.get("bom")) if isinstance(spec, dict) else "",
                "mf": {"type": str(typ).lower(), "lo": lo, "mid": mid, "hi": hi, "invert": invert},
            }
        if crit_out:
            out[crit] = crit_out

    _ensure_bom_field(out)
    return out



for idx, (crit, label) in enumerate(CRIT_TABS):
    with tabs[idx]:
        available = sorted(ALL_METRIC_OPTIONS.get(crit, []), key=_metric_sort_key)
        defaults = [m for m in available if m in cfg_state.get(crit, {})]
        selected = st.multiselect(
            "Select metrics",
            options=available,
            default=defaults,
            key=f"sel_{crit}",
        )
        cfg_state.setdefault(crit, {})
        # Auto-tune controls and diagnostics
        st.session_state.setdefault("fasl_autotune_debug", {})
        st.session_state.setdefault("fasl_autotuned_flags", {})
        with st.container(border=True):
            st.markdown("**Auto-tune parameters for each metric**")
            st.warning("⚠️ auto-tune feature does not work yet")
            col_auto_btn, col_auto_opts = st.columns([1, 3])
            with col_auto_btn:
                if st.button(f"Auto-tune {crit}", key=f"auto_{crit}"):
                    try:
                        ws_label = st.session_state.get(f"auto_w_{crit}", "Spread (P90-P10)")
                        _wmap = {"Spread (P90-P10)": "spread", "Variance": "variance", "Equal": "equal"}
                        ws = _wmap.get(ws_label, "spread")
                        res: AutoTuneResult = auto_tune_for_criterion(crit, selected, ALL_DAILY, HIW_MAP, weight_strategy=ws)
                    except Exception:
                        res: AutoTuneResult = auto_tune_for_criterion(crit, selected, ALL_DAILY, HIW_MAP)
                    try:
                        for m in [r.metric for r in res.metrics]:
                            cfg_state[crit].setdefault(m, {"w": 0.1, "mf": {"type": "tri", "lo": 0.0, "mid": 0.0, "hi": 0.0, "invert": False}})
                        for r in res.metrics:
                            cfg_state[crit][r.metric]["w"] = float(r.weight)
                            _hi = float(r.hi) if np.isfinite(float(r.hi)) else float(r.mid) + max(1e-6, abs(float(r.mid) - float(r.lo)) if np.isfinite(float(r.lo)) else 1.0)
                            cfg_state[crit][r.metric]["mf"] = {"type": "tri", "lo": float(r.lo), "mid": float(r.mid), "hi": float(_hi), "invert": bool(r.invert)}
                            st.session_state[f"w_{crit}_{r.metric}"] = float(r.weight)
                            st.session_state[f"lo_{crit}_{r.metric}"] = float(r.lo)
                            st.session_state[f"mid_{crit}_{r.metric}"] = float(r.mid)
                            st.session_state[f"hi_{crit}_{r.metric}"] = float(_hi)
                            st.session_state[f"inv_{crit}_{r.metric}"] = bool(r.invert)
                            st.session_state[f"mft_{crit}_{r.metric}"] = "tri"
                        st.session_state["fasl_autotune_debug"][crit] = res
                        st.session_state["fasl_autotuned_flags"][crit] = True
                        try:
                            st.toast("Auto-tuned parameters applied.")
                        except Exception:
                            st.success("Auto-tuned parameters applied.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Auto-tune failed: {e}")
                if st.session_state.get("fasl_autotuned_flags", {}).get(crit):
                    st.markdown("<span style='background:#dcfce7; color:#065f46; padding:3px 8px; border-radius:6px; font-size:0.85rem;'>auto-tuned</span>", unsafe_allow_html=True)
            with col_auto_opts:
                st.session_state.setdefault(f"auto_k_{crit}", 3)
                st.session_state.setdefault(f"auto_w_{crit}", "Spread (P90-P10)")
                st.session_state.setdefault(f"auto_z_{crit}", True)
                try:
                    with st.popover("Auto-tune options"):
                        st.slider("k clusters (PCA plot)", 2, 8, value=int(st.session_state.get(f"auto_k_{crit}", 3)), key=f"auto_k_{crit}")
                        st.selectbox("Weighting", ["Spread (P90-P10)", "Variance", "Equal"], key=f"auto_w_{crit}")
                        st.checkbox("Standardize (z-score) for PCA", value=bool(st.session_state.get(f"auto_z_{crit}", True)), key=f"auto_z_{crit}")
                except Exception:
                    pass
                _res = st.session_state.get("fasl_autotune_debug", {}).get(crit)
                if _res is not None:
                    with st.expander(f"Auto-tune diagnostics - {crit}", expanded=False):
                        res = _res
                        show_ov = st.checkbox("Show overlays (centers/boundaries)", value=True, key=f"diag_ov_{crit}")
                        for r in getattr(res, 'metrics', []) or []:
                            st.markdown(f"**{r.metric}**")
                            c1d, c2d = st.columns([2, 3])
                            with c1d:
                                try:
                                    inv_bg = "#dcfce7" if bool(r.invert) else "#e2e8f0"; inv_fg = "#065f46" if bool(r.invert) else "#334155"
                                    inv_txt = "Invert: True" if bool(r.invert) else "Invert: False"
                                    w_bg = "#eff6ff"; w_fg = "#1e3a8a"
                                    st.markdown(
                                        f"<div style='display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:6px;'>"
                                        f"<span style='background:{w_bg}; color:{w_fg}; padding:2px 8px; border-radius:6px; font-size:0.85rem;'>Weight: {r.weight:.3f}</span>"
                                        f"<span style='background:{inv_bg}; color:{inv_fg}; padding:2px 8px; border-radius:6px; font-size:0.85rem;'>{inv_txt}</span>"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )
                                except Exception:
                                    st.caption(f"Weight: {r.weight:.3f} • Invert: {bool(r.invert)}")
                                try:
                                    c1v, c2v, c3v = (float(r.centers[0]), float(r.centers[1]), float(r.centers[2]))
                                    b1, b2 = (float(r.boundaries[0]), float(r.boundaries[1]))
                                    sq = "&#9632;"
                                    chips = (
                                        f"<span style='color:#0ea5e9'>{sq}</span> center lo: {c1v:.3g}  "
                                        f"<span style='color:#f59e0b'>{sq}</span> center mid: {c2v:.3g}  "
                                        f"<span style='color:#ef4444'>{sq}</span> center hi: {c3v:.3g}<br>"
                                        f"<span style='color:#9333ea'>{sq}</span> t12: {b1:.3g}  <span style='color:#9333ea'>{sq}</span> t23: {b2:.3g}"
                                    )
                                    st.markdown(chips, unsafe_allow_html=True)
                                except Exception:
                                    pass
                                try:
                                    _hi_eval = float(r.hi if np.isfinite(r.hi) else r.mid)
                                    with st.popover("Details"):
                                        st.markdown(
                                            "- Centers (lo/mid/hi): we group daily values into three clusters (low, medium, high) and take the average of each cluster (the centers).\n"
                                            "- Thresholds (t12, t23): midpoints between low–mid and mid–high centers; they split the range.\n"
                                            "- Membership (mu in [0,1]): triangular shape with lo/mid/hi; near mid gives mu~1, far gives mu~0; 'invert' flips direction."
                                        )
                                        st.latex(r"c_{lo} < c_{mid} < c_{hi}")
                                        st.latex(r"t_{12} = \\tfrac{c_{lo} + c_{mid}}{2},\\quad t_{23} = \\tfrac{c_{mid} + c_{hi}}{2}")
                                        st.markdown("We map values to [0,1] via the triangular membership:")
                                        st.latex(_mf_tri_latex(r.metric, float(r.lo), float(r.mid), float(_hi_eval), bool(r.invert)))
                                except Exception:
                                    pass
                            with c2d:
                                try:
                                    tab_raw, tab_norm, tab_membership = st.tabs(["Raw", "Normalised", "Membership"])
                                    with tab_raw:
                                        _boxplot_with_ranges_marks(
                                            ALL_DAILY,
                                            r.metric,
                                            float(r.lo),
                                            float(r.mid),
                                            float(r.hi if np.isfinite(r.hi) else r.mid),
                                            invert=bool(r.invert),
                                            centers=(float(r.centers[0]), float(r.centers[1]), float(r.centers[2])),
                                            boundaries=(float(r.boundaries[0]), float(r.boundaries[1])),
                                            show_overlays=bool(show_ov),
                                            theta=float(theta_default),
                                        )
                                    with tab_norm:
                                        _boxplot_membership(
                                            ALL_DAILY,
                                            r.metric,
                                            float(r.lo),
                                            float(r.mid),
                                            float(r.hi if np.isfinite(r.hi) else r.mid),
                                            invert=bool(r.invert),
                                            theta=float(theta_default),
                                        )
                                    with tab_membership:
                                        _membership_curve_chart(
                                            ALL_DAILY,
                                            r.metric,
                                            float(r.lo),
                                            float(r.mid),
                                            float(r.hi if np.isfinite(r.hi) else r.mid),
                                            invert=bool(r.invert),
                                            theta=float(theta_default),
                                        )
                                except Exception:
                                    st.info("No values for plotting.")
        # Drop metrics not available anymore
        for m in list(cfg_state[crit].keys()):
            if m not in available or m not in selected:
                cfg_state[crit].pop(m)
        for m in selected:
            cfg_state[crit].setdefault(
                m,
                {
                    "w": 0.1,
                    "bom": "",
                    "mf": {
                        "type": "tri",
                        "lo": 0.0,
                        "mid": 0.0,
                        "hi": 0.0,
                        "invert": not HIW_MAP.get(m, True),
                    },
                },
            )
        for k in sorted(selected, key=_metric_sort_key):
            mf = cfg_state[crit][k].setdefault(
                "mf",
                {
                    "type": "tri",
                    "lo": 0.0,
                    "mid": 0.0,
                    "hi": 0.0,
                    "invert": not HIW_MAP.get(k, True),
                },
            )
            cfg_state[crit][k].setdefault("bom", "")
            cfg_state[crit][k]["bom"] = _normalise_bom_value(cfg_state[crit][k].get("bom"))
            with st.expander(f"**{k}**", expanded=False):
                c1, c2, c3 = st.columns([1, 1, 2], gap="small")
                with c1:
                    w = st.slider(
                        f"Weight – {k}", 0.0, 1.0, float(cfg_state[crit][k]["w"]), 0.05, key=f"w_{crit}_{k}"
                    )
                    cfg_state[crit][k]["w"] = float(w)
                    st.selectbox(
                        f"MF type – {k}", ["tri"], index=0, key=f"mft_{crit}_{k}",
                        help="Currently only triangular membership is supported.",
                    )

                    available_boms = CRIT_BOM_OPTIONS.get(crit, [])
                    bom_options = ["No BOM defined"] + available_boms
                    current_bom = _normalise_bom_value(cfg_state[crit][k].get("bom"))
                    default_option = current_bom if current_bom in available_boms else "No BOM defined"
                    bom_state_key = f"bom_{crit}_{k}"
                    if bom_state_key not in st.session_state:
                        st.session_state[bom_state_key] = default_option
                    elif st.session_state[bom_state_key] not in bom_options:
                        st.session_state[bom_state_key] = default_option
                    selection = st.selectbox(
                        f"BOM – {k}",
                        bom_options,
                        key=bom_state_key,
                        help="Select the behavioral observation mapping for this metric.",
                    )
                    cfg_state[crit][k]["bom"] = _normalise_bom_value(selection if selection != "No BOM defined" else "")

                    inv = st.checkbox(
                        f"invert – {k}",
                        value=bool(mf.get("invert", False)),
                        key=f"inv_{crit}_{k}",
                        help="Higher values indicate the criterion; check to flip if lower values should count as higher likelihood.",
                    )
                    mf["invert"] = bool(inv)
                    expl = MD_EXPLANATIONS.get(k)
                    if expl:
                        with st.popover("Details"):
                            st.markdown(expl)
                with c2:
                    hi_val = st.number_input(
                        f"hi – {k}", value=float(mf.get("hi", 0.0)), key=f"hi_{crit}_{k}"
                    )
                    mid_val = st.number_input(
                        f"mid – {k}", value=float(mf.get("mid", 0.0)), key=f"mid_{crit}_{k}"
                    )
                    lo_val = st.number_input(
                        f"lo – {k}", value=float(mf.get("lo", 0.0)), key=f"lo_{crit}_{k}"
                    )
                    mf["hi"], mf["mid"], mf["lo"] = float(hi_val), float(mid_val), float(lo_val)

                    if not (lo_val <= mid_val <= hi_val):
                        st.warning("Ensure lo <= mid <= hi.")
                with c3:
                    try:
                        tab_raw, tab_norm, tab_membership = st.tabs(["Raw", "Normalised", "Membership"])
                        with tab_raw:
                            _boxplot_with_ranges(
                                ALL_DAILY,
                                k,
                                float(mf.get("lo", 0.0)),
                                float(mf.get("mid", 0.0)),
                                float(mf.get("hi", 0.0)),
                                invert=bool(mf.get("invert", False)),
                                theta=float(theta_default),
                            )
                        with tab_norm:
                            _boxplot_membership(
                                ALL_DAILY,
                                k,
                                float(mf.get("lo", 0.0)),
                                float(mf.get("mid", 0.0)),
                                float(mf.get("hi", 0.0)),
                                invert=bool(mf.get("invert", False)),
                                theta=float(theta_default),
                            )
                        with tab_membership:
                            _membership_curve_chart(
                                ALL_DAILY,
                                k,
                                float(mf.get("lo", 0.0)),
                                float(mf.get("mid", 0.0)),
                                float(mf.get("hi", 0.0)),
                                invert=bool(mf.get("invert", False)),
                                theta=float(theta_default),
                            )
                    except Exception:
                        pass
                with c1:
                    st.latex(
                        _mf_tri_latex(
                            k,
                            mf.get("lo", 0.0),
                            mf.get("mid", 0.0),
                            mf.get("hi", 0.0),
                            mf.get("invert", False),
                        )
                    )



# -------------------------- Evaluate & Visualize ------------------------------

# Show guidance if no per-criterion model configuration exists yet
_has_model_cfg = any(
    isinstance(cfg_state.get(c), dict) and len(cfg_state.get(c)) > 0 for c in CRIT_KEYS
)
if not _has_model_cfg:
    st.info('Please create a model configuration or upload a configuration in JSON format.')


# Time series (bottom, full width) in a bordered container
with st.container(border=True):
    try:
        import plotly.express as px

        melted = DF_L.melt(id_vars=["Date"], var_name="Criterion", value_name="Likelihood")
        try:
            melted["Code"] = melted["Criterion"].str.replace("L_", "", regex=False)
            melted["Display"] = melted["Code"].map(CRIT_DISPLAY).fillna(melted["Code"])
        except Exception:
            melted["Display"] = melted["Criterion"]
        fig = px.line(
            melted,
            x="Date",
            y="Likelihood",
            color="Display",
            title="Criterion likelihoods over time",
        )
        fig.update_yaxes(range=[0, 1])
        fig.add_hrect(
            y0=0,
            y1=theta_default,
            line_width=0,
            fillcolor="rgba(34,197,94,0.10)",
            layer="below",
        )
        fig.add_hrect(
            y0=theta_default,
            y1=1,
            line_width=0,
            fillcolor="rgba(239,68,68,0.10)",
            layer="below",
        )
        fig.add_hline(y=theta_default, line_dash="dot", line_color="gray")
        st.plotly_chart(fig, use_container_width=True, key="crit_ts")
        # Popover placed below the time series plot
        try:
            with st.popover("Recent per-day evaluations"):
                st.dataframe(DF_L.tail(30), use_container_width=True)
        except Exception:
            pass
    except Exception:
        pass





st.write("")
with st.container(border=True):
    st.subheader("Configuration Summary & JSON Export")
    st.caption("Compare your current FASL settings with the built-in defaults and share them as JSON.")

    total_metrics = sum(len(cfg_state.get(crit, {})) for crit in CRIT_KEYS)
    adjusted_metrics = sum(
        1
        for crit in CRIT_KEYS
        for metric, spec in (cfg_state.get(crit, {}) or {}).items()
        if _metric_differs_from_default(crit, metric, spec)
    )
    added_metrics = sum(
        1
        for crit in CRIT_KEYS
        for metric in (cfg_state.get(crit, {}) or {}).keys()
        if metric not in (DEFAULT_CFG.get(crit, {}) or {})
    )

    st.markdown(f"**Metrics configured:** {total_metrics}")
    st.markdown(f"**Adjusted vs defaults:** {adjusted_metrics}")
    if added_metrics:
        st.caption(f"{added_metrics} metric(s) are not part of the default configuration.")
    st.caption(f"Default config path: `{FASL_CONFIG_PATH}`")

    config_json = json.dumps(cfg_state, indent=2, ensure_ascii=False)
    download_col, save_col, upload_col = st.columns(3)
    with download_col:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "Download configuration",
            data=config_json.encode("utf-8"),
            file_name=f"fasl_config_{ts}.json",
            mime="application/json",
        )
    with save_col:
        if st.button("Save current configuration as default", key="fasl_save_cfg"):
            try:
                FASL_CONFIG_PATH.write_text(config_json, encoding="utf-8")
                st.success(f"Saved to {FASL_CONFIG_PATH}")
            except Exception as exc:
                st.error(f"Failed to write configuration: {exc}")
    with upload_col:
        up = st.file_uploader("Upload configuration (.json)", type=["json"], key="fasl_cfg_upload_footer")
        if up is not None:
            try:
                uploaded_cfg_raw = json.loads(up.read().decode("utf-8"))
                uploaded_cfg = _normalize_uploaded_config(uploaded_cfg_raw)
                if not isinstance(uploaded_cfg, dict):
                    raise ValueError("JSON must be an object")
                st.success("Configuration parsed. Apply to use.")
                if st.button("Apply configuration", key="apply_cfg_footer"):
                    cfg_state.clear()
                    cfg_state.update(uploaded_cfg)
                    _ensure_bom_field(cfg_state)
                    try:
                        st.session_state["fasl_gate_M"] = int(cfg_state.get("M", 14))
                        st.session_state["fasl_gate_N"] = int(cfg_state.get("N", 10))
                        st.session_state["fasl_gate_theta"] = float(cfg_state.get("theta", 0.7))
                        st.session_state["fasl_gate_core"] = list(cfg_state.get("core_symptoms", ["C2"]))
                        for _crit in CRIT_KEYS:
                            selected_metrics = [
                                m for m in cfg_state.get(_crit, {}).keys() if m in ALL_METRIC_OPTIONS.get(_crit, [])
                            ]
                            st.session_state[f"sel_{_crit}"] = selected_metrics
                            for _m in selected_metrics:
                                _spec = cfg_state[_crit].get(_m, {}) or {}
                                _mf = _spec.get("mf", {}) or {}
                                st.session_state[f"w_{_crit}_{_m}"] = float(_spec.get("w", 0.1))
                                st.session_state[f"lo_{_crit}_{_m}"] = float(_mf.get("lo", 0.0))
                                st.session_state[f"mid_{_crit}_{_m}"] = float(_mf.get("mid", 0.0))
                                st.session_state[f"hi_{_crit}_{_m}"] = float(_mf.get("hi", 0.0))
                                st.session_state[f"inv_{_crit}_{_m}"] = bool(_mf.get("invert", False))
                                st.session_state[f"mft_{_crit}_{_m}"] = "tri"
                                bom_val = _normalise_bom_value(_spec.get("bom"))
                                bom_options = CRIT_BOM_OPTIONS.get(_crit, [])
                                st.session_state[f"bom_{_crit}_{_m}"] = bom_val if bom_val in bom_options else "No BOM defined"
                    except Exception:
                        pass
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to load: {e}")


# Gate decisions per criterion
present = {}
for crit in crit_cols:
    s = DF_L[f"L_{crit}"] if f"L_{crit}" in DF_L.columns else pd.Series(dtype=float)
    present[crit] = gate_present(s, theta=theta_default, need_days=gate_need, window=gate_window)

with st.status("Evaluating DSM‑Gate…", expanded=False) as s3:
    s3.update(label="Computed per‑criterion presence flags.", state="complete")

# Depression episode decision
core_ok = any(present.get(c, False) for c in core_criteria)
total_present = int(sum(1 for v in present.values() if v))
episode_likely = core_ok and (total_present >= 5)

# Summary cards: horizontally aligned with conditional backgrounds (no borders)
GREEN_BG = "#d1fae5"
RED_BG = "#fee2e2"
cols_summary = st.columns(3)

with cols_summary[0]:
    st.empty()

with cols_summary[1]:
    st.empty()

with cols_summary[2]:
    st.empty()


# --------------------------- Contribution plots --------------------------------
with st.expander("Per‑criterion contribution over time", expanded=False):
    try:
        import plotly.express as px
        for crit in CRIT_KEYS:
            spec = cfg_state.get(crit, {})
            if not spec:
                continue
            records: list[dict[str, Any]] = []
            for _, row in ALL_DAILY.iterrows():
                date = pd.to_datetime(row["Date"])
                for k, cfg in spec.items():
                    w = float(cfg.get("w", 0.0))
                    mf = cfg.get("mf", {})
                    invert = bool(mf.get("invert", False))
                    x = row.get(k)
                    lo = float(mf.get("lo", 0.0))
                    mid = float(mf.get("mid", 0.0))
                    hi = float(mf.get("hi", 0.0))
                    mu = mf_tri(x, lo, mid, hi, invert=invert)
                    records.append({
                        "Date": date,
                        "Metric": k,
                        "Contribution": float(w * mu),
                    })
            if records:
                dfp = pd.DataFrame(records)
                fig = px.line(
                    dfp,
                    x="Date",
                    y="Contribution",
                    color="Metric",
                    title=f"{crit}: weight×membership contributions",
                )
                st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass
