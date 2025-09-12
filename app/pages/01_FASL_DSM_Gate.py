# pages/01_FASL_DSM_Gate.py

from __future__ import annotations

import os, re, json, hashlib, copy
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from metrics.base_features import compute_daily_base_record
from metrics.common import enrich_with_hostnames
from md_explanations import MD_EXPLANATIONS
from metrics.criterion1 import C1_DEFS
from metrics.criterion2 import C2_DEFS
from metrics.criterion3 import C3_DEFS
from metrics.criterion4 import C4_DEFS
from metrics.criterion5 import C5_DEFS
from metrics.criterion6 import C6_DEFS
from metrics.criterion7 import C7_DEFS
from metrics.criterion8 import C8_DEFS
from metrics.criterion9 import C9_DEFS

CRIT_KEYS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

# ------------------------------ Page header / Auth -----------------------------

st.set_page_config(
    page_title="PCAP Analyzer for Behavioral Research",
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



def login():
    st.title("PCAP Analyzer for Behavioral Research")
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
    st.title("Fuzzy‑Additive Symptom Likelihood (FASL) + DSM‑Gate")
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

PROCESSED_DIR = "processed_parquet"
FEATURE_CACHE_DIR = "feature_cache"
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)


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
    rows: List[dict] = []
    for d in days:
        frames = []
        for bn in base_names:
            df_b = load_day_dataframe(bn, d)
            if not df_b.empty:
                frames.append(df_b)
        if not frames:
            continue
        df_day = pd.concat(frames, ignore_index=True)
        df_day["Timestamp"] = pd.to_datetime(df_day["Timestamp"], errors="coerce")
        today = compute_daily_base_record(df_day)
        today["Date"] = pd.to_datetime(d).normalize()
        rows.append(today)
    return pd.DataFrame(rows).sort_values("Date") if rows else pd.DataFrame(columns=["Date"])


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
        # Remove DQI sentence
        _md = re.sub(r"^> \*\*Data quality \(DQI\)\*\*:[^\n]*\n", "", _md, flags=re.M)
        # Remove the entire 'Second example: Anhedonia' section up to the next horizontal rule
        _md = re.sub(r"^## Second example:.*?(?:\n---\n)", "", _md, flags=re.S | re.M)
        # Fix any stray $ inside the display equation if still present
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
    selected_group_tokens = st.multiselect(
        "Select dataset groups (prefix match)",
        options=group_display_options,
        key="fasl_group_tokens",
        default=["[ALL OTHER]"]
    )

auto_selected_from_groups: set[str] = set()
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
if not selected_base_names:
    st.info("No datasets selected. Choose dataset type(s) and group(s) in the sidebar.")
    st.stop()

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

# All‑days aufbauen (leichtgewichtige Rekonstruktion)
with st.status("Indexing available days for the current selection…", expanded=False) as s1:
    days = available_days_for(selected_base_names)
    if not days:
        s1.update(
            label="No 5‑minute partitions found for the current selection.", state="error"
        )
        st.stop()
    s1.update(label=f"Found {len(days)} day(s).", state="complete")


def _cache_key_for_selection(base_names: list[str]) -> str:
    return hashlib.md5("|".join(sorted(base_names)).encode("utf-8")).hexdigest()


def _cache_path_for_selection(base_names: list[str]) -> str:
    return os.path.join(
        FEATURE_CACHE_DIR, f"features_{_cache_key_for_selection(base_names)}.csv"
    )


with st.status(
    "Building per‑day base features (ALL_DAILY)…", expanded=False
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

    if loaded.empty or missing:
        add_rows = compute_all_daily(selected_base_names, missing if missing else days)
        ALL_DAILY = (
            pd.concat([loaded, add_rows], ignore_index=True)
            if not loaded.empty
            else add_rows
        ).sort_values("Date")
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


    if ALL_DAILY.empty:
        st.error("No day metrics available")
        st.stop()

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
    "C1": "C1 – Sleep disturbance",
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

# Summary metrics directly below the subtitle and gauge tabs (average/day)
# Compute DSM-Gate presence flags for summary metrics
present = {}
for crit in crit_cols:
    s = DF_L[f"L_{crit}"] if f"L_{crit}" in DF_L.columns else pd.Series(dtype=float)
    present[crit] = gate_present(s, theta=theta_default, need_days=gate_need, window=gate_window)

# Episode decision summary
core_ok = any(present.get(c, False) for c in core_criteria)
total_present = int(sum(1 for v in present.values() if v))
episode_likely = core_ok and (total_present >= 5)

# Render summary cards
st.write("---")
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
    c2_ok = bool(present.get("C2", False))
    bg = RED_BG if c2_ok else GREEN_BG
    st.markdown(
        f"""
        <div style=\"background-color:{bg}; border-radius:8px; padding:10px;\">
          <div style=\"font-size:0.9rem; opacity:0.7;\">Core symptom satisfied</div>
          <div style=\"font-size:1.4rem; font-weight:600;\">{'Yes' if c2_ok else 'No'}</div>
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


st.write("---")
st.subheader("Criterion Likelihoods and DSM-Gate")
st.write(
    f"""
    The DSM‑Gate uses a rolling window M={int(cfg_state.get('M', 14))} days and requires N={int(cfg_state.get('N', 10))} days ≥ θ to mark a criterion as present.
    A depressive episode is likely when ≥5 criteria are present and at least one core symptom ({', '.join(cfg_state.get('core_symptoms', ['C2']))}) is present.
    """
)

with st.expander("DSM-5 Diagnostic Reference", expanded=False):
    st.markdown(
        """
    DSM‑5 describes a Major Depressive Episode as having at least 5 symptoms present during the same 2‑week period, and at least one is either depressed mood or markedly diminished interest/pleasure (anhedonia). This page implements a transparent, rule‑based approximation: daily likelihoods (L_k) aggregated over a rolling window M with threshold θ and count N. Tuning M/N/θ adjusts sensitivity while staying faithful to the spirit of the DSM‑5 criteria.
    """
    )


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
    ("C1", "C1 – Sleep disturbance"),
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
    "C1": "C1 – Sleep disturbance",
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
                "mf": {"type": str(typ).lower(), "lo": lo, "mid": mid, "hi": hi, "invert": invert},
            }
        if crit_out:
            out[crit] = crit_out

    return out





c1, c2 = st.columns(2)
with c1:
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "Export configuration as JSON",
        data=json.dumps(cfg_state, indent=2).encode("utf-8"),
        file_name=f"fasl_config_{ts}.json",
        mime="application/json",
    )

    up = st.file_uploader("Import JSON", type=["json"], accept_multiple_files=False)
    if up is not None:
        try:
            uploaded_cfg_raw = json.loads(up.read().decode("utf-8"))
            uploaded_cfg = _normalize_uploaded_config(uploaded_cfg_raw)
            if not isinstance(uploaded_cfg, dict):
                raise ValueError("JSON must be an object")
            st.success("Configuration parsed. Apply to use.")
            if st.button("Apply configuration", key="apply_cfg"):
                cfg_state.clear()
                cfg_state.update(uploaded_cfg)
                # Ensure widget state reflects the applied configuration on rerun
                try:
                    st.session_state["fasl_gate_M"] = int(cfg_state.get("M", 14))
                    st.session_state["fasl_gate_N"] = int(cfg_state.get("N", 10))
                    st.session_state["fasl_gate_theta"] = float(cfg_state.get("theta", 0.7))
                    st.session_state["fasl_gate_core"] = list(cfg_state.get("core_symptoms", ["C2"]))
                    for _crit in CRIT_KEYS:
                        # Select exactly the metrics from the uploaded config (filtered to available)
                        selected_metrics = [
                            m for m in cfg_state.get(_crit, {}).keys() if m in ALL_METRIC_OPTIONS.get(_crit, [])
                        ]
                        st.session_state[f"sel_{_crit}"] = selected_metrics
                        # Also push their parameter values into widget state so UI reflects JSON precisely
                        for _m in selected_metrics:
                            _spec = cfg_state[_crit].get(_m, {})
                            _w = float(_spec.get("w", 0.1))
                            _mf = _spec.get("mf", {})
                            _lo = float(_mf.get("lo", 0.0))
                            _mid = float(_mf.get("mid", 0.0))
                            _hi = float(_mf.get("hi", 0.0))
                            _inv = bool(_mf.get("invert", False))
                            st.session_state[f"w_{_crit}_{_m}"] = _w
                            st.session_state[f"lo_{_crit}_{_m}"] = _lo
                            st.session_state[f"mid_{_crit}_{_m}"] = _mid
                            st.session_state[f"hi_{_crit}_{_m}"] = _hi
                            st.session_state[f"inv_{_crit}_{_m}"] = _inv
                            # Membership function type (currently only 'tri')
                            st.session_state[f"mft_{_crit}_{_m}"] = "tri"
                except Exception:
                    pass
                st.rerun()
        except Exception as e:
            st.error(f"Failed to load: {e}")


cfg_state = st.session_state.setdefault("fasl_cfg", {})
for _k in ("M", "N", "theta", "core_symptoms"):
    if _k not in cfg_state:
        _v = DEFAULT_CFG.get(_k)
        cfg_state[_k] = copy.deepcopy(_v) if isinstance(_v, (dict, list)) else _v


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


def _boxplot_with_ranges(
    df: pd.DataFrame, col: str, lo: float, mid: float, hi: float, invert: bool = False
):
    import plotly.express as px, plotly.graph_objects as go

    if col not in df.columns or df[col].dropna().empty:
        st.info("No historical values available for this metric.")
        return
    series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    fig = px.box(series, points="all")
    fig.update_layout(title_text="")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(title=None)
    # Clip infinite bounds to data range for visualization
    ymin = float(series.min())
    ymax = float(series.max())
    lo_draw = float(lo) if np.isfinite(lo) else ymin
    hi_draw = float(hi) if np.isfinite(hi) else ymax
    try:
        good = "rgba(34,197,94,0.10)"
        bad = "rgba(239,68,68,0.10)"
        if lo_draw <= mid:
            fig.add_hrect(
                y0=lo_draw,
                y1=mid,
                line_width=0,
                fillcolor=bad if invert else good,
                layer="below",
            )
        if mid <= hi_draw:
            fig.add_hrect(
                y0=mid,
                y1=hi_draw,
                line_width=0,
                fillcolor=good if invert else bad,
                layer="below",
            )
        fig.add_hline(y=mid, line_dash="dot", line_color="gray")
    except Exception:
        pass
    st.plotly_chart(fig, use_container_width=True)


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
                "mf": {"type": str(typ).lower(), "lo": lo, "mid": mid, "hi": hi, "invert": invert},
            }
        if crit_out:
            out[crit] = crit_out

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
        # Drop metrics not available anymore
        for m in list(cfg_state[crit].keys()):
            if m not in available or m not in selected:
                cfg_state[crit].pop(m)
        for m in selected:
            cfg_state[crit].setdefault(
                m,
                {
                    "w": 0.1,
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
            with st.expander(f"**{k}**", expanded=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    w = st.slider(
                        f"Weight – {k}", 0.0, 1.0, float(cfg_state[crit][k]["w"]), 0.05, key=f"w_{crit}_{k}"
                    )
                    cfg_state[crit][k]["w"] = float(w)
                    st.selectbox(
                        f"MF type – {k}", ["tri"], index=0, key=f"mft_{crit}_{k}",
                        help="Currently only triangular membership is supported.",
                    )
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
                    if not (lo_val < mid_val < hi_val):
                        st.warning("Ensure lo < mid < hi.")
                with c3:
                    try:
                        _boxplot_with_ranges(
                            ALL_DAILY,
                            k,
                            mf.get("lo", 0.0),
                            mf.get("mid", 0.0),
                            mf.get("hi", 0.0),
                            mf.get("invert", False),
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
