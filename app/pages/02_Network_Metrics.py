# pages/02_Network_Metrics.py

from __future__ import annotations

import os
import sys
import re
import hashlib
import json
import copy
import math
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pyarrow as pa
import pyarrow.parquet as pq
from utils.acronyms import render_acronyms_helper_in_sidebar


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ---- criterion implementations ----
from metrics.criterion1 import Criterion1, C1_DEFS
from metrics.criterion2 import Criterion2, C2_DEFS
from metrics.criterion3 import Criterion3, C3_DEFS
from metrics.criterion4 import Criterion4, C4_DEFS
from metrics.criterion5 import Criterion5, C5_DEFS
from metrics.criterion6 import Criterion6, C6_DEFS
from metrics.criterion7 import Criterion7, C7_DEFS
from metrics.criterion8 import Criterion8, C8_DEFS
from metrics.criterion9 import Criterion9, C9_DEFS

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
    username = st.text_input("Username", placeholder="username")
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

FASL_CONFIG_PATH = APP_DIR / "fasl_config.json"
DEFAULT_FASL_CONFIG_PATH = APP_DIR / "utils" / "fasl_config_20250912_0946.json"
CRITERION_CODES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
CRIT_KEYS = CRITERION_CODES

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
        "(HOB1) Guilt-tripping / Negative Self-talk",
        "(HOB2) Catastrophic Outlook",
        "(HOB3) Reputation / Account Security Anxiety",
    ],
    "C8": [
        "(HOB1) Fragmented Browsing / Task Switching",
        "(HOB2) Escalating Search Reformulation",
        "(HOB3) Decision Paralysis via Backtracking",
    ],
    "C9": [
        "(HOB1) Active Crisis Searching",
        "(HOB2) Self-harm Community Immersion",
        "(HOB3) End-of-life Administrative Planning",
    ],
}


def _ensure_bom_field(cfg: dict) -> None:
    for crit_val in cfg.values():
        if isinstance(crit_val, dict):
            for spec in crit_val.values():
                if isinstance(spec, dict) and "w" in spec:
                    spec["bom"] = _normalise_bom_value(spec.get("bom"))


def _get_default_cfg() -> dict:
    defaults = st.session_state.get("__fasl_cfg_default__")
    return defaults if isinstance(defaults, dict) else {}


def _metric_differs_from_default(crit: str, metric: str, spec: dict) -> bool:
    defaults = _get_default_cfg()
    default_bucket = defaults.get(crit)
    default = default_bucket.get(metric) if isinstance(default_bucket, dict) else None
    if default is None:
        return True
    if not isinstance(spec, dict):
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



def _safe_float_config(value) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    if np.isnan(v):
        return None
    return v


def _get_fasl_membership(criterion: str | None, metric_key: str | None) -> tuple[float, float, float, bool]:
    if not criterion or not metric_key:
        return 0.0, 0.0, 0.0, False
    cfg = st.session_state.get("fasl_cfg", {})
    crit_bucket = cfg.get(criterion, {}) if isinstance(cfg, dict) else {}
    spec = crit_bucket.get(metric_key, {}) if isinstance(crit_bucket, dict) else {}
    if not isinstance(spec, dict):
        spec = {}
    mf = spec.get("mf") if isinstance(spec.get("mf"), dict) else {}
    lo = _safe_float_config(mf.get("lo")) or 0.0
    mid = _safe_float_config(mf.get("mid")) or 0.0
    hi = _safe_float_config(mf.get("hi")) or 0.0
    invert = bool(mf.get("invert", False))
    return lo, mid, hi, invert


def _apply_tri_background(fig, lo, mid, hi, invert, y_min, y_max):
    import math
    vals = (lo, mid, hi)
    if all((not np.isfinite(v)) or math.isclose(float(v), 0.0, abs_tol=1e-9) for v in vals):
        return
    if y_min is None or not np.isfinite(y_min):
        y_min = float('-inf')
    if y_max is None or not np.isfinite(y_max):
        y_max = float('inf')
    if not np.isfinite(mid):
        return
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
    fig.add_hline(y=mid_draw, line_dash="dot", line_color="gray")



def _normalize_fasl_cfg(cfg: dict | None) -> dict:
    if not isinstance(cfg, dict):
        return {}
    normalized: dict[str, dict] = {}
    general_keys = {"M", "N", "theta"}
    for key, value in cfg.items():
        if key in general_keys:
            normalized[key] = value
            continue
        if key == "core_symptoms":
            if isinstance(value, (list, tuple, set)):
                normalized[key] = list(value)
            else:
                normalized[key] = []
            continue
        if isinstance(key, str) and len(key) == 2 and key.startswith("C") and key[1].isdigit():
            bucket: dict[str, dict] = {}
            if isinstance(value, dict):
                for metric_key, spec in value.items():
                    if not isinstance(spec, dict):
                        continue
                    spec_out: dict[str, object] = {}
                    if "w" in spec:
                        w_val = _safe_float_config(spec.get("w"))
                        spec_out["w"] = w_val if w_val is not None else spec.get("w", 0.1)
                    if "bom" in spec:
                        spec_out["bom"] = spec.get("bom")
                    mf_raw = spec.get("mf") if isinstance(spec.get("mf"), dict) else {}
                    lo = _safe_float_config(mf_raw.get("lo"))
                    mid = _safe_float_config(mf_raw.get("mid"))
                    hi = _safe_float_config(mf_raw.get("hi"))
                    spec_out["mf"] = {
                        "type": (mf_raw.get("type") or "tri"),
                        "lo": lo if lo is not None else mf_raw.get("lo", 0.0),
                        "mid": mid if mid is not None else mf_raw.get("mid", 0.0),
                        "hi": hi if hi is not None else mf_raw.get("hi", 0.0),
                        "invert": bool(mf_raw.get("invert", False)),
                    }
                    bucket[str(metric_key)] = spec_out
            normalized[key] = bucket
            continue
        normalized[key] = value
    return normalized


def _load_default_fasl_config() -> dict:
    if DEFAULT_FASL_CONFIG_PATH.exists():
        try:
            data = json.loads(DEFAULT_FASL_CONFIG_PATH.read_text(encoding='utf-8'))
            return _normalize_fasl_cfg(data)
        except Exception:
            return {}
    return {}


def _load_fasl_config_from_disk() -> tuple[dict | None, str | None]:
    if not FASL_CONFIG_PATH.exists():
        return None, None
    try:
        data = json.loads(FASL_CONFIG_PATH.read_text(encoding='utf-8'))
        return _normalize_fasl_cfg(data), None
    except Exception as exc:
        return None, str(exc)


def _ensure_fasl_config_state():
    default_cfg = st.session_state.setdefault("__fasl_cfg_default__", copy.deepcopy(_load_default_fasl_config()))
    fasl_state = st.session_state.get("fasl_cfg")
    if not isinstance(fasl_state, dict):
        cfg, err = _load_fasl_config_from_disk()
        if err:
            st.session_state["__fasl_cfg_load_error__"] = err
        if cfg is None:
            cfg = copy.deepcopy(default_cfg)
            st.session_state["__fasl_cfg_source__"] = "defaults"
        else:
            st.session_state["__fasl_cfg_source__"] = "disk"
        st.session_state["fasl_cfg"] = copy.deepcopy(cfg)
    else:
        st.session_state.setdefault("__fasl_cfg_source__", "session")

    cfg_ref = st.session_state.get("fasl_cfg")
    if isinstance(cfg_ref, dict):
        try:
            st.session_state.setdefault("fasl_gate_M", int(cfg_ref.get("M", 14)))
        except Exception:
            st.session_state.setdefault("fasl_gate_M", 14)
        try:
            st.session_state.setdefault("fasl_gate_N", int(cfg_ref.get("N", 10)))
        except Exception:
            st.session_state.setdefault("fasl_gate_N", 10)
        try:
            st.session_state.setdefault("fasl_gate_theta", float(cfg_ref.get("theta", 0.7)))
        except Exception:
            st.session_state.setdefault("fasl_gate_theta", 0.7)
        core_default = cfg_ref.get("core_symptoms", ["C2"])
        if isinstance(core_default, (list, tuple, set)):
            core_list = [str(c) for c in core_default if str(c) in CRIT_KEYS]
        elif isinstance(core_default, str) and core_default in CRIT_KEYS:
            core_list = [core_default]
        else:
            core_list = ["C2"]
        st.session_state.setdefault("fasl_gate_core", core_list)

        for crit in CRIT_KEYS:
            sel_key = f"sel_{crit}"
            if isinstance(st.session_state.get(sel_key), list):
                continue
            bucket = cfg_ref.get(crit, {})
            if not isinstance(bucket, dict):
                st.session_state.setdefault(sel_key, [])
                continue
            allowed = set(ALL_METRIC_OPTIONS.get(crit, []))
            default_metrics = [m for m in bucket.keys() if m in allowed]
            st.session_state.setdefault(sel_key, default_metrics)
            for metric_key in default_metrics:
                spec = bucket.get(metric_key)
                if not isinstance(spec, dict):
                    continue
                try:
                    st.session_state.setdefault(f"w_{crit}_{metric_key}", float(spec.get("w", 0.1)))
                except Exception:
                    st.session_state.setdefault(f"w_{crit}_{metric_key}", 0.1)
                mf = spec.get("mf", {}) if isinstance(spec.get("mf"), dict) else {}
                try:
                    st.session_state.setdefault(f"lo_{crit}_{metric_key}", float(mf.get("lo", 0.0)))
                except Exception:
                    st.session_state.setdefault(f"lo_{crit}_{metric_key}", 0.0)
                try:
                    st.session_state.setdefault(f"mid_{crit}_{metric_key}", float(mf.get("mid", 0.0)))
                except Exception:
                    st.session_state.setdefault(f"mid_{crit}_{metric_key}", 0.0)
                try:
                    st.session_state.setdefault(f"hi_{crit}_{metric_key}", float(mf.get("hi", 0.0)))
                except Exception:
                    st.session_state.setdefault(f"hi_{crit}_{metric_key}", 0.0)
                invert_val = bool(mf.get("invert", False))
                st.session_state.setdefault(f"inv_{crit}_{metric_key}", invert_val)
                st.session_state.setdefault(f"mft_{crit}_{metric_key}", str(mf.get("type", "tri")).lower())
                bom_val = _normalise_bom_value(spec.get("bom"))
                bom_options = CRIT_BOM_OPTIONS.get(crit, [])
                st.session_state.setdefault(
                    f"bom_{crit}_{metric_key}",
                    bom_val if bom_val in bom_options else "No BOM defined",
                )


def _normalize_uploaded_config(cfg: dict) -> dict:
    normalized = _normalize_fasl_cfg(cfg)
    if not isinstance(normalized, dict):
        return {}

    out: dict[str, object] = {}
    for key in ("M", "N", "theta"):
        if key in normalized:
            out[key] = normalized[key]

    core = normalized.get("core_symptoms", [])
    if isinstance(core, (list, tuple, set)):
        out["core_symptoms"] = [str(c) for c in core if str(c) in CRIT_KEYS]
    elif isinstance(core, str) and core in CRIT_KEYS:
        out["core_symptoms"] = [core]

    for crit in CRIT_KEYS:
        bucket = normalized.get(crit)
        if not isinstance(bucket, dict):
            continue
        allowed = set(ALL_METRIC_OPTIONS.get(crit, []))
        filtered: dict[str, dict] = {}
        for metric, spec in bucket.items():
            if metric not in allowed or not isinstance(spec, dict):
                continue
            spec_copy = copy.deepcopy(spec)
            spec_copy["bom"] = _normalise_bom_value(spec_copy.get("bom"))
            filtered[metric] = spec_copy
        if filtered:
            out[crit] = filtered

    _ensure_bom_field(out)
    return out


def _ensure_fasl_metric_entry(criterion: str, metric_key: str) -> dict:
    cfg = st.session_state.setdefault("fasl_cfg", {})
    if not isinstance(cfg, dict):
        cfg = {}
        st.session_state["fasl_cfg"] = cfg
    crit_bucket = cfg.setdefault(criterion, {})
    if not isinstance(crit_bucket, dict):
        crit_bucket = {}
        cfg[criterion] = crit_bucket
    spec = crit_bucket.setdefault(metric_key, {})
    if not isinstance(spec, dict):
        spec = {}
        crit_bucket[metric_key] = spec
    spec.setdefault("w", 0.1)
    mf = spec.get("mf")
    if not isinstance(mf, dict):
        mf = {}
        spec["mf"] = mf
    mf.setdefault("type", "tri")
    mf.setdefault("lo", 0.0)
    mf.setdefault("mid", 0.0)
    mf.setdefault("hi", 0.0)
    mf.setdefault("invert", False)
    return spec


def _fasl_default_for(criterion: str, metric_key: str) -> dict | None:
    defaults = st.session_state.get("__fasl_cfg_default__", {})
    if not isinstance(defaults, dict):
        return None
    crit_defaults = defaults.get(criterion)
    if not isinstance(crit_defaults, dict):
        return None
    value = crit_defaults.get(metric_key)
    if not isinstance(value, dict):
        return None
    return copy.deepcopy(value)


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

_ensure_fasl_config_state()

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

@st.dialog("Metric details", width="large")
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
    criterion = payload.get("criterion")
    base_cfg = payload.get("range_cfg") or {}
    current_value = payload.get("current_value")
    ts_df = payload.get("ts_df")

    effective_cfg = get_effective_range_cfg(label, dist_col, base_cfg)
    tri_lo, tri_mid, tri_hi, tri_invert = _get_fasl_membership(criterion, dist_col)
    tri_all_zero = all(abs(v) < 1e-9 for v in (tri_lo, tri_mid, tri_hi))

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

                    if not tri_all_zero:
                        _apply_tri_background(fig_w, tri_lo, tri_mid, tri_hi, tri_invert, y_min, y_max)

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

                    if not tri_all_zero:
                        _apply_tri_background(fig_t, tri_lo, tri_mid, tri_hi, tri_invert, y_min, y_max)

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

                    if not tri_all_zero:
                        _apply_tri_background(fig_d, tri_lo, tri_mid, tri_hi, tri_invert, y_min, y_max)

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
                        if not tri_all_zero:
                            _apply_tri_background(fig_box, tri_lo, tri_mid, tri_hi, tri_invert, yb_min, yb_max)

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

    st.markdown("### Triangular membership (FASL config)")
    with st.container(border=True):
        if not criterion or not dist_col:
            st.info("Triangular membership parameters are not available for this metric.")
        else:
            spec = _ensure_fasl_metric_entry(criterion, dist_col)
            mf = spec.get("mf") if isinstance(spec.get("mf"), dict) else {}
            lo_current = _safe_float_config(mf.get("lo"))
            mid_current = _safe_float_config(mf.get("mid"))
            hi_current = _safe_float_config(mf.get("hi"))
            if lo_current is None:
                lo_current = 0.0
            if mid_current is None:
                mid_current = 0.0
            if hi_current is None:
                hi_current = 0.0
            key_hash = hashlib.md5(f"{criterion}|{dist_col}".encode()).hexdigest()[:8]
            col_lo, col_mid, col_hi = st.columns(3)
            lo_input = col_lo.number_input("lo", value=float(lo_current), step=0.1, key=f"tri_lo_{key_hash}")
            mid_input = col_mid.number_input("mid", value=float(mid_current), step=0.1, key=f"tri_mid_{key_hash}")
            hi_input = col_hi.number_input("hi", value=float(hi_current), step=0.1, key=f"tri_hi_{key_hash}")
            invert_input = st.checkbox("invert", value=bool(mf.get("invert", False)), key=f"tri_invert_{key_hash}")
            invalid_order = not (lo_input <= mid_input <= hi_input)
            if invalid_order:
                st.warning("Ensure lo <= mid <= hi for a triangular membership.")
            apply_col, reset_col = st.columns([1, 1])
            if apply_col.button("Apply membership", key=f"tri_apply_{key_hash}"):
                if invalid_order:
                    st.warning("Membership not updated: require lo <= mid <= hi.")
                else:
                    spec = _ensure_fasl_metric_entry(criterion, dist_col)
                    mf_spec = spec.setdefault("mf", {})
                    mf_spec["type"] = mf_spec.get("type", "tri") or "tri"
                    mf_spec["lo"] = float(lo_input)
                    mf_spec["mid"] = float(mid_input)
                    mf_spec["hi"] = float(hi_input)
                    mf_spec["invert"] = bool(invert_input)
                    st.session_state["fasl_cfg"] = st.session_state.get("fasl_cfg", {})
                    st.session_state["__fasl_cfg_dirty__"] = True
                    try:
                        st.toast("Triangular membership updated.")
                    except Exception:
                        st.success("Triangular membership updated.")
                    st.rerun()
            default_spec = _fasl_default_for(criterion, dist_col)
            reset_disabled = default_spec is None
            if reset_col.button("Reset membership", key=f"tri_reset_{key_hash}", disabled=reset_disabled):
                if default_spec:
                    spec = _ensure_fasl_metric_entry(criterion, dist_col)
                    default_mf = default_spec.get("mf", {})
                    mf_spec = spec.setdefault("mf", {})
                    mf_spec["type"] = default_mf.get("type", "tri") or "tri"
                    for name in ("lo", "mid", "hi"):
                        val = _safe_float_config(default_mf.get(name))
                        mf_spec[name] = val if val is not None else 0.0
                    mf_spec["invert"] = bool(default_mf.get("invert", False))
                    st.session_state["fasl_cfg"] = st.session_state.get("fasl_cfg", {})
                    st.session_state["__fasl_cfg_dirty__"] = True
                    try:
                        st.toast("Membership reset to default.")
                    except Exception:
                        st.success("Membership reset to default.")
                    st.rerun()
            if default_spec is None:
                st.caption("No default membership defined for this metric in the reference config.")

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
    criterion_code: str | None = None,
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
                criterion=criterion_code,
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
    criterion: str | None = None,
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
        tri_lo, tri_mid, tri_hi, tri_invert = _get_fasl_membership(criterion, dist_col)
        tri_all_zero = all(abs(v) < 1e-9 for v in (tri_lo, tri_mid, tri_hi))
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
                    "criterion": criterion,
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

                if not tri_all_zero:
                    _apply_tri_background(fig_box, tri_lo, tri_mid, tri_hi, tri_invert, y_min, y_max)

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
        criterion_code = CRITERION_CODES[tab_index]

        # Popover: status & metric name filters
        with st.popover("Metric filters", use_container_width=True):
            selected_statuses = metric_filter_ui(key_prefix)
            metric_labels = [m["label"] for m in metrics]

            cfg_metrics = st.session_state.get(f"sel_{criterion_code}")
            if isinstance(cfg_metrics, list):
                cfg_metric_keys = [m for m in cfg_metrics if m in ALL_METRIC_OPTIONS.get(criterion_code, [])]
            else:
                cfg_state = st.session_state.get("fasl_cfg", {})
                bucket = cfg_state.get(criterion_code, {}) if isinstance(cfg_state, dict) else {}
                if isinstance(bucket, dict):
                    cfg_metric_keys = [m for m in bucket.keys() if m in ALL_METRIC_OPTIONS.get(criterion_code, [])]
                else:
                    cfg_metric_keys = []

            target_keys = set(cfg_metric_keys)
            desired_default_labels: list[str] = []
            for item in metrics:
                dist_col = item.get("dist_col")
                label = item.get("label")
                if dist_col in target_keys and isinstance(label, str):
                    desired_default_labels.append(label)
            if not desired_default_labels:
                desired_default_labels = []

            widget_key = f"{key_prefix}_metric_names"
            sig_key = f"__cfg_sel_sig_{criterion_code}"
            config_sig = tuple(cfg_metric_keys)
            prev_sig = st.session_state.get(sig_key)
            options_set = set(metric_labels)

            if prev_sig != config_sig:
                st.session_state[sig_key] = config_sig
                st.session_state[widget_key] = [lbl for lbl in desired_default_labels if lbl in options_set]
            else:
                st.session_state.setdefault(sig_key, config_sig)
                current_values = st.session_state.get(widget_key)
                if not isinstance(current_values, list):
                    st.session_state[widget_key] = [lbl for lbl in desired_default_labels if lbl in options_set]
                else:
                    sanitized = [lbl for lbl in current_values if lbl in options_set]
                    if sanitized != current_values:
                        st.session_state[widget_key] = sanitized if sanitized else [
                            lbl for lbl in desired_default_labels if lbl in options_set
                        ]
                    else:
                        st.session_state[widget_key] = sanitized

            selected_metric_labels = st.multiselect(
                "Select metrics",
                options=metric_labels,
                default=desired_default_labels,
                key=widget_key,
                help="Choose which KPIs to display. Metrics defined in the FASL configuration are selected by default.",
            )

        # Gauges (OK / Caution / N/A)
        counts = {"OK": 0, "Caution": 0, "N/A": 0}
        for m in metrics:
            eff_cfg = get_effective_range_cfg(m.get("label"), m.get("dist_col"), m.get("range_cfg"))
            stt = status_from_value(m.get("value"), eff_cfg, m["status_tuple"][0])
            counts[stt] = counts.get(stt, 0) + 1
        _ = counts  # (left gauges out to reduce clutter, keep logic if you re-add)

        # Metric cards/grid
        render_metric_grid(
            metrics,
            selected_statuses,
            ALL_DAILY,
            selected_metric_labels,
            key_prefix=key_prefix,
            criterion_code=criterion_code,
        )

compute_and_render(0, "Criterion 1 - Depressed mood", "Insomnia or hypersomnia, nearly every day.")
compute_and_render(1, "Criterion 2 — Loss of interest / anhedonia", "Markedly diminished interest or pleasure.")
compute_and_render(2, "Criterion 3 — Appetite / weight change", "Significant weight loss/gain or appetite change.")
compute_and_render(3, "Criterion 4 — Sleep timing & duration", "Insomnia or hypersomnia proxies.")
compute_and_render(4, "Criterion 5 — Psychomotor agitation/retardation", "Observable agitation or slowing.")
compute_and_render(5, "Criterion 6 — Fatigue / low energy", "Fatigue or loss of energy, nearly every day.")
compute_and_render(6, "Criterion 7 — Worthlessness / guilt", "Feelings of worthlessness or excessive/inappropriate guilt.")
compute_and_render(7, "Criterion 8 — Difficulty concentrating / indecisiveness", "Diminished ability to think or concentrate; indecisiveness.")
compute_and_render(8, "Criterion 9 — Suicidality", "Recurrent thoughts of death or suicidal ideation.")

st.write("")
with st.container(border=True):
    st.subheader("Configuration Summary & JSON Export")
    st.caption("Compare your current FASL settings with the built-in defaults and share them as JSON.")

    cfg_state = st.session_state.setdefault("fasl_cfg", {})
    defaults = _get_default_cfg()

    total_metrics = 0
    adjusted_metrics = 0
    added_metrics = 0
    for crit in CRIT_KEYS:
        bucket = cfg_state.get(crit, {})
        if not isinstance(bucket, dict):
            continue
        total_metrics += len(bucket)
        default_bucket = defaults.get(crit, {})
        if not isinstance(default_bucket, dict):
            default_bucket = {}
        for metric, spec in bucket.items():
            if isinstance(spec, dict) and _metric_differs_from_default(crit, metric, spec):
                adjusted_metrics += 1
            if metric not in default_bucket:
                added_metrics += 1

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
        if st.button("Save current configuration as default", key="fasl_save_cfg_network"):
            try:
                FASL_CONFIG_PATH.write_text(config_json, encoding="utf-8")
                st.session_state["__fasl_cfg_source__"] = "disk"
                st.session_state["__fasl_cfg_dirty__"] = False
                st.success(f"Saved to {FASL_CONFIG_PATH}")
            except Exception as exc:
                st.error(f"Failed to write configuration: {exc}")
    with upload_col:
        up = st.file_uploader("Upload configuration (.json)", type=["json"], key="fasl_cfg_upload_network")
        if up is not None:
            try:
                uploaded_cfg_raw = json.loads(up.read().decode("utf-8"))
                uploaded_cfg = _normalize_uploaded_config(uploaded_cfg_raw)
                if not isinstance(uploaded_cfg, dict):
                    raise ValueError("JSON must be an object")
                st.success("Configuration parsed. Apply to use.")
                if st.button("Apply configuration", key="apply_cfg_network"):
                    cfg_state.clear()
                    cfg_state.update(uploaded_cfg)
                    _ensure_bom_field(cfg_state)
                    st.session_state["fasl_cfg"] = cfg_state
                    st.session_state["__fasl_cfg_source__"] = "upload"
                    st.session_state["__fasl_cfg_dirty__"] = False
                    try:
                        st.session_state["fasl_gate_M"] = int(cfg_state.get("M", 14))
                        st.session_state["fasl_gate_N"] = int(cfg_state.get("N", 10))
                        st.session_state["fasl_gate_theta"] = float(cfg_state.get("theta", 0.7))
                        st.session_state["fasl_gate_core"] = list(cfg_state.get("core_symptoms", ["C2"]))
                        for _crit in CRIT_KEYS:
                            selected_metrics = [
                                m for m in (cfg_state.get(_crit, {}) or {}).keys() if m in ALL_METRIC_OPTIONS.get(_crit, [])
                            ]
                            st.session_state[f"sel_{_crit}"] = selected_metrics
                            for _m in selected_metrics:
                                _spec = (cfg_state[_crit].get(_m, {}) or {}) if isinstance(cfg_state.get(_crit), dict) else {}
                                _mf = (_spec.get("mf", {}) or {}) if isinstance(_spec, dict) else {}
                                try:
                                    st.session_state[f"w_{_crit}_{_m}"] = float(_spec.get("w", 0.1))
                                except Exception:
                                    st.session_state[f"w_{_crit}_{_m}"] = 0.1
                                try:
                                    st.session_state[f"lo_{_crit}_{_m}"] = float(_mf.get("lo", 0.0))
                                except Exception:
                                    st.session_state[f"lo_{_crit}_{_m}"] = 0.0
                                try:
                                    st.session_state[f"mid_{_crit}_{_m}"] = float(_mf.get("mid", 0.0))
                                except Exception:
                                    st.session_state[f"mid_{_crit}_{_m}"] = 0.0
                                try:
                                    st.session_state[f"hi_{_crit}_{_m}"] = float(_mf.get("hi", 0.0))
                                except Exception:
                                    st.session_state[f"hi_{_crit}_{_m}"] = 0.0
                                st.session_state[f"inv_{_crit}_{_m}"] = bool(_mf.get("invert", False))
                                st.session_state[f"mft_{_crit}_{_m}"] = str(_mf.get("type", "tri")).lower()
                                bom_val = _normalise_bom_value(_spec.get("bom"))
                                bom_options = CRIT_BOM_OPTIONS.get(_crit, [])
                                st.session_state[f"bom_{_crit}_{_m}"] = bom_val if bom_val in bom_options else "No BOM defined"
                    except Exception:
                        pass
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to load: {e}")
