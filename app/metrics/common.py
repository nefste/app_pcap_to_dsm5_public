# metrics/common.py


from __future__ import annotations
import os, re, ipaddress, math
from typing import Dict, Any, Tuple, List, Callable

import numpy as np
import pandas as pd
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Constants / curated sets used by feature resolvers
# ──────────────────────────────────────────────────────────────────────────────

MSG_PORTS = [5222, 5223, 443]  # XMPP / TLS (most chats multiplex on 443)

SOCIAL_SLDS = {
    "facebook.com","fbcdn.net","instagram.com","whatsapp.com","snapchat.com",
    "tiktok.com","tiktokcdn.com","twitter.com","x.com","reddit.com","discord.com",
    "discord.gg","telegram.org","signal.org","wechat.com","line.me","vk.com",
    "gmx.com","slack.com","xing.com","skype.com","teams.microsoft.com","yammer.com"
}

STREAMING_SLDS = {
    "googlevideo.com","youtube.com","ytimg.com","netflix.com","nflxvideo.net",
    "primevideo.com","amazonvideo.com","cloudfront.net","disneyplus.com",
    "spotify.com","soundcloud.com","deezer.com","hulu.com",
    # DACH public broadcasters / portals
    "srf.ch","play.srf.ch","zdf.de","ard.de","ardmediathek.de","orf.at","tvthek.orf.at","rtlplus.com"
}

PRODUCTIVITY_SLDS = {
    "notion.so","todoist.com","trello.com","asana.com",
    "microsoft.com","office.com","outlook.com","live.com",
    "google.com","calendar.google.com","docs.google.com",
    "github.com","gitlab.com","bitbucket.org","evernote.com",
    # DACH email/productivity portals
    "gmx.net","web.de","posteo.de","proton.me","tutanota.com"
}

FOOD_DELIVERY_SLDS = {
    "ubereats.com","deliveroo.com","just-eat.co.uk","justeat.com",
    "doordash.com","grubhub.com","lieferando.de","takeaway.com",
    "dominos.com","pizza.de","foodpanda.com","wolt.com",
    # DACH variants
    "justeat.ch","just-eat.ch","lieferando.at","lieferando.ch","mjam.at","pizza.ch","pizza.at"
}

DIET_SLDS = {
    "myfitnesspal.com","fitbit.com","loseit.com","yazio.com",
    "cronometer.com","noom.com","lifesum.com","weightwatchers.com","fatsecret.com",
    # DACH tracking portals
    "fddb.info","fatsecret.de"
}

SMART_SCALE_SLDS = {
    "withings.com","garmin.com","renpho.com","tanita.com","xiaomi.com","fitbit.com",
    "beurer.com","omronconnect.com"
}

# For C7/C9 (mental health / crisis / cloud uploads)
MENTAL_HEALTH_SLDS = {
    "psychologytoday.com","mind.org.uk","mentalhealth.gov","betterhelp.com",
    "verywellmind.com","healthline.com","headtohealth.gov.au",
    # DACH resources
    "depression.ch","promente.at","seelsorge.net","psy.ch","psychenet.de"
}
CRISIS_SLDS = {
    "988lifeline.org","samiritans.org","samaritans.org","be-friending.org",
    "telefonseelsorge.de","befrienders.org","crisistextline.org",
    # DACH hotlines
    "telefonseelsorge.at","143.ch","147.ch"
}
THERAPY_SLDS = {
    "psychologytoday.com","zocdoc.com","therapyroute.com","goodtherapy.org",
    # DACH therapy finders / booking
    "therapie.de","psychotherapiesuche.de","psychotherapie.at","psychotherapie.ch","doctolib.de","doctolib.at","doktor.ch"
}
CLOUD_STORAGE_SLDS = {"drive.google.com","dropbox.com","icloud.com","box.com","onedrive.live.com"}

SELF_HARM_FORUM_PATTERNS = [
    r"/r/suicidewatch", r"/r/selfharm", r"/r/depression", r"/suicidewatch", r"/selfharm"
]
SUICIDE_QUERY_PATTERNS = [
    r"painless\+?suicide", r"how\+?to\+?hang", r"end\+?my\+?life", r"overdose\+?mg",
    r"helium\+?suicide", r"jump\+?off\+?bridge", r"suicide\+?methods?"
]

TRACKER_BURST_THRESHOLD = 3  # used by DIET tracker bursts

# ──────────────────────────────────────────────────────────────────────────────
# Tiny network helpers (direction, timing, sessions, circular std)
# ──────────────────────────────────────────────────────────────────────────────

# -------------------- DNS/host enrichment (minimal, safe) --------------------
import pandas as pd
import numpy as np

def _sld(host: str | None) -> str | None:
    """
    Very light eTLD+1 extractor (naïve): 'sub.domain.tld' -> 'domain.tld'.
    If host is already short or invalid, return as-is/None.
    """
    if not host or not isinstance(host, str):
        return None
    parts = host.strip().split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host.strip()

def build_dns_map(df: pd.DataFrame) -> dict[str, str]:

    if df is None or df.empty or "IsDNS" not in df.columns:
        return {}

    mapping: dict[str, str] = {}

    # Restrict to DNS answer rows if available
    dns_rows = df[df["IsDNS"] == True].copy()

    # Prefer DNS_ANS_NAME; fall back to DNS_QNAME
    name_col = "DNS_ANS_NAME" if "DNS_ANS_NAME" in dns_rows.columns else None
    if name_col is None or dns_rows[name_col].isna().all():
        name_col = "DNS_QNAME" if "DNS_QNAME" in dns_rows.columns else None
    if name_col is None:
        return {}

    if "DNS_ANS_IPS" not in dns_rows.columns:
        return {}

    for _, r in dns_rows.iterrows():
        name = r.get(name_col)
        if not isinstance(name, str) or not name.strip():
            continue
        ips_raw = r.get("DNS_ANS_IPS")
        if ips_raw is None or (isinstance(ips_raw, float) and np.isnan(ips_raw)):
            continue

        # DNS_ANS_IPS can be "1.2.3.4, 5.6.7.8" or a list
        if isinstance(ips_raw, (list, tuple, set)):
            ips = [str(x).strip() for x in ips_raw if str(x).strip()]
        else:
            ips = [p.strip() for p in str(ips_raw).split(",") if p.strip()]

        for ip in ips:
            # Do not overwrite a previous mapping unless the previous is empty
            if ip and ip not in mapping:
                mapping[ip] = name.strip()

    return mapping

def enrich_with_hostnames(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    if "Destination IP" not in out.columns:
        # Nothing to map; still try to create empty columns to keep callers happy
        if "ResolvedHost" not in out.columns:
            out["ResolvedHost"] = pd.Series([None] * len(out), index=out.index)
        if "SLD" not in out.columns:
            out["SLD"] = pd.Series([None] * len(out), index=out.index)
        return out

    ip2host = build_dns_map(out)
    # Fill ResolvedHost via IP map where available
    out["ResolvedHost"] = out["Destination IP"].map(ip2host) if ip2host else None

    # If we still have gaps, optionally fall back to DNS_QNAME/ANS_NAME on DNS rows themselves
    if "ResolvedHost" in out.columns:
        needs_fill = out["ResolvedHost"].isna()
        if needs_fill.any():
            if "DNS_ANS_NAME" in out.columns:
                out.loc[needs_fill & out["DNS_ANS_NAME"].notna(), "ResolvedHost"] = out.loc[
                    needs_fill & out["DNS_ANS_NAME"].notna(), "DNS_ANS_NAME"
                ]
            if "DNS_QNAME" in out.columns:
                needs_fill = out["ResolvedHost"].isna()
                out.loc[needs_fill & out["DNS_QNAME"].notna(), "ResolvedHost"] = out.loc[
                    needs_fill & out["DNS_QNAME"].notna(), "DNS_QNAME"
                ]
    else:
        out["ResolvedHost"] = None

    # Compute SLD from the resolved host (if any)
    out["SLD"] = out["ResolvedHost"].apply(_sld) if "ResolvedHost" in out.columns else None
    return out



def is_private_ip(ip_str: str) -> bool:
    try:
        return ipaddress.ip_address(ip_str).is_private
    except Exception:
        return False

def is_outbound(src: str, dst: str) -> bool:
    return is_private_ip(src) and (not is_private_ip(dst))

def is_inbound(src: str, dst: str) -> bool:
    return (not is_private_ip(src)) and is_private_ip(dst)

def interarrival_seconds(df: pd.DataFrame) -> np.ndarray:
    if df.empty or "Timestamp" not in df.columns:
        return np.array([], dtype=int)
    ts = df["Timestamp"].sort_values().values
    if len(ts) < 2:
        return np.array([], dtype=int)
    return np.diff(ts).astype("timedelta64[s]").astype(int)

def sessions_from_timestamps(df: pd.DataFrame, gap_sec: int = 300) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Split sorted timestamps into sessions separated by > gap_sec seconds."""
    if df.empty or "Timestamp" not in df.columns:
        return []
    times = sorted(pd.to_datetime(df["Timestamp"]).tolist())
    if not times:
        return []
    sessions: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    start = times[0]
    for i in range(1, len(times)):
        if (times[i] - times[i-1]).total_seconds() > gap_sec:
            sessions.append((start, times[i-1]))
            start = times[i]
    sessions.append((start, times[-1]))
    return sessions

def coef_variation(x: np.ndarray | List[float]) -> float | None:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return None
    m = float(np.mean(x))
    if m <= 0:
        return None
    s = float(np.std(x, ddof=1))
    return s / m

def circular_std_minutes(timestamps: List[datetime]) -> float | None:
    """Circular SD of times-of-day (in minutes)."""
    if not timestamps or len(timestamps) < 3:
        return None
    mins = np.array([t.hour*60 + t.minute + t.second/60.0 for t in timestamps], dtype=float)
    ang = 2*np.pi*(mins/1440.0)
    C = np.cos(ang).sum(); S = np.sin(ang).sum()
    R = np.sqrt(C**2 + S**2)/len(ang)
    if R <= 0:
        return None
    std_rad = np.sqrt(-2.0*np.log(R))
    return float(std_rad * (1440.0/(2*np.pi)))

def chat_mask(df: pd.DataFrame) -> pd.Series:
    """Heuristic chat subset: social SLDs OR known messaging ports."""
    mask = pd.Series(False, index=df.index)
    if "SLD" in df.columns:
        mask |= df["SLD"].isin(SOCIAL_SLDS)
    if "Destination Port" in df.columns:
        mask |= df["Destination Port"].isin(MSG_PORTS)
    return mask

def streaming_inbound_mask(df: pd.DataFrame) -> pd.Series:
    """Inbound streaming subset (public -> private) on known streaming SLDs."""
    mask = pd.Series(False, index=df.index)
    if "SLD" in df.columns:
        mask |= df["SLD"].isin(STREAMING_SLDS)
    if {"Source IP","Destination IP"}.issubset(df.columns):
        mask &= df.apply(lambda r: is_inbound(r["Source IP"], r["Destination IP"]), axis=1)
    return mask

# ──────────────────────────────────────────────────────────────────────────────
# Single-catalog loader (sheet 'metrics') + row selection
# ──────────────────────────────────────────────────────────────────────────────

def ensure_full_day_minutes(df_time: pd.DataFrame, day, value_col: str, fill_value=0) -> pd.DataFrame:
    day = pd.to_datetime(day).normalize()
    idx = pd.date_range(day, day + pd.Timedelta(days=1), freq="1min", inclusive="left")
    return (df_time.set_index("Timestamp")
                   .reindex(idx, fill_value=fill_value)
                   .rename_axis("Timestamp")
                   .reset_index())


# You may override by env var METRICS_CATALOG_PATH or programmatic call.
_CATALOG_CACHE: Dict[str, pd.DataFrame] = {}
_CATALOG_PATH_OVERRIDE: str | None = None

def set_catalog_path(path: str) -> None:
    """Optional: set absolute/relative path to metrics_catalog.xlsx."""
    global _CATALOG_PATH_OVERRIDE, _CATALOG_CACHE
    _CATALOG_PATH_OVERRIDE = os.path.abspath(path) if path else None
    _CATALOG_CACHE.clear()

def _catalog_path() -> str:
    if _CATALOG_PATH_OVERRIDE:
        return _CATALOG_PATH_OVERRIDE
    env = os.environ.get("METRICS_CATALOG_PATH", "").strip()
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "metrics_catalog.xlsx"))

def _empty_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "FullID","Criterion","Feature","Label","DistCol","ValueRef",
        "OK_Threshold","HigherIsWorse","Fmt","FormulaLaTeX",
        "LatexNumbersTemplate","ExplanationMD","MissingImpactMD","Notes"
    ])

def _load_catalog_df() -> pd.DataFrame:
    path = _catalog_path()
    if path in _CATALOG_CACHE:
        return _CATALOG_CACHE[path]
    if not os.path.isfile(path):
        df = _empty_metrics_df()
        _CATALOG_CACHE[path] = df
        return df
    try:
        df = pd.read_excel(path, sheet_name="metrics")
    except Exception:
        df = _empty_metrics_df()
    _CATALOG_CACHE[path] = df
    return df


def load_metric_catalog(path: str | None = None) -> pd.DataFrame:
    """
    Back‑compat: return the metrics catalog as a DataFrame.
    If `path` is provided, override the catalog location for this process.
    """
    if path:
        try:
            set_catalog_path(path)
        except Exception:
            pass
    return _load_catalog_df()


def _to_bool(x) -> bool:
    if isinstance(x, bool): return x
    s = str(x).strip().lower()
    return s in {"1","true","yes","y","t"}

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def catalog_rows(criterion_tag: str) -> List[Dict[str, Any]]:
    df = _load_catalog_df()
    if df.empty:
        return []
    ct = str(criterion_tag).strip().upper()
    mask = False
    if "Criterion" in df.columns:
        mask = (df["Criterion"].astype(str).str.upper() == ct)
    if "FullID" in df.columns:
        mask = mask | df["FullID"].astype(str).str.upper().str.startswith(ct + "/")

    sub = df.loc[mask].copy()
    if sub.empty:
        return []

    # Clean & normalize a few columns used in the criterion classes
    for col in ("Label","DistCol","ValueRef","Fmt","FormulaLaTeX","LatexNumbersTemplate",
                "ExplanationMD","MissingImpactMD","Notes"):
        if col in sub.columns:
            sub[col] = sub[col].apply(
                lambda x: (str(x).strip() if isinstance(x, str) and x.strip().lower() != "nan" else None)
            )

    if "OK_Threshold" in sub:
        sub["OK_Threshold"] = sub["OK_Threshold"].apply(_to_float)
    if "HigherIsWorse" in sub:
        sub["HigherIsWorse"] = sub["HigherIsWorse"].apply(_to_bool)

    return sub.to_dict(orient="records")

# ──────────────────────────────────────────────────────────────────────────────
# Formatters used by the UI (Fmt & LatexNumbersTemplate)
# ──────────────────────────────────────────────────────────────────────────────

def fmt_from_template(tpl: str | None) -> Callable[[Any], str]:

    tpl = tpl or "{v}"
    def _fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v))):
            return "N/A"
        try:
            return tpl.format(v=v)
        except Exception:
            try:
                return tpl.format(v=float(v))
            except Exception:
                return str(v)
    return _fmt

class _SafeMap(dict):
    def __missing__(self, k):  # missing keys become empty strings
        return ""

def latex_fill_from_template(template: str | None, ctx: Dict[str, Any] | None) -> str | None:
    """Robust templating for LatexNumbersTemplate (no KeyError on missing keys)."""
    if not template:
        return None
    try:
        return template.format_map(_SafeMap(**(ctx or {})))
    except Exception:
        return None

def update_extras_with_value(extras: Dict[str, Any] | None, value: Any) -> Dict[str, Any]:
    out = dict(extras or {})
    out.setdefault("value", value)
    try:
        out.setdefault("value_int", int(round(float(value))))
    except Exception:
        pass
    return out



FUNC_REGISTRY: Dict[str, Callable[..., Tuple[float | None, Dict[str, Any]]]] = {}


_FUNC_CALL_RE = re.compile(r"^\s*FUNC\s*:\s*([A-Za-z0-9_]+)\s*(?:\((.*)\))?\s*$")

def _parse_kwargs_str(s: str) -> dict:
    import ast
    out = {}
    if not s or not s.strip():
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = ast.literal_eval(v)
        except Exception:
            out[k] = v.strip("'\"")
    return out

def resolve_value(
    value_ref: str,
    df_day: pd.DataFrame,
    today: dict | None,
    aux_ctx: dict | None,
    today_row: dict | None
) -> Tuple[float | None, Dict[str, Any]]:

    if not isinstance(value_ref, str) or ":" not in value_ref:
        return (np.nan, {})

    src, rest = value_ref.split(":", 1)
    src = src.strip().upper()

    try:
        if src == "AUX":
            return ((aux_ctx or {}).get(rest.strip(), np.nan), {})
        if src == "TODAY":
            return ((today or {}).get(rest.strip(), np.nan), {})
        if src == "TR":
            return ((today_row or {}).get(rest.strip(), np.nan), {})
        if src == "CONST":
            return (float(rest.strip()), {})
        if src == "FUNC":
            m = _FUNC_CALL_RE.match(value_ref)
            if not m:
                # bare "FUNC:Name"
                fn = FUNC_REGISTRY.get(rest.strip())
                return fn(df_day, today, aux_ctx, today_row) if fn else (np.nan, {})
            fname = m.group(1).strip()
            kwargs = _parse_kwargs_str(m.group(2) or "")
            fn = FUNC_REGISTRY.get(fname)
            return fn(df_day, today, aux_ctx, today_row, **kwargs) if fn else (np.nan, {})
    except Exception:
        return (np.nan, {})

    return (np.nan, {})

# ──────────────────────────────────────────────────────────────────────────────
# Function resolvers used in the catalog  (C2..C9)
# Each returns: (value, extras: dict)
# ──────────────────────────────────────────────────────────────────────────────

# —— C2: Anhedonia — chat activity / passivity ———————————————

def _fn_chat_session_count(df: pd.DataFrame, *_):
    d = df[chat_mask(df)]
    if d.empty:
        return (np.nan, {})
    sess = sessions_from_timestamps(d, gap_sec=300)
    return (float(len(sess)), {"chat_sessions": len(sess)})

def _fn_median_reply_latency_sec(df: pd.DataFrame, *_):
    req = ["Timestamp","Source IP","Destination IP"]
    if not all(c in df.columns for c in req):
        return (np.nan, {})
    d = df[chat_mask(df)][req].sort_values("Timestamp").copy()
    if d.empty:
        return (np.nan, {})
    d["inb"]  = d.apply(lambda r: is_inbound(r["Source IP"], r["Destination IP"]), axis=1)
    d["outb"] = d.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1)
    ts = d["Timestamp"].values; inn = d["inb"].values; out = d["outb"].values
    gaps = []
    for i in range(len(d) - 1):
        if inn[i] and out[i + 1]:
            delta = (ts[i + 1] - ts[i]).astype("timedelta64[s]").astype(int)
            if 0 < delta <= 120:
                gaps.append(delta)
    if not gaps:
        return (np.nan, {})
    med = float(np.median(gaps))
    return (med, {"reply_pairs": len(gaps), "reply_median_s": med})

def _fn_mean_upstream_rate_bps(df: pd.DataFrame, *_):
    req = ["Timestamp","Source IP","Destination IP","Length"]
    if not all(c in df.columns for c in req):
        return (np.nan, {})
    chat = df[chat_mask(df)]
    if chat.empty:
        return (np.nan, {})
    sess = sessions_from_timestamps(chat, gap_sec=300)
    rates = []
    for a,b in sess:
        win = chat[(chat["Timestamp"]>=a) & (chat["Timestamp"]<=b)]
        if win.empty:
            continue
        dur = max(1.0, (b-a).total_seconds())
        up_b = win[win.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1)]["Length"].sum()
        rates.append(float(up_b)/dur)
    return (float(np.mean(rates)) if rates else np.nan, {"sessions_n": len(sess)})

def _fn_passive_active_byte_ratio(df: pd.DataFrame, *_):
    req = ["Source IP","Destination IP","Length"]
    if not all(c in df.columns for c in req):
        return (np.nan, {})
    Bp = float(df.loc[streaming_inbound_mask(df), "Length"].sum())
    chat = df[chat_mask(df)]
    Ba = float(chat.loc[chat.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1), "Length"].sum()) if not chat.empty else 0.0
    if Ba > 0:
        return (Bp/Ba, {"Bp": int(Bp), "Ba": int(Ba)})
    return (np.inf if Bp > 0 else np.nan, {"Bp": int(Bp), "Ba": int(Ba)})

# —— C3: Appetite/Weight — deliveries / diet / scale ————————————

def _fn_food_delivery_hits_day(df: pd.DataFrame, *_):
    if "SLD" not in df.columns:
        return (np.nan, {})
    rows = df[df["SLD"].isin(FOOD_DELIVERY_SLDS)]
    if rows.empty:
        return (0.0, {"orders_total": 0})
    sess = sessions_from_timestamps(rows, gap_sec=300)
    return (float(len(sess)), {"orders_total": len(sess)})

def _fn_late_night_delivery_ratio(df: pd.DataFrame, *_):
    if "SLD" not in df.columns or "Timestamp" not in df.columns:
        return (np.nan, {})
    rows = df[df["SLD"].isin(FOOD_DELIVERY_SLDS)].copy()
    if rows.empty:
        return (np.nan, {})
    sess = sessions_from_timestamps(rows, gap_sec=300)
    starts = [a for (a, _) in sess]
    if not starts:
        return (np.nan, {})
    night = sum(1 for t in starts if pd.to_datetime(t).hour >= 22 or pd.to_datetime(t).hour < 6)
    return (night/len(starts), {"orders_night": night, "orders_total": len(starts)})

def _fn_mean_inter_order_days_day(df: pd.DataFrame, *_):
    if "SLD" not in df.columns or "Timestamp" not in df.columns:
        return (np.nan, {})
    rows = df[df["SLD"].isin(FOOD_DELIVERY_SLDS)].copy()
    if rows.empty:
        return (np.nan, {})
    sess = sessions_from_timestamps(rows, gap_sec=300)
    if len(sess) < 2:
        return (np.nan, {"orders_in_day": len(sess)})
    starts = sorted([pd.to_datetime(a) for (a,_) in sess])
    diffs_days = np.diff(np.array(starts, dtype="datetime64[s]")).astype("timedelta64[s]").astype(int)/86400.0
    return (float(np.mean(diffs_days)), {"orders_in_day": len(sess)})

def _fn_diet_site_visits_day(df: pd.DataFrame, *_):
    if "SLD" not in df.columns:
        return (np.nan, {})
    n = int(df[df["SLD"].isin(DIET_SLDS)].shape[0])
    return (float(n), {"diet_hits": n})

def _fn_calorie_tracker_burst_count(df: pd.DataFrame, *_ , window_minutes: int = 10):
    if "SLD" not in df.columns or "Timestamp" not in df.columns:
        return (np.nan, {})
    rows = df[df["SLD"].isin(DIET_SLDS)].copy()
    if rows.empty:
        return (0.0, {"burst_windows": 0, "threshold": TRACKER_BURST_THRESHOLD})
    rows["bin"] = pd.to_datetime(rows["Timestamp"]).dt.floor(f"{window_minutes}min")
    counts = rows.groupby(["SLD","bin"]).size().reset_index(name="cnt")
    bursts = int((counts["cnt"] >= TRACKER_BURST_THRESHOLD).sum())
    return (float(bursts), {"burst_windows": bursts, "threshold": TRACKER_BURST_THRESHOLD})

def _fn_smart_scale_upload_events(df: pd.DataFrame, *_):
    if "SLD" not in df.columns or "Timestamp" not in df.columns:
        return (np.nan, {})
    rows = df[df["SLD"].isin(SMART_SCALE_SLDS)].copy()
    if {"Source IP","Destination IP"}.issubset(rows.columns):
        rows = rows[rows.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1)]
    if rows.empty:
        return (0.0, {"uploads": 0})
    sess = sessions_from_timestamps(rows, gap_sec=600)
    return (float(len(sess)), {"uploads": len(sess)})

def _fn_weigh_in_time_var_min_day(df: pd.DataFrame, *_):
    if "SLD" not in df.columns or "Timestamp" not in df.columns:
        return (np.nan, {})
    rows = df[df["SLD"].isin(SMART_SCALE_SLDS)].copy()
    tlist = pd.to_datetime(rows["Timestamp"]).tolist()
    if len(tlist) < 3:
        return (np.nan, {"n_events": len(tlist)})
    val = circular_std_minutes(tlist)
    return (float(val) if val is not None else np.nan, {"n_events": len(tlist)})

# —— C4: Sleep — onset/wake/fragmentation/day-idle —————————————

def _minutes_from_2200(ts: pd.Timestamp) -> float:
    anchor = ts.normalize() + pd.Timedelta(hours=22)
    if ts < anchor:
        # times after midnight belong to the next day relative to 22:00 anchor
        return (ts + pd.Timedelta(days=1) - anchor).total_seconds()/60.0
    return (ts - anchor).total_seconds()/60.0

def _find_night_idle_gap(df: pd.DataFrame, min_gap_min: int = 30):
    """Return (onset_ts, wake_ts, gap_minutes) for first ≥min_gap within 22–04h."""
    if df.empty or "Timestamp" not in df.columns:
        return None, None, None
    d = df.sort_values("Timestamp").copy()
    night = d[d["Timestamp"].dt.hour.isin([22,23,0,1,2,3,4])]
    if night.empty or night.shape[0] < 2:
        return None, None, None
    ts = night["Timestamp"].reset_index(drop=True)
    diffs = (ts.shift(-1) - ts).dt.total_seconds()/60.0
    idx = diffs[diffs >= float(min_gap_min)].index
    if len(idx) == 0:
        return None, None, None
    i = int(idx[0])
    return ts.iloc[i], ts.iloc[i+1], float(diffs.iloc[i])

def _fn_sleep_onset_delay_from_2200_min(df: pd.DataFrame, *_):
    onset, wake, gap_m = _find_night_idle_gap(df, 30)
    if onset is None:
        return (np.nan, {})
    return (_minutes_from_2200(onset), {"onset_hhmm": f"{onset.hour:02d}:{onset.minute:02d}", "gap_minutes": int(gap_m) if gap_m else None})

def _fn_wake_after_0400_min(df: pd.DataFrame, *_):
    onset, wake, gap_m = _find_night_idle_gap(df, 30)
    if wake is None:
        return (np.nan, {})
    anchor = wake.normalize() + pd.Timedelta(hours=4)
    delta = (wake - anchor).total_seconds()/60.0
    return (float(delta) if delta > 0 else 0.0, {"wake_hhmm": f"{wake.hour:02d}:{wake.minute:02d}"})

def _fn_onset_time_var_14d_min(df: pd.DataFrame, __, aux_ctx: dict | None, ___, *_):
    """Needs ALL_DAILY['C4_F1_OnsetLocalMin'] (optional). Returns NaN if not present."""
    ALL = (aux_ctx or {}).get("ALL_DAILY")
    if ALL is None or ALL.empty or "C4_F1_OnsetLocalMin" not in ALL.columns:
        return (np.nan, {})
    hist = pd.to_numeric(ALL["C4_F1_OnsetLocalMin"], errors="coerce").dropna().tail(14).to_numpy()
    if hist.size < 3:
        return (np.nan, {})
    mins = np.asarray(hist, dtype=float)
    ang = 2*np.pi*(mins/1440.0); C = np.cos(ang).sum(); S = np.sin(ang).sum()
    R = np.sqrt(C**2 + S**2)/len(ang)
    if R <= 0:
        return (np.nan, {})
    std_min = float(np.sqrt(-2.0*np.log(R)) * (1440.0/(2*np.pi)))
    return (std_min, {})

def _fn_sleep_duration_zabs_30d(df: pd.DataFrame, today: dict | None, aux_ctx: dict | None, *_):
    L_today = (today or {}).get("LongestInactivityHours")
    ALL = (aux_ctx or {}).get("ALL_DAILY")
    if L_today is None or ALL is None or ALL.empty or "LongestInactivityHours" not in ALL.columns:
        return (np.nan, {})
    hist = pd.to_numeric(ALL["LongestInactivityHours"], errors="coerce").dropna().tail(30).to_numpy()
    if hist.size < 7:
        return (np.nan, {})
    mu = float(np.mean(hist)); sd = float(np.std(hist, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return (np.nan, {})
    return (abs((float(L_today) - mu)/sd), {})

def _fn_nocturnal_micro_session_count_0106(df: pd.DataFrame, *_):
    if df.empty or "Timestamp" not in df.columns:
        return (np.nan, {})
    night = df[df["Timestamp"].dt.hour.isin([1,2,3,4,5,6])]
    if night.empty:
        return (0.0, {})
    sess = sessions_from_timestamps(night, gap_sec=300)
    durations = [(b-a).total_seconds()/60.0 for a,b in sess]
    count = sum(1 for m in durations if m <= 5.0)
    return (float(count), {"micro_sessions": int(count)})

def _fn_mean_inter_awak_gap_min_0106(df: pd.DataFrame, *_):
    night = df[df["Timestamp"].dt.hour.isin([1,2,3,4,5,6])]
    if night.empty:
        return (np.nan, {})
    sess = sessions_from_timestamps(night, gap_sec=300)
    if len(sess) < 2:
        return (np.nan, {})
    gaps = [(sess[i+1][0] - sess[i][1]).total_seconds()/60.0 for i in range(len(sess)-1)]
    gaps = [g for g in gaps if g >= 0]
    return (float(np.mean(gaps)) if gaps else np.nan, {})

def _fn_daytime_idle_ratio_0818(df: pd.DataFrame, *_):
    if df.empty or "Timestamp" not in df.columns:
        return (np.nan, {})
    per_min = df.set_index("Timestamp").assign(cnt=1)["cnt"].resample("1Min").sum()
    day = per_min[per_min.index.hour.isin(range(8,18))]
    if day.empty:
        return (np.nan, {})
    idle = int((day == 0).sum())
    return (idle/float(len(day)), {"idle_day_minutes": idle, "total_day_minutes": int(len(day))})

def _fn_night_day_traffic_ratio_bytes(df: pd.DataFrame, *_):
    if df.empty or "Timestamp" not in df.columns:
        return (np.nan, {})
    has_len = "Length" in df.columns
    d = df.copy()
    d["bucket"] = d["Timestamp"].dt.hour.apply(lambda h: "night" if h in [0,1,2,3,4,5] else ("day" if 6 <= h < 22 else "ignore"))
    dn = d[d["bucket"]=="night"]; dd = d[d["bucket"]=="day"]
    if has_len:
        bn = float(dn["Length"].sum()); bd = float(dd["Length"].sum())
    else:
        bn = float(len(dn)); bd = float(len(dd))
    if bd <= 0 and bn <= 0: return (np.nan, {})
    if bd <= 0: return (np.inf, {"bytes_night": int(bn), "bytes_day": 0})
    return (bn/bd, {"bytes_night": int(bn), "bytes_day": int(bd)})

# —— C5: Psychomotor — DHCP, dwell, typing, session burstiness ————————

_DHCP_PORTS_V4 = {67, 68}; _DHCP_PORTS_V6 = {546, 547}

def _is_dhcp_row(row) -> bool:
    try:
        if str(row.get("Protocol","")).upper() != "UDP":
            return False
        ports = set()
        sp = row.get("Source Port", None); dp = row.get("Destination Port", None)
        if pd.notna(sp): ports.add(int(sp))
        if pd.notna(dp): ports.add(int(dp))
        return len(ports & (_DHCP_PORTS_V4 | _DHCP_PORTS_V6)) > 0
    except Exception:
        return False

def _fn_dhcp_reassoc_per_hour(df: pd.DataFrame, *_):
    if df.empty or "Timestamp" not in df.columns:
        return (np.nan, {})
    d = df.loc[df.apply(_is_dhcp_row, axis=1)].sort_values("Timestamp")
    events_total = int(d.shape[0])
    if df["Timestamp"].notna().any():
        span_h = (df["Timestamp"].max() - df["Timestamp"].min()).total_seconds()/3600.0
        hours_considered = float(min(24.0, max(1.0, span_h)))
    else:
        hours_considered = 24.0
    val = float(events_total / hours_considered) if hours_considered > 0 else np.nan
    return (val, {"events_total": events_total, "hours_considered": hours_considered})

def _fn_mean_wifi_dwell_minutes(df: pd.DataFrame, *_):
    d = df.loc[df.apply(_is_dhcp_row, axis=1)].sort_values("Timestamp")
    n = int(d.shape[0])
    if n < 2:
        return (np.nan, {})
    gaps_min = (d["Timestamp"].diff().dt.total_seconds().dropna()/60.0).to_numpy()
    return (float(np.mean(gaps_min)) if gaps_min.size else np.nan, {"n_events": n})

def _chat_outbound_timestamps(df: pd.DataFrame) -> pd.Series:
    if df.empty or "Timestamp" not in df.columns:
        return pd.Series([], dtype="datetime64[ns]")
    m = chat_mask(df)
    chat = df.loc[m]
    if chat.empty or (not {"Source IP","Destination IP"}.issubset(chat.columns)):
        return pd.Series([], dtype="datetime64[ns]")
    chat = chat.loc[chat.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1)]
    return chat["Timestamp"].sort_values()

def _fn_median_iks_sec(df: pd.DataFrame, *_):
    ts = _chat_outbound_timestamps(df)
    if ts.empty:
        return (np.nan, {})
    gaps = ts.diff().dt.total_seconds().dropna()
    gaps = gaps[(gaps > 0) & (gaps <= 3.0)]
    if gaps.empty:
        return (np.nan, {})
    return (float(gaps.median()), {"n_gaps": int(gaps.shape[0])})

def _fn_iks_std_sec(df: pd.DataFrame, *_):
    ts = _chat_outbound_timestamps(df)
    if ts.empty:
        return (np.nan, {})
    gaps = ts.diff().dt.total_seconds().dropna()
    gaps = gaps[(gaps > 0) & (gaps <= 3.0)]
    if gaps.shape[0] < 3:
        return (np.nan, {})
    return (float(gaps.std(ddof=1)), {"n_gaps": int(gaps.shape[0])})

def _session_durations_sec(df: pd.DataFrame, gap_sec: int = 300) -> np.ndarray:
    if df.empty or "Timestamp" not in df.columns:
        return np.array([], dtype=float)
    sess = sessions_from_timestamps(df.sort_values("Timestamp"), gap_sec=gap_sec)
    if not sess:
        return np.array([], dtype=float)
    return np.array([(b - a).total_seconds() for a, b in sess], dtype=float)

def _fn_sub30s_session_count(df: pd.DataFrame, *_):
    L = _session_durations_sec(df, gap_sec=300)
    if L.size == 0:
        return (0.0, {"n_sessions_short": 0, "n_sessions_total": 0})
    n_short = int((L < 30.0).sum())
    return (float(n_short), {"n_sessions_short": n_short, "n_sessions_total": int(L.size)})

def _fn_session_fano(df: pd.DataFrame, *_):
    L = _session_durations_sec(df, gap_sec=300)
    if L.size < 3:
        return (np.nan, {})
    meanL = float(np.mean(L))
    if meanL <= 0:
        return (np.nan, {})
    varL = float(np.var(L, ddof=0))
    return (varL/meanL, {"mean_len_sec": meanL, "var_len_sec": varL, "n_sessions": int(L.size)})

# —— C6: Fatigue — optional per-day resolvers (when not pulled from TODAY) ———

def _fn_upstream_bps_per_active_hour(df: pd.DataFrame, *_):
    req = {"Timestamp","Source IP","Destination IP","Length"}
    if not req.issubset(df.columns):
        return (np.nan, {})
    up_mask = df.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1)
    up_bytes = float(df.loc[up_mask, "Length"].sum())
    per_min = df.set_index("Timestamp").assign(cnt=1)["cnt"].resample("1Min").sum()
    per_h = per_min.resample("1h").sum()
    n_active_h = int((per_h > 0).sum())
    if n_active_h <= 0:
        return (np.nan, {})
    return (up_bytes/(n_active_h*3600.0), {"up_bytes": int(up_bytes), "active_hours": n_active_h})

def _fn_post_chat_count(df: pd.DataFrame, *_):
    if not {"Timestamp","Source IP","Destination IP","Length"}.issubset(df.columns):
        return (np.nan, {})
    up_mask = df.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1)
    small = (df["Length"].fillna(0) <= 300)
    msgp = df["Destination Port"].isin(MSG_PORTS) if "Destination Port" in df.columns else False
    sldp = df["SLD"].isin(SOCIAL_SLDS) if "SLD" in df.columns else False
    mask = (up_mask & small & (msgp | sldp))
    return (float(int(mask.sum())), {})

def _fn_median_inter_req_sec(df: pd.DataFrame, *_):
    if not {"Timestamp","Source IP","Destination IP","Length"}.issubset(df.columns):
        return (np.nan, {})
    up_mask = df.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1)
    small = (df["Length"].fillna(0) <= 300)
    req_ts = df.loc[up_mask & small, "Timestamp"].sort_values()
    if req_ts.shape[0] < 2:
        return (np.nan, {})
    diffs = np.diff(req_ts.values).astype("timedelta64[s]").astype(int)
    return (float(np.median(diffs)), {"n_gaps": int(len(diffs))})

def _fn_inter_req_fano(df: pd.DataFrame, *_):
    if not {"Timestamp","Source IP","Destination IP","Length"}.issubset(df.columns):
        return (np.nan, {})
    up_mask = df.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1)
    small = (df["Length"].fillna(0) <= 300)
    req_ts = df.loc[up_mask & small, "Timestamp"].sort_values()
    if req_ts.shape[0] < 3:
        return (np.nan, {})
    diffs = np.diff(req_ts.values).astype("timedelta64[s]").astype(int).astype(float)
    mu = float(np.mean(diffs))
    if mu <= 0:
        return (np.nan, {})
    return (float(np.var(diffs)/mu), {"n_gaps": int(len(diffs))})

def _fn_first_activity_min(df: pd.DataFrame, *_):
    if df.empty or "Timestamp" not in df.columns:
        return (np.nan, {})
    ts = df[df["Timestamp"].dt.hour >= 4]["Timestamp"]
    if ts.empty:
        return (np.nan, {})
    t0 = ts.min()
    return (float(t0.hour*60 + t0.minute + t0.second/60.0), {"first_hhmm": f"{t0.hour:02d}:{t0.minute:02d}"})

# Generic: delta from median of ALL_DAILY[col]
def _fn_delta_from_median(df, today, aux_ctx, today_row, *, col: str, window: int = 28, **_):
    ALL = (aux_ctx or {}).get("ALL_DAILY")
    if ALL is None or ALL.empty or "Date" not in ALL.columns or col not in ALL.columns:
        return (np.nan, {"col": col, "window": window})
    # today date
    if isinstance(today, dict) and "Date" in today:
        today_date = pd.to_datetime(today["Date"]).normalize()
    else:
        today_date = pd.to_datetime(df["Timestamp"].iloc[0]).normalize() if len(df) else pd.NaT
    if pd.isna(today_date):
        return (np.nan, {"col": col, "window": window})
    ALL = ALL.copy()
    ALL["Date"] = pd.to_datetime(ALL["Date"]).dt.normalize()
    prev = ALL[(ALL["Date"] < today_date)].tail(window)
    prev_med = float(prev[col].median()) if (not prev.empty and prev[col].notna().any()) else np.nan
    trow = ALL[ALL["Date"] == today_date]
    today_val = float(trow[col].iloc[-1]) if (not trow.empty and pd.notna(trow[col].iloc[-1])) else np.nan
    delta = (today_val - prev_med) if (np.isfinite(today_val) and np.isfinite(prev_med)) else np.nan
    return (delta, {"baseline_median": prev_med, "today_val": today_val, "window": window})

# —— C7: Worthlessness / guilt — visits, search ratio, posting share ———————

def _fn_mental_health_site_visits_day(df: pd.DataFrame, *_):
    if "SLD" not in df.columns:
        return (np.nan, {})
    n = int(df[df["SLD"].isin(MENTAL_HEALTH_SLDS)].shape[0])
    return (float(n), {"mh_hits": n})

def _extract_queries_series(df: pd.DataFrame) -> pd.Series:
    """Best-effort source of plaintext queries from DNS_QNAME / DNS_ANS_NAME / URL."""
    if "DNS_QNAME" in df.columns and df["DNS_QNAME"].notna().any():
        return df["DNS_QNAME"].dropna().astype(str)
    if "DNS_ANS_NAME" in df.columns and df["DNS_ANS_NAME"].notna().any():
        return df["DNS_ANS_NAME"].dropna().astype(str)
    if "URL" in df.columns and df["URL"].notna().any():
        return df["URL"].dropna().astype(str)
    return pd.Series([], dtype=str)

def _fn_negative_self_search_ratio(df: pd.DataFrame, *_):
    q = _extract_queries_series(df)
    if q.empty:
        return (np.nan, {})
    patt = re.compile(r"(am\+?i\+?(?:worthless|useless|a\+?burden)|i\+?hate\+?myself|why\+?am\+?i\+?useless)", re.I)
    neg = int(q.str.contains(patt).sum())
    total = int(q.shape[0])
    return (neg/total if total else np.nan, {"neg": neg, "total": total})

def _fn_social_outgoing_share_upstream(df: pd.DataFrame, *_):
    if "SLD" not in df.columns:
        return (np.nan, {})
    soc = df[df["SLD"].isin(SOCIAL_SLDS)]
    if soc.empty:
        return (np.nan, {})
    if {"Source IP","Destination IP","Length"}.issubset(soc.columns):
        up = float(soc.loc[soc.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1), "Length"].sum())
        down = float(soc.loc[soc.apply(lambda r: is_inbound(r["Source IP"], r["Destination IP"]), axis=1), "Length"].sum())
    else:
        up = float(len(soc)); down = 0.0
    den = up + down
    return (up/den if den > 0 else np.nan, {"bytes_up": int(up), "bytes_down": int(down)})

def _fn_account_delete_unsubscribe_count(df: pd.DataFrame, *_):
    # Works only if URL/Path columns exist; otherwise returns 0/NaN gracefully.
    col = None
    for c in ("URL","HTTP_Path","Path"):
        if c in df.columns:
            col = c; break
    if not col:
        return (np.nan, {})
    patt = re.compile(r"(delete[_-]?account|unsubscribe)", re.I)
    n = int(df[col].dropna().astype(str).str.contains(patt).sum())
    return (float(n), {"events": n})

def _fn_settings_privacy_dwell_sec(df: pd.DataFrame, *_):
    # Requires per-flow timing; if unavailable, fall back to 0
    if "URL" not in df.columns:
        return (np.nan, {})
    if "FlowDurationSec" in df.columns:
        mask = df["URL"].astype(str).str.contains(r"/(settings|privacy)", case=False, regex=True)
        sec = float(df.loc[mask, "FlowDurationSec"].sum())
        return (sec, {"flows": int(mask.sum())})
    return (np.nan, {})

def _fn_cloud_upload_bytes_today(df: pd.DataFrame, *_):
    if "SLD" not in df.columns:
        return (np.nan, {})
    rows = df[df["SLD"].isin(CLOUD_STORAGE_SLDS)]
    if rows.empty:
        return (0.0, {"up_bytes": 0})
    if {"Source IP","Destination IP","Length"}.issubset(rows.columns):
        up = rows.loc[rows.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1), "Length"].sum()
        return (float(up), {"up_bytes": int(up)})
    # Fall back to counts if byte lengths are missing
    return (float(len(rows)), {"events": int(len(rows))})

# —— C8: Concentration — burstiness, dwell, notifications ————————————————

def _fn_dns_burst_rate_per_hour(df: pd.DataFrame, *_):
    if "SLD" not in df.columns or "Timestamp" not in df.columns:
        return (np.nan, {})
    d = df[["Timestamp","SLD"]].dropna()
    if d.empty:
        return (np.nan, {})
    # Sliding window via 60s bins (approximation): count bins with >=3 distinct SLDs
    d["bin"] = pd.to_datetime(d["Timestamp"]).dt.floor("60s")
    by = d.groupby("bin")["SLD"].nunique()
    bursts = int((by >= 3).sum())
    # normalize per hour of observed data
    span_h = max(1.0, (d["bin"].max() - d["bin"].min()).total_seconds()/3600.0)
    return (bursts/span_h, {"bursts": bursts, "hours": span_h})

def _fn_notification_micro_sessions_count(df: pd.DataFrame, *_):
    """
    Count '<30s, 2-step' sessions likely triggered by notifications:
      inbound burst (push) followed by immediate short outbound fetch.
    We approximate by sessions <30s with at least one inbound then outbound packet.
    """
    if "Timestamp" not in df.columns or not {"Source IP","Destination IP"}.issubset(df.columns):
        return (np.nan, {})
    sess = sessions_from_timestamps(df, gap_sec=120)  # tighter gap for tiny peeks
    count = 0
    for a,b in sess:
        dur = (b-a).total_seconds()
        if dur >= 30: 
            continue
        win = df[(df["Timestamp"]>=a) & (df["Timestamp"]<=b)]
        if win.empty:
            continue
        has_in = bool(win.apply(lambda r: is_inbound(r["Source IP"], r["Destination IP"]), axis=1).any())
        has_out= bool(win.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1).any())
        if has_in and has_out:
            count += 1
    return (float(count), {"micro_sessions": count})

def _fn_repeated_query_ratio_60m(df: pd.DataFrame, *_):
    q = _extract_queries_series(df)
    if q.empty:
        return (np.nan, {})
    # Within-day proxy: exact repeats / total
    total = int(q.shape[0])
    uniq  = int(q.nunique())
    rep = max(0, total - uniq)
    return (rep/total if total else np.nan, {"repeats": rep, "total": total})

# —— C9: Suicidal ideation — crisis / method / night spikes ——————————————

def _fn_crisis_line_hits(df: pd.DataFrame, *_):
    if "SLD" not in df.columns:
        return (np.nan, {})
    n = int(df[df["SLD"].isin(CRISIS_SLDS)].shape[0])
    return (float(n), {"crisis_hits": n})

def _fn_suicide_method_query_ratio(df: pd.DataFrame, *_):
    q = _extract_queries_series(df)
    if q.empty:
        return (np.nan, {})
    patt = re.compile("|".join(SUICIDE_QUERY_PATTERNS), re.I)
    hits = int(q.str.contains(patt).sum())
    total = int(q.shape[0])
    return (hits/total if total else np.nan, {"hits": hits, "total": total})

def _fn_self_harm_forum_visits(df: pd.DataFrame, *_):
    # Needs URL (best) or falls back to SLD==reddit.com
    if "URL" in df.columns:
        patt = re.compile("|".join(SELF_HARM_FORUM_PATTERNS), re.I)
        n = int(df["URL"].dropna().astype(str).str.contains(patt).sum())
        return (float(n), {"visits": n})
    if "SLD" in df.columns:
        n = int(df[df["SLD"].isin({"reddit.com"})].shape[0])
        return (float(n), {"visits": n, "note": "SLD fallback"})
    return (np.nan, {})

def _fn_self_harm_forum_up_bytes(df: pd.DataFrame, *_):
    if "URL" not in df.columns or not {"Source IP","Destination IP","Length"}.issubset(df.columns):
        return (np.nan, {})
    mask = df["URL"].astype(str).str.contains("|".join(SELF_HARM_FORUM_PATTERNS), case=False, regex=True)
    rows = df.loc[mask]
    if rows.empty:
        return (0.0, {})
    up = rows.loc[rows.apply(lambda r: is_outbound(r["Source IP"], r["Destination IP"]), axis=1), "Length"].sum()
    return (float(up), {"up_bytes": int(up)})

def _fn_self_harm_forum_mean_session_len_sec(df: pd.DataFrame, *_):
    if "URL" not in df.columns or "Timestamp" not in df.columns:
        return (np.nan, {})
    rows = df[df["URL"].astype(str).str.contains("|".join(SELF_HARM_FORUM_PATTERNS), case=False, regex=True)]
    if rows.empty:
        return (np.nan, {})
    L = _session_durations_sec(rows, gap_sec=300)
    return (float(np.mean(L)) if L.size else np.nan, {"n_sessions": int(L.size)})

def _fn_will_insurance_downloads(df: pd.DataFrame, *_):
    if "URL" not in df.columns:
        return (np.nan, {})
    patt = re.compile(r"(estate|last-?will|testament|insurance).*\.pdf", re.I)
    n = int(df["URL"].dropna().astype(str).str.contains(patt).sum())
    return (float(n), {"downloads": n})

def _fn_cloud_backup_up_bytes_today(df: pd.DataFrame, *_):
    # Same as cloud uploads but kept separate for C9 wording if you prefer
    return _fn_cloud_upload_bytes_today(df, *_)

def _fn_account_deletion_requests(df: pd.DataFrame, *_):
    return _fn_account_delete_unsubscribe_count(df, *_)

def _fn_night_suicide_query_bursts(df: pd.DataFrame, *_):
    if "Timestamp" not in df.columns:
        return (np.nan, {})
    q = _extract_queries_series(df)
    if q.empty:
        return (np.nan, {})
    d = pd.DataFrame({"ts": pd.to_datetime(df["Timestamp"]), "q": q})
    night = d[d["ts"].dt.hour.isin([0,1,2,3,4,5])]
    if night.empty:
        return (0.0, {})
    night["bin"] = night["ts"].dt.floor("60s")
    patt = re.compile("|".join(SUICIDE_QUERY_PATTERNS), re.I)
    grp = night.groupby("bin")["q"].apply(lambda s: int(s.astype(str).str.contains(patt).sum()))
    bursts = int((grp >= 2).sum())
    return (float(bursts), {"bursts": bursts})

def _fn_night_negative_search_ratio(df: pd.DataFrame, *_):
    if "Timestamp" not in df.columns:
        return (np.nan, {})
    q = _extract_queries_series(df)
    if q.empty:
        return (np.nan, {})
    d = pd.DataFrame({"ts": pd.to_datetime(df["Timestamp"]), "q": q})
    night = d[d["ts"].dt.hour.isin([0,1,2,3,4,5])]
    if night.empty:
        return (np.nan, {})
    patt = re.compile("|".join(SUICIDE_QUERY_PATTERNS), re.I)
    neg = int(night["q"].astype(str).str.contains(patt).sum())
    total = int(night.shape[0])
    return (neg/total if total else np.nan, {"night_neg": neg, "night_total": total})

# ──────────────────────────────────────────────────────────────────────────────
# Registry wiring (names used by ValueRef like FUNC:Name or FUNC:Name(...))
# ──────────────────────────────────────────────────────────────────────────────

FUNC_REGISTRY.update({
    # C2
    "ChatSessionCount": _fn_chat_session_count,
    "MedianReplyLatencySec": _fn_median_reply_latency_sec,
    "MeanUpstreamRateBps": _fn_mean_upstream_rate_bps,
    "PassiveActiveByteRatio": _fn_passive_active_byte_ratio,

    # C3
    "FoodDeliveryHitsDay": _fn_food_delivery_hits_day,
    "LateNightDeliveryRatio": _fn_late_night_delivery_ratio,
    "MeanInterOrderDaysDay": _fn_mean_inter_order_days_day,
    "DietSiteVisitsDay": _fn_diet_site_visits_day,
    "CalorieTrackerBurstCount": _fn_calorie_tracker_burst_count,
    "SmartScaleUploadEvents": _fn_smart_scale_upload_events,
    "WeighInTimeVarMinDay": _fn_weigh_in_time_var_min_day,

    # C4
    "SleepOnsetDelayFrom2200Min": _fn_sleep_onset_delay_from_2200_min,
    "WakeAfter0400Min": _fn_wake_after_0400_min,
    "OnsetTimeVar14dMin": _fn_onset_time_var_14d_min,
    "SleepDurationZAbs30d": _fn_sleep_duration_zabs_30d,
    "NocturnalMicroSessionCount0106": _fn_nocturnal_micro_session_count_0106,
    "MeanInterAwakeningGapMin0106": _fn_mean_inter_awak_gap_min_0106,
    "DaytimeIdleRatio0818": _fn_daytime_idle_ratio_0818,
    "NightDayTrafficRatioBytes": _fn_night_day_traffic_ratio_bytes,

    # C5
    "DhcpReassocPerHour": _fn_dhcp_reassoc_per_hour,
    "MeanWifiDwellMinutes": _fn_mean_wifi_dwell_minutes,
    "MedianIKSsec": _fn_median_iks_sec,
    "IKSStdSec": _fn_iks_std_sec,
    "Sub30sSessionCount": _fn_sub30s_session_count,
    "SessionFano": _fn_session_fano,

    # C6
    "UpstreamBpsPerActiveHour": _fn_upstream_bps_per_active_hour,
    "PostChatCount": _fn_post_chat_count,
    "MedianInterReqSec": _fn_median_inter_req_sec,
    "InterReqFano": _fn_inter_req_fano,
    "FirstActivityMin": _fn_first_activity_min,
    "DELTA_FROM_MEDIAN": _fn_delta_from_median,   # generic

    # C7
    "MentalHealthSiteVisitsDay": _fn_mental_health_site_visits_day,
    "NegativeSelfSearchRatio": _fn_negative_self_search_ratio,
    "SettingsPrivacyDwellSec": _fn_settings_privacy_dwell_sec,
    "AccountDeleteUnsubscribeCount": _fn_account_delete_unsubscribe_count,
    "SocialOutgoingShareUpstream": _fn_social_outgoing_share_upstream,
    "CloudUploadBytesToday": _fn_cloud_upload_bytes_today,

    # C8
    "DNSBurstRatePerHour": _fn_dns_burst_rate_per_hour,
    "NotificationMicroSessionsCount": _fn_notification_micro_sessions_count,
    "RepeatedQueryRatio60m": _fn_repeated_query_ratio_60m,
    # (MedianIKSsec reused from C5)

    # C9
    "CrisisLineHits": _fn_crisis_line_hits,
    "SuicideMethodQueryRatio": _fn_suicide_method_query_ratio,
    "SelfHarmForumVisits": _fn_self_harm_forum_visits,
    "SelfHarmForumUpBytes": _fn_self_harm_forum_up_bytes,
    "SelfHarmForumMeanSessionLenSec": _fn_self_harm_forum_mean_session_len_sec,
    "WillInsuranceDownloads": _fn_will_insurance_downloads,
    "CloudBackupUpBytesToday": _fn_cloud_backup_up_bytes_today,
    "AccountDeletionRequestsCount": _fn_account_deletion_requests,
    "NightSuicideQueryBursts": _fn_night_suicide_query_bursts,
    "NightNegativeSearchRatio": _fn_night_negative_search_ratio,
})




