# pages/1_Upload_and_Overview.py

import os, re, hashlib
import logging as _logging

os.environ.setdefault("SCAPY_USE_LIBPCAP", "no")
_logging.getLogger("scapy.runtime").setLevel(_logging.ERROR)
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px, plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scapy.all import IP, TCP, UDP, DNS, DNSRR, PcapReader
from scapy.utils import PcapNgReader
import pyarrow as pa
import pyarrow.parquet as pq

from metrics.common import enrich_with_hostnames, ensure_full_day_minutes

# ------------------------------- Page/UI -------------------------------------
st.set_page_config(
    page_title="PCAP Analyzer for Behavioral Research",
    page_icon="https://upload.wikimedia.org/wikipedia/de/thumb/7/77/Uni_St_Gallen_Logo.svg/2048px-Uni_St_Gallen_Logo.svg.png",
    layout="wide",
)

st.logo(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/HSG_Logo_DE_RGB.svg/1024px-HSG_Logo_DE_RGB.svg.png",
    link="https://www.unisg.ch/en/",
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
    login(); st.stop()

# -------------------- CONFIG --------------------
PROCESSED_DIR = "processed_parquet"; os.makedirs(PROCESSED_DIR, exist_ok=True)
RESAMPLE_FREQ = "1Min"; PARTITION_MINUTES = 5; COMMIT_LAG_WINDOWS = 2

def sanitize_name(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return base

def floor_to_interval(ts: datetime, minutes: int = PARTITION_MINUTES) -> datetime:
    return ts.replace(second=0, microsecond=0, minute=(ts.minute // minutes) * minutes)

def iter_packets_any(pcap_path: str):
    ext = os.path.splitext(pcap_path)[1].lower()
    reader_cls = PcapNgReader if ext == ".pcapng" else PcapReader
    with reader_cls(pcap_path) as reader:
        for pkt in reader: yield pkt

def parse_packet_row(pkt):
    if not (IP in pkt): return None
    try:
        ts = datetime.fromtimestamp(float(pkt.time))
        ip = pkt[IP]
        proto = "TCP" if TCP in pkt else ("UDP" if UDP in pkt else "OTHER")
        sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else None)
        dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else None)
        tcp_flags = (pkt[TCP].flags if TCP in pkt else None)
        length = len(pkt)
        row = {"Timestamp": ts,"Date": ts.date(),"Hour": ts.hour,
               "Protocol": proto,"Source IP": ip.src,"Destination IP": ip.dst,
               "Source Port": sport,"Destination Port": dport,"TCP Flags": str(tcp_flags) if tcp_flags is not None else None,
               "Length": length}
        if pkt.haslayer(DNS):
            dns = pkt[DNS]; row["IsDNS"] = True
            if dns.qr == 0 and dns.qd is not None:
                try: row["DNS_QNAME"] = dns.qd.qname.decode(errors="ignore").rstrip(".")
                except Exception: row["DNS_QNAME"] = str(dns.qd.qname)
                row["DNS_QTYPE"] = dns.qd.qtype
            if dns.ancount and dns.an is not None:
                ans_ips, ans_names = [], []
                for i in range(int(dns.ancount)):
                    rr = dns.an[i] if isinstance(dns.an[i], DNSRR) else None
                    if isinstance(rr, DNSRR):
                        rrname = rr.rrname.decode(errors="ignore").rstrip(".") if isinstance(rr.rrname, (bytes, bytearray)) else str(rr.rrname)
                        ans_names.append(rrname)
                        try: ans_ips.append(str(rr.rdata))
                        except Exception: pass
                if ans_ips: row["DNS_ANS_IPS"] = ",".join(ans_ips)
                if ans_names: row["DNS_ANS_NAME"] = ans_names[0]
        else:
            row["IsDNS"] = False
        return row
    except Exception:
        return None

def _commit_partition(dataset_dir: str, base_name: str, key_dt: datetime, rows: list, engine="pyarrow", compression="snappy"):
    fname = f"{base_name}__{key_dt.strftime('%Y%m%d_%H%M')}.parquet"
    fpath = os.path.join(dataset_dir, fname)
    pd.DataFrame(rows).to_parquet(fpath, index=False, engine=engine, compression=compression)
    return fpath, len(rows)

# -------------------- Robust Parquet helpers --------------------
def _try_read_parquet(fp: str, columns=None):
    """Attempt to read a Parquet file robustly.

    1) Try pandas/pyarrow directly.
    2) If that fails (e.g., OSError: repetition histogram mismatch), try reading
       row group by row group via pyarrow and concat the good ones.

    Returns (df, err). If reading fully fails, df=None and err is the exception.
    """
    try:
        df = pd.read_parquet(fp, columns=columns)
        return df, None
    except Exception as e1:
        # Fallback: best-effort per row-group read
        try:
            pf = pq.ParquetFile(fp)
            tables = []
            for i in range(pf.num_row_groups):
                try:
                    t = pf.read_row_group(i, columns=columns)
                    tables.append(t)
                except Exception:
                    # Skip corrupted row group
                    continue
            if not tables:
                return None, e1
            table = pa.concat_tables(tables, promote=True)
            # Use Arrow-backed dtypes when possible; fall back if unavailable
            try:
                df = table.to_pandas(types_mapper=pd.ArrowDtype)
            except Exception:
                df = table.to_pandas()
            return df, None
        except Exception as e2:
            return None, e2

def _parquet_file_summary(fp: str) -> dict:
    """Collect lightweight metadata for a Parquet file for display in the UI.
    Never raises; returns a dict with keys suitable for rendering.
    """
    out = {
        "exists": os.path.exists(fp),
        "path": fp,
        "size_bytes": None,
        "modified": None,
        "format_version": None,
        "created_by": None,
        "num_row_groups": None,
        "num_rows": None,
        "columns": [],  # list of (name, arrow_type)
        "codecs": {},   # column -> set(codecs)
        "preview": None,
        "error": None,
    }
    try:
        if not out["exists"]:
            out["error"] = "File does not exist"
            return out
        stat = os.stat(fp)
        out["size_bytes"] = stat.st_size
        out["modified"] = datetime.fromtimestamp(stat.st_mtime)

        pf = pq.ParquetFile(fp)
        meta = pf.metadata
        out["format_version"] = getattr(meta, "format_version", None)
        out["created_by"] = getattr(meta, "created_by", None)
        out["num_row_groups"] = getattr(meta, "num_row_groups", None)
        out["num_rows"] = getattr(meta, "num_rows", None)

        schema = pf.schema_arrow
        for fld in schema:  # pyarrow.Field
            out["columns"].append((fld.name, str(fld.type)))

        # Gather compression codecs seen per column across row groups
        codecs = {}
        try:
            for rg in range(meta.num_row_groups):
                rgm = meta.row_group(rg)
                for col_idx in range(rgm.num_columns):
                    cm = rgm.column(col_idx)
                    name = cm.path_in_schema
                    codec = str(getattr(cm.compression, "name", cm.compression))
                    codecs.setdefault(name, set()).add(codec)
        except Exception:
            pass
        out["codecs"] = {k: sorted(list(v)) for k, v in codecs.items()}

        # Small preview from the first healthy row group only (non-fatal on failure)
        try:
            if pf.num_row_groups > 0:
                for i in range(pf.num_row_groups):
                    try:
                        t = pf.read_row_group(i)
                        try:
                            df = t.to_pandas(types_mapper=pd.ArrowDtype)
                        except Exception:
                            df = t.to_pandas()
                        out["preview"] = df.head(10)
                        break
                    except Exception:
                        continue
        except Exception:
            pass
    except Exception as e:
        out["error"] = str(e)
    return out

def partition_pcap_to_parquet(pcap_path: str, base_name: str, status=None, commit_lag: int = COMMIT_LAG_WINDOWS) -> str:
    dataset_dir = os.path.join(PROCESSED_DIR, base_name); os.makedirs(dataset_dir, exist_ok=True)
    buffers, max_key_seen = {}, None
    total_rows = total_parts = 0; first_ts = last_ts = None
    if status is not None: status.write(f"Reading: {os.path.basename(pcap_path)}")
    for pkt in iter_packets_any(pcap_path):
        row = parse_packet_row(pkt); 
        if row is None: continue
        ts = row["Timestamp"]; first_ts = ts if first_ts is None else first_ts; last_ts = ts
        key = floor_to_interval(ts)
        if (max_key_seen is None) or (key > max_key_seen): max_key_seen = key
        buffers.setdefault(key, []).append(row)
        if max_key_seen is not None:
            threshold = max_key_seen - timedelta(minutes=PARTITION_MINUTES * commit_lag)
            old_keys = [k for k in list(buffers.keys()) if k < threshold]
            for k in sorted(old_keys):
                fpath, n = _commit_partition(dataset_dir, base_name, k, buffers.pop(k))
                total_rows += n; total_parts += 1
                if status is not None: status.write(f"Committed {os.path.basename(fpath)} with {n} packets")
    for k in sorted(buffers.keys()):
        fpath, n = _commit_partition(dataset_dir, base_name, k, buffers.pop(k))
        total_rows += n; total_parts += 1
        if status is not None: status.write(f"Committed {os.path.basename(fpath)} with {n} packets")
    if status is not None:
        if first_ts and last_ts:
            span = last_ts - first_ts
            status.write(f"Timespan: {first_ts} → {last_ts}  (≈ {span})")
        status.write(f"Total partitions: {total_parts} | Total packets: {total_rows}")
        status.update(label=f"Done partitioning {os.path.basename(pcap_path)}", state="complete")
    return dataset_dir

def list_partition_files(base_name: str) -> list[str]:
    dataset_dir = os.path.join(PROCESSED_DIR, base_name)
    if not os.path.isdir(dataset_dir): return []
    return sorted([os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".parquet")])

def partition_file_to_start_dt(path: str):
    m = re.search(r"__(\d{8})_(\d{4})\.parquet$", os.path.basename(path))
    if not m: return None
    datestr, timestr = m.groups()
    return datetime.strptime(datestr + timestr, "%Y%m%d%H%M")

def dataset_type(name: str) -> str:
    n = name.lower()
    if "onu" in n: return "ONU"
    if "bras" in n: return "BRAS"
    return "Other"

def group_prefix(name: str) -> str:
    return re.sub(r"([_-]?\d+)$", "", name)

def group_token_from_prefix(prefix: str) -> str:
    s = os.path.basename(prefix).lower()
    s = re.sub(r"^(onu_|bras_|other_)", "", s)
    s = re.sub(r"^capture_", "", s)
    s = re.sub(r"^[_-]+", "", s)
    return s

def partition_counts_by_date(base_names: list[str]) -> dict[pd.Timestamp, int]:
    from collections import defaultdict
    counts = defaultdict(int)
    for bn in base_names:
        for fp in list_partition_files(bn):
            dt = partition_file_to_start_dt(fp)
            if dt: counts[pd.to_datetime(dt.date())] += 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))

# -------------------- UI Intro --------------------
st.title("PCAP Loader – ETL and Overview")

with st.container(border=True):
    st.markdown("Upload PCAP/PCAPNG → partition into **5‑min Parquet** → select (Type/Group/Day or Week) → **Overview plots**.")

    # -------------------- Upload & Partition --------------------
    uploaded_files = st.file_uploader("Upload PCAP/PCAPNG file(s)", type=["pcap","pcapng"], accept_multiple_files=True)
    force_repartition = st.checkbox("Force re-partition uploaded PCAPs (overwrite existing partitions)", value=False)

    selected_base_names = set()
    if uploaded_files:
        for uploaded_file in uploaded_files:
            base_name = sanitize_name(uploaded_file.name); selected_base_names.add(base_name)
            dataset_dir = os.path.join(PROCESSED_DIR, base_name)
            need_partition = force_repartition or not os.path.isdir(dataset_dir) or not list_partition_files(base_name)
            if need_partition:
                with st.status(f"Partitioning {uploaded_file.name} into 5‑minute Parquet files…", expanded=False) as status:
                    tmp_pcap = os.path.join(PROCESSED_DIR, f"__tmp__{uploaded_file.name}")
                    with open(tmp_pcap, "wb") as f: f.write(uploaded_file.read())
                    status.write("Saved temporary PCAP. Starting pass…")
                    partition_pcap_to_parquet(tmp_pcap, base_name, status=status)
                    try: os.remove(tmp_pcap); status.write("Removed temporary PCAP.")
                    except Exception: status.write("Could not remove temporary PCAP (safe to ignore).")
                    status.update(label=f"Done partitioning {uploaded_file.name}", state="complete")
            else:
                st.info(f"Found partitions for {uploaded_file.name} — using cached Parquet files.")

# -------------------- Choose options --------------------
with st.container(border=True):
    st.subheader("Choose options")
    with st.spinner("Loading existing datasets…"):
        all_datasets = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]
        all_datasets = sorted(set(all_datasets) | selected_base_names)

    selected_types = st.multiselect(
        "Filter by dataset type",
        options=["ONU","BRAS","Other"],
        default=["ONU","BRAS","Other"],
        help="ONU = Edge capture; BRAS = ISP aggregation; Other = miscellaneous"
    )
    def type_filter(name: str) -> bool: return dataset_type(name) in selected_types
    filtered_datasets = [d for d in all_datasets if type_filter(d)]

    token_to_dsets = {}
    for name in filtered_datasets:
        pref = group_prefix(name); tok = group_token_from_prefix(pref)
        token_to_dsets.setdefault(tok, set()).add(name)

    token_options = sorted(token_to_dsets.keys())
    quick_picks = ["[ALL]","[ALL ONU]","[ALL BRAS]","[ALL OTHER]"]
    group_display_options = quick_picks + token_options
    selected_group_tokens = st.multiselect("Select dataset groups (prefix match)", options=group_display_options)

    auto_selected_from_groups = set()
    if "[ALL]" in selected_group_tokens: auto_selected_from_groups |= set(filtered_datasets)
    if "[ALL ONU]" in selected_group_tokens: auto_selected_from_groups |= {d for d in filtered_datasets if dataset_type(d)=="ONU"}
    if "[ALL BRAS]" in selected_group_tokens: auto_selected_from_groups |= {d for d in filtered_datasets if dataset_type(d)=="BRAS"}
    if "[ALL OTHER]" in selected_group_tokens: auto_selected_from_groups |= {d for d in filtered_datasets if dataset_type(d)=="Other"}
    for tok in selected_group_tokens:
        if tok in quick_picks: continue
        auto_selected_from_groups |= token_to_dsets.get(tok, set())

    selected_individual = st.multiselect("Additionally select individual datasets", options=filtered_datasets,
                                        default=sorted(auto_selected_from_groups))

    selected_base_names = sorted(set(selected_individual) | set(auto_selected_from_groups))
    if not selected_base_names:
        st.info("No datasets selected. Upload files or choose existing datasets above."); st.stop()

st.warning("⚠️ not all functions are ready to run within docker- or cloud deployment yet. therefore you see a limited view here.")

# -------------------- Parquet File Info (expander) --------------------
# with st.expander("Parquet File Info (what's inside?)", expanded=False):
#     # Let user inspect any partition of any dataset (defaults to current selection)
#     ds_all = [d for d in all_datasets if os.path.isdir(os.path.join(PROCESSED_DIR, d))]
#     ds_default = selected_base_names[0] if selected_base_names else (ds_all[0] if ds_all else None)
#     chosen_ds = st.selectbox("Dataset", options=ds_all, index=(ds_all.index(ds_default) if (ds_default in ds_all) else 0) if ds_all else 0)
#     files = list_partition_files(chosen_ds) if chosen_ds else []
#     if files:
#         # Display friendly labels: timestamp + filename
#         labels = []
#         for p in files:
#             dt = partition_file_to_start_dt(p)
#             lbl = f"{os.path.basename(p)}"
#             if dt:
#                 lbl = f"{dt.strftime('%Y-%m-%d %H:%M')} — {os.path.basename(p)}"
#             labels.append(lbl)
#         idx_default = 0
#         sel_label = st.selectbox("Partition file", options=labels, index=idx_default)
#         fp = files[labels.index(sel_label)]
#         info = _parquet_file_summary(fp)

#         if not info["exists"]:
#             st.error("File not found.")
#         else:
#             # Top-level facts in a friendly way
#             size_mb = (info["size_bytes"] or 0) / (1024*1024)
#             st.markdown(
#                 f"This file contains network packet records saved every 5 minutes. "
#                 f"It was written by '{info.get('created_by') or 'unknown writer'}'."
#             )
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric("Rows (records)", value=(info.get("num_rows") or 0))
#                 st.metric("Row groups", value=(info.get("num_row_groups") or 0))
#             with col2:
#                 st.metric("File size", value=f"{size_mb:.2f} MB")
#                 st.metric("Format version", value=str(info.get("format_version") or "-"))
#             with col3:
#                 st.metric("Last modified", value=info.get("modified").strftime("%Y-%m-%d %H:%M:%S") if info.get("modified") else "-")

#             # Columns and plain-English meaning for known ones
#             KNOWN_DESCRIPTIONS = {
#                 "Timestamp": "When the packet was captured (date and time)",
#                 "Date": "Calendar date of the packet",
#                 "Hour": "Hour of the day (0-23)",
#                 "Protocol": "Network protocol used (TCP/UDP/OTHER)",
#                 "Source IP": "Where the packet came from",
#                 "Destination IP": "Where the packet went",
#                 "Source Port": "Port used by the sender",
#                 "Destination Port": "Port used by the receiver",
#                 "TCP Flags": "Technical markers for TCP control (e.g., SYN/ACK)",
#                 "Length": "Packet size in bytes",
#                 "IsDNS": "Whether this row is a DNS query/answer",
#                 "DNS_QNAME": "Domain name asked in DNS",
#                 "DNS_QTYPE": "DNS question type (e.g., A/AAAA)",
#                 "DNS_ANS_NAME": "Domain name returned by DNS",
#                 "DNS_ANS_IPS": "IP addresses returned by DNS",
#                 "ResolvedHost": "Best hostname matched from DNS for the destination",
#                 "SLD": "Simplified domain (eTLD+1), e.g., google.com",
#                 "Dataset": "Dataset name this row belongs to",
#             }

#             col_table, col_codecs = st.columns([2,1])
#             with col_table:
#                 if info["columns"]:
#                     df_cols = pd.DataFrame(info["columns"], columns=["Column", "Type"])  # type: ignore[arg-type]
#                     df_cols["Meaning (plain English)"] = df_cols["Column"].map(KNOWN_DESCRIPTIONS).fillna("")
#                     st.dataframe(df_cols, use_container_width=True, hide_index=True)
#             with col_codecs:
#                 if info["codecs"]:
#                     st.markdown("Compression (by column)")
#                     for k, v in sorted(info["codecs"].items()):
#                         st.caption(f"{k}: {', '.join(v)}")

#             if isinstance(info.get("preview"), pd.DataFrame):
#                 st.markdown("Sample rows (first 10)")
#                 st.dataframe(info["preview"], use_container_width=True)

#             if info.get("error"):
#                 st.warning(f"Some metadata could not be read: {info['error']}")
#     else:
#         st.info("No Parquet partitions found yet for this dataset.")

# -------------------- Days + Calendar heatmap --------------------
def partition_counts_by_date_for_selection(base_names: list[str]) -> dict[pd.Timestamp, int]:
    return partition_counts_by_date(base_names)

counts_by_date = partition_counts_by_date_for_selection(selected_base_names)
available_days = sorted(list(counts_by_date.keys()))
if not available_days:
    st.error("No 5‑minute partitions found for the current selection."); st.stop()

def plot_calendar_heatmap(date_counts: dict[pd.Timestamp,int], title: str, key: str):
    if not date_counts: st.info("No partitions available."); return
    dates = sorted(date_counts.keys())
    start = pd.to_datetime(dates[0]); end = pd.to_datetime(dates[-1])
    start_monday = start - pd.to_timedelta(start.weekday(), unit="D")
    end_sunday   = end + pd.to_timedelta(6 - end.weekday(), unit="D")
    all_days = pd.date_range(start_monday, end_sunday, freq="D")
    num_weeks = len(all_days)//7
    z = [[None]*num_weeks for _ in range(7)]; text_daynum = [[""]*num_weeks for _ in range(7)]; hover=[[""]*num_weeks for _ in range(7)]
    for idx, day in enumerate(all_days):
        week_idx = idx//7; weekday = day.weekday()
        cnt = int(date_counts.get(pd.to_datetime(day.date()), 0))
        z[weekday][week_idx] = cnt; text_daynum[weekday][week_idx] = str(day.day)
        hover[weekday][week_idx] = f"{day.date()} — {cnt} partitions"
    week_labels   = [(start_monday + pd.to_timedelta(7*i, unit="D")).strftime("%Y-%m-%d") for i in range(num_weeks)]
    weekday_labels= ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    fig = go.Figure(data=go.Heatmap(z=z, x=week_labels, y=weekday_labels, colorscale="Blues",
                                    hoverinfo="text", text=hover, colorbar=dict(title="5‑min partitions / day"), zmin=0))
    for i in range(7):
        for j in range(num_weeks):
            if z[i][j] is not None:
                fig.add_annotation(x=week_labels[j], y=weekday_labels[i], text=text_daynum[i][j],
                                   showarrow=False, font=dict(size=10, color="black"))
    fig.update_layout(title=title, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True, key=key)

# ---------- NEW helpers for week/range loading ----------
def start_of_week(d: pd.Timestamp) -> pd.Timestamp:
    d = pd.to_datetime(d).normalize()
    return d - pd.to_timedelta(d.weekday(), unit="D")  # Monday

def load_day_dataframe(base_name: str, day) -> pd.DataFrame:
    day = pd.to_datetime(day).normalize(); next_day = day + pd.Timedelta(days=1)
    files = list_partition_files(base_name); chosen = []
    for p in files:
        dt = partition_file_to_start_dt(p)
        if dt and (day <= pd.to_datetime(dt) < next_day): chosen.append(p)
    if not chosen:
        return pd.DataFrame(columns=["Timestamp","Date","Hour","Protocol","Source IP","Destination IP",
                                     "Source Port","Destination Port","Length","IsDNS","DNS_QNAME","DNS_ANS_NAME","DNS_ANS_IPS"])
    dfs = []
    bad_files = []
    for fp in chosen:
        dfp, err = _try_read_parquet(fp)
        if err is not None or dfp is None:
            bad_files.append((fp, str(err)))
            continue
        if "Timestamp" not in dfp.columns and {"Date","Hour"}.issubset(dfp.columns):
            dfp["Timestamp"] = pd.to_datetime(dfp["Date"].astype(str)) + pd.to_timedelta(dfp["Hour"], unit="h")
        dfp["Timestamp"] = pd.to_datetime(dfp["Timestamp"]); dfs.append(dfp)
    if bad_files:
        st.warning(
            "Skipped {n} corrupted/unsupported Parquet partition(s) for {bn} on {d}."
            .format(n=len(bad_files), bn=base_name, d=pd.to_datetime(day).strftime("%Y-%m-%d"))
        )
        for fp, msg in bad_files[:3]:
            st.caption(f"- {os.path.basename(fp)}: {msg}")
    if not dfs:
        return pd.DataFrame(columns=["Timestamp","Date","Hour","Protocol","Source IP","Destination IP",
                                     "Source Port","Destination Port","Length","IsDNS","DNS_QNAME","DNS_ANS_NAME","DNS_ANS_IPS"])
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["Timestamp"] >= day) & (df["Timestamp"] < next_day)].copy()
    df["Date"] = df["Timestamp"].dt.date; df["Hour"] = df["Timestamp"].dt.hour; df["Dataset"] = base_name
    return df

def load_range_dataframe(base_name: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Load all partitions of one dataset overlapping [start_ts, end_ts)."""
    files = list_partition_files(base_name); chosen = []
    for p in files:
        dt = partition_file_to_start_dt(p)
        if dt and (start_ts <= pd.to_datetime(dt) < end_ts):
            chosen.append(p)
    if not chosen:
        return pd.DataFrame(columns=[
            "Timestamp","Date","Hour","Protocol","Source IP","Destination IP",
            "Source Port","Destination Port","Length","IsDNS","DNS_QNAME","DNS_ANS_NAME","DNS_ANS_IPS"
        ])
    dfs = []
    bad_files = []
    for fp in chosen:
        dfp, err = _try_read_parquet(fp)
        if err is not None or dfp is None:
            bad_files.append((fp, str(err)))
            continue
        if "Timestamp" not in dfp.columns and {"Date","Hour"}.issubset(dfp.columns):
            dfp["Timestamp"] = pd.to_datetime(dfp["Date"].astype(str)) + pd.to_timedelta(dfp["Hour"], unit="h")
        dfp["Timestamp"] = pd.to_datetime(dfp["Timestamp"])
        dfs.append(dfp)
    if bad_files:
        st.warning(
            "Skipped {n} corrupted/unsupported Parquet partition(s) for {bn} in range {s} to {e}."
            .format(n=len(bad_files), bn=base_name, s=pd.to_datetime(start_ts).strftime("%Y-%m-%d"), e=pd.to_datetime(end_ts).strftime("%Y-%m-%d"))
        )
        for fp, msg in bad_files[:3]:
            st.caption(f"- {os.path.basename(fp)}: {msg}")
    if not dfs:
        return pd.DataFrame(columns=[
            "Timestamp","Date","Hour","Protocol","Source IP","Destination IP",
            "Source Port","Destination Port","Length","IsDNS","DNS_QNAME","DNS_ANS_NAME","DNS_ANS_IPS"
        ])
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["Timestamp"] >= start_ts) & (df["Timestamp"] < end_ts)].copy()
    df["Date"] = df["Timestamp"].dt.date
    df["Hour"] = df["Timestamp"].dt.hour
    df["Dataset"] = base_name
    return df

def ensure_full_minutes_range(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                              value_col: str, fill_value=0) -> pd.DataFrame:
    """Like ensure_full_day_minutes but for arbitrary [start,end)."""
    full_idx = pd.date_range(start=start_ts, end=end_ts - pd.Timedelta(minutes=1), freq=RESAMPLE_FREQ)
    out = (df.set_index("Timestamp")[[value_col]]
             .reindex(full_idx, fill_value=fill_value)
             .rename_axis("Timestamp").reset_index())
    out[value_col] = out[value_col].fillna(fill_value)
    return out

# ---------- NEW: analysis scope + multi-selects ----------
analysis_scope = st.radio(
    "Analysis scope",
    options=["Days", "Weeks"],
    index=0,
    horizontal=True,
    help="Choose whether to compare individual days or entire Mon–Sun weeks."
)

if analysis_scope == "Days":
    selected_days = st.multiselect(
        "Select day(s)",
        options=available_days,
        default=[available_days[-1]],
        format_func=lambda d: pd.to_datetime(d).strftime("%Y-%m-%d"),
        help="Select one or multiple days to overlay in the plots below."
    )
    selected_weeks = []
else:
    available_week_starts = sorted({ start_of_week(d) for d in available_days })
    if not available_week_starts:
        st.error("No complete weeks available from the current partitions."); st.stop()
    selected_weeks = st.multiselect(
        "Select week(s) (Mon–Sun)",
        options=available_week_starts,
        default=[available_week_starts[-1]],
        format_func=lambda d: f"Week of {pd.to_datetime(d).strftime('%Y-%m-%d')}",
        help="Weeks are Monday–Sunday. Select one or multiple weeks to overlay."
    )
    selected_days = []

# A small signature for component keys (keeps widgets stable across selections)
sel_str = (
    "days:" + ",".join(pd.to_datetime(selected_days).strftime("%Y-%m-%d").tolist())
    if analysis_scope == "Days"
    else "weeks:" + ",".join(pd.to_datetime(selected_weeks).strftime("%Y-%m-%d").tolist())
)
key_prefix = f"{analysis_scope}_{hashlib.md5((sel_str + '|' + '|'.join(sorted(selected_base_names))).encode()).hexdigest()[:8]}"

# Calendar heatmap still helpful to see coverage
plot_calendar_heatmap(counts_by_date, "🗓 Data availability — 5‑min Parquet files per day", key=f"{key_prefix}_calendar")

# -------------------- Load selection & enrich --------------------
if analysis_scope == "Days" and not selected_days:
    st.warning("Select at least one day."); st.stop()
if analysis_scope == "Weeks" and not selected_weeks:
    st.warning("Select at least one week."); st.stop()

with st.status("Loading and preparing data for the current selection…", expanded=False) as load_status:
    parts = []
    if analysis_scope == "Days":
        for bn in selected_base_names:
            load_status.write(f"Loading partitions for: {bn}")
            for day in selected_days:
                df_b = load_day_dataframe(bn, day)
                if not df_b.empty:
                    df_b["Label"] = pd.to_datetime(day).strftime("%Y-%m-%d")  # which day this row belongs to
                    df_b["Scope"] = "Day"
                    parts.append(df_b)
    else:  # Weeks
        for bn in selected_base_names:
            load_status.write(f"Loading partitions for: {bn}")
            for wk_start in selected_weeks:
                wk_start = pd.to_datetime(wk_start).normalize()
                wk_end = wk_start + pd.Timedelta(days=7)
                df_b = load_range_dataframe(bn, wk_start, wk_end)
                if not df_b.empty:
                    df_b["WeekStart"] = wk_start
                    df_b["Label"] = f"Week of {wk_start.strftime('%Y-%m-%d')}"
                    df_b["Scope"] = "Week"
                    parts.append(df_b)

    if not parts:
        load_status.update(label="No traffic for the current selection.", state="error"); st.stop()

    df_all = pd.concat(parts, ignore_index=True)
    df_all["Timestamp"] = pd.to_datetime(df_all["Timestamp"])

    load_status.write("Enriching with DNS hostnames / SLD…")
    df_all = enrich_with_hostnames(df_all)

    token_for_dataset = {bn: group_token_from_prefix(group_prefix(bn)) for bn in selected_base_names}
    if "Dataset" not in df_all.columns:
        df_all["Dataset"] = df_all.get("Dataset", np.nan)
    df_all["Group"] = df_all["Dataset"].map(token_for_dataset).fillna("other")

    df_group = (df_all.groupby("Group")
                .agg(Packets=("Timestamp", "count"), Bytes=("Length", "sum"))
                .reset_index()
                .sort_values("Packets", ascending=False))

    load_status.update(label="Data loaded & enriched.", state="complete")

# ===================== OVERVIEW SECTION =====================
if analysis_scope == "Days":
    desc = ", ".join(sorted({pd.to_datetime(d).strftime("%Y-%m-%d") for d in selected_days}))
else:
    desc = ", ".join([f"{pd.to_datetime(w).strftime('%Y-%m-%d')}" for w in selected_weeks])
st.header(f"Traffic Overview — {analysis_scope}: {desc}")

ov_tabs = st.tabs(["Packets", "Bytes", "Protocol mix", "Hourly heatmap", "Groups"])

# ---------- Packets ----------
with ov_tabs[0]:
    if analysis_scope == "Days":
        # Per-minute packets, aligned to time-of-day, one line per selected day
        day_frames = []
        for label in sorted(df_all["Label"].unique()):
            dfi = df_all[df_all["Label"] == label]
            dfts = (dfi.set_index("Timestamp").assign(Packets=1)["Packets"]
                    .resample(RESAMPLE_FREQ).sum().reset_index())
            day_start = pd.to_datetime(label)
            dff = ensure_full_day_minutes(dfts, day_start, "Packets", fill_value=0)
            dff["MinuteOfDay"] = dff["Timestamp"].dt.hour * 60 + dff["Timestamp"].dt.minute
            dff["Clock"] = dff["Timestamp"].dt.strftime("%H:%M")
            dff["Label"] = label
            day_frames.append(dff[["MinuteOfDay","Clock","Packets","Label"]])
        df_packets_days = pd.concat(day_frames, ignore_index=True) if day_frames else pd.DataFrame()

        fig = go.Figure()
        for label, dfg in df_packets_days.groupby("Label"):
            fig.add_trace(go.Scatter(
                x=dfg["MinuteOfDay"], y=dfg["Packets"], mode="lines", name=label,
                hovertext=dfg["Clock"], hovertemplate="%{hovertext}<br>Packets=%{y}<extra>%{fullData.name}</extra>"
            ))
        fig.update_layout(
            title=f"Packets per {RESAMPLE_FREQ} (time‑of‑day, overlaid by day)",
            xaxis_title="Time of day", yaxis_title="Packets", legend_title="Day"
        )
        fig.update_xaxes(tickmode="array",
                         tickvals=list(range(0, 24*60, 60)),
                         ticktext=[f"{h:02d}:00" for h in range(24)])
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_packets_days")

    else:  # Weeks
        week_frames = []
        for wk_start in sorted(df_all.get("WeekStart", []).unique()):
            if pd.isna(wk_start): continue
            wk_start = pd.to_datetime(wk_start)
            dfi = df_all[df_all["WeekStart"] == wk_start]
            dfts = (dfi.set_index("Timestamp").assign(Packets=1)["Packets"]
                    .resample(RESAMPLE_FREQ).sum().reset_index())
            dff = ensure_full_minutes_range(dfts, wk_start, wk_start + pd.Timedelta(days=7), "Packets", fill_value=0)
            dff["MinuteOfWeek"] = ((dff["Timestamp"] - wk_start).dt.total_seconds() // 60).astype(int)
            dff["Hover"] = dff["Timestamp"].dt.strftime("%a %H:%M")
            dff["Label"] = f"Week of {wk_start.strftime('%Y-%m-%d')}"
            week_frames.append(dff[["MinuteOfWeek","Packets","Hover","Label"]])
        df_packets_weeks = pd.concat(week_frames, ignore_index=True) if week_frames else pd.DataFrame()

        fig = go.Figure()
        for label, dfg in df_packets_weeks.groupby("Label"):
            fig.add_trace(go.Scatter(
                x=dfg["MinuteOfWeek"], y=dfg["Packets"], mode="lines", name=label,
                hovertext=dfg["Hover"], hovertemplate="%{hovertext}<br>Packets=%{y}<extra>%{fullData.name}</extra>"
            ))
        # vertical guides at day boundaries
        for i in range(1, 7): fig.add_vline(x=i*1440, line_width=0.5, line_dash="dot", line_color="gray")
        fig.update_layout(
            title=f"Packets per {RESAMPLE_FREQ} (Mon–Sun, overlaid by week)",
            xaxis_title="Day of week & time", yaxis_title="Packets", legend_title="Week"
        )
        fig.update_xaxes(tickmode="array",
                         tickvals=[i*1440 for i in range(0,7)],
                         ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_packets_weeks")

# ---------- Bytes ----------
with ov_tabs[1]:
    if "Length" in df_all.columns and not df_all.empty:
        if analysis_scope == "Days":
            day_frames = []
            for label in sorted(df_all["Label"].unique()):
                dfi = df_all[df_all["Label"] == label]
                dfts = (dfi.set_index("Timestamp")["Length"]
                        .resample(RESAMPLE_FREQ).sum().reset_index(name="Bytes"))
                day_start = pd.to_datetime(label)
                dff = ensure_full_day_minutes(dfts, day_start, "Bytes", fill_value=0)
                dff["MinuteOfDay"] = dff["Timestamp"].dt.hour * 60 + dff["Timestamp"].dt.minute
                dff["Clock"] = dff["Timestamp"].dt.strftime("%H:%M")
                dff["Label"] = label
                day_frames.append(dff[["MinuteOfDay","Clock","Bytes","Label"]])
            df_bytes_days = pd.concat(day_frames, ignore_index=True) if day_frames else pd.DataFrame()

            fig = go.Figure()
            for label, dfg in df_bytes_days.groupby("Label"):
                fig.add_trace(go.Scatter(
                    x=dfg["MinuteOfDay"], y=dfg["Bytes"], mode="lines", name=label,
                    hovertext=dfg["Clock"], hovertemplate="%{hovertext}<br>Bytes=%{y}<extra>%{fullData.name}</extra>"
                ))
            fig.update_layout(
                title=f"Bytes per {RESAMPLE_FREQ} (time‑of‑day, overlaid by day)",
                xaxis_title="Time of day", yaxis_title="Bytes", legend_title="Day"
            )
            fig.update_xaxes(tickmode="array",
                             tickvals=list(range(0, 24*60, 60)),
                             ticktext=[f"{h:02d}:00" for h in range(24)])
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_bytes_days")

        else:  # Weeks
            week_frames = []
            for wk_start in sorted(df_all.get("WeekStart", []).unique()):
                if pd.isna(wk_start): continue
                wk_start = pd.to_datetime(wk_start)
                dfi = df_all[df_all["WeekStart"] == wk_start]
                dfts = (dfi.set_index("Timestamp")["Length"]
                        .resample(RESAMPLE_FREQ).sum().reset_index(name="Bytes"))
                dff = ensure_full_minutes_range(dfts, wk_start, wk_start + pd.Timedelta(days=7), "Bytes", fill_value=0)
                dff["MinuteOfWeek"] = ((dff["Timestamp"] - wk_start).dt.total_seconds() // 60).astype(int)
                dff["Hover"] = dff["Timestamp"].dt.strftime("%a %H:%M")
                dff["Label"] = f"Week of {wk_start.strftime('%Y-%m-%d')}"
                week_frames.append(dff[["MinuteOfWeek","Bytes","Hover","Label"]])
            df_bytes_weeks = pd.concat(week_frames, ignore_index=True) if week_frames else pd.DataFrame()

            fig = go.Figure()
            for label, dfg in df_bytes_weeks.groupby("Label"):
                fig.add_trace(go.Scatter(
                    x=dfg["MinuteOfWeek"], y=dfg["Bytes"], mode="lines", name=label,
                    hovertext=dfg["Hover"], hovertemplate="%{hovertext}<br>Bytes=%{y}<extra>%{fullData.name}</extra>"
                ))
            for i in range(1, 7): fig.add_vline(x=i*1440, line_width=0.5, line_dash="dot", line_color="gray")
            fig.update_layout(
                title=f"Bytes per {RESAMPLE_FREQ} (Mon–Sun, overlaid by week)",
                xaxis_title="Day of week & time", yaxis_title="Bytes", legend_title="Week"
            )
            fig.update_xaxes(tickmode="array",
                             tickvals=[i*1440 for i in range(0,7)],
                             ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_bytes_weeks")
    else:
        st.info("No byte lengths available to plot.")

# ---------- Protocol mix ----------
with ov_tabs[2]:
    if "Protocol" in df_all.columns and not df_all.empty:
        # Show protocol mix on the absolute timeline over the selection
        proto_time = (df_all.assign(cnt=1)
                      .groupby([pd.Grouper(key="Timestamp", freq=RESAMPLE_FREQ), "Protocol"])["cnt"]
                      .sum().reset_index())
        fig_proto_stack = px.bar(proto_time, x="Timestamp", y="cnt", color="Protocol",
                                 title=f"Protocol mix per {RESAMPLE_FREQ} (absolute time over selection)",
                                 barmode="stack")
        fig_proto_stack.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_proto_stack, use_container_width=True, key=f"{key_prefix}_proto_stack")
    else:
        st.info("No protocol data available.")

# ---------- Hourly heatmap ----------
with ov_tabs[3]:
    if analysis_scope == "Days":
        # Heatmap: rows = selected days, cols = hours 0..23
        heat = (df_all.assign(Day=df_all["Label"], Hour=df_all["Timestamp"].dt.hour)
                .groupby(["Day","Hour"]).size().unstack(fill_value=0))
        heat = heat.reindex(index=sorted(heat.index), columns=range(24), fill_value=0)
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heat.values, x=[f"{h:02d}:00" for h in range(24)],
            y=list(heat.index), colorscale="Blues", colorbar=dict(title="Packets / hour")
        ))
        fig_heatmap.update_layout(title="Hourly Activity Heatmap (rows = selected days)")
        st.plotly_chart(fig_heatmap, use_container_width=True, key=f"{key_prefix}_hour_heatmap_days")
    else:
        # Heatmap aggregated across selected weeks: rows = Mon..Sun, cols = hours
        heat = (df_all.assign(WD=df_all["Timestamp"].dt.day_name().str[:3],
                              Hour=df_all["Timestamp"].dt.hour)
                .groupby(["WD","Hour"]).size().unstack(fill_value=0))
        heat = heat.reindex(index=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], columns=range(24), fill_value=0)
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heat.values, x=[f"{h:02d}:00" for h in range(24)],
            y=list(heat.index), colorscale="Blues", colorbar=dict(title="Packets / hour")
        ))
        fig_heatmap.update_layout(title="Hourly Activity Heatmap (aggregated across selected weeks)")
        st.plotly_chart(fig_heatmap, use_container_width=True, key=f"{key_prefix}_hour_heatmap_weeks")

# ---------- Groups ----------
with ov_tabs[4]:
    if "Group" in df_all.columns and not df_group.empty:
        fig_groups = make_subplots(specs=[[{"secondary_y": True}]])
        fig_groups.add_trace(go.Bar(x=df_group["Group"], y=df_group["Packets"], name="Packets"), secondary_y=False)
        fig_groups.add_trace(go.Scatter(x=df_group["Group"], y=df_group["Bytes"], name="Bytes", mode="lines+markers"), secondary_y=True)
        fig_groups.update_layout(title="Traffic by Group — Packets (bars, left) & Bytes (line, right)",
                                 xaxis_title="Group", legend_title="Metric")
        fig_groups.update_yaxes(title_text="Packets", secondary_y=False)
        fig_groups.update_yaxes(title_text="Bytes", secondary_y=True)
        st.plotly_chart(fig_groups, use_container_width=True, key=f"{key_prefix}_group_mix")
    else:
        st.info("No group information available.")
