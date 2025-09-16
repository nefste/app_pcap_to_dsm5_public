"""
Acronyms helper for the app. Provides a central list of acronyms used across
pages, metrics, and criterion modules, and a small helper to show them in a
Streamlit dialog from a sidebar button.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import streamlit as st
import pandas as pd


# Curated, app-wide acronyms with short explanations.
# Keep entries concise (<= 1 line) and user-facing.
ACRONYMS: Dict[str, str] = OrderedDict(
    {
        # Clinical / app logic
        "DSM-5": "Diagnostic and Statistical Manual of Mental Disorders, 5th ed.",
        "DSM": "Short for DSM-5 used generically in the app.",
        "DSM5": "Same as DSM-5 (alternate spelling)",
        "C1": "Depressed mood (DSM-5 criterion)",
        "C2": "Anhedonia: loss of interest/pleasure",
        "C3": "Weight/appetite change",
        "C4": "Sleep disturbance (insomnia / hypersomnia)",
        "C5": "Psychomotor change (agitation / retardation)",
        "C6": "Fatigue / loss of energy",
        "C7": "Worthlessness / excessive guilt",
        "C8": "Concentration / indecision",
        "C9": "Death / suicidality",
        "FASL": "Fuzzy Additive Signals and Likelihood (combination framework)",
        "DQI": "Data Quality Index (exclude unclear days)",
        "PHQ-9": "Patient Health Questionnaire‑9 (depression screen)",
        "DASS-21": "Depression Anxiety Stress Scales‑21",
        "MF": "Membership function (returns value in [0,1])",

        # Key daily metrics (used in formulas/labels)
        "IS": "Interdaily Stability (regularity across days)",
        "IV": "Intradaily Variability (rhythm fragmentation)",
        "CV": "Coefficient of variation",
        "LNS": "Late Night Share (share of packets at night)",
        "LIH": "Longest Inactivity Hours (longest gap in a day)",
        "ANM": "Active Night Minutes (00:00–05:00)",
        "ND": "Night/Day ratio (activity or packets)",
        "IKS": "Inter‑keystroke seconds (typing proxy in chat)",
        "SERP": "Search Engine Results Page",
        "P90-P10": "90th minus 10th percentile (robust spread)",
        "SI": "Self‑injury (or suicidal ideation) keyword tag",
        "SD": "Standard deviation",

        # Network / data
        "PCAP": "Packet capture file format",
        "PCAPNG": "PCAP Next Generation file format",
        "DNS": "Domain Name System (names ↔ addresses)",
        "SLD": "Second‑level domain (e.g., example.com)",
        "SNI": "Server Name Indication (TLS hostname)",
        "IP": "Internet Protocol (address)",
        "TCP": "Transmission Control Protocol",
        "UDP": "User Datagram Protocol",
        "DNSRR": "DNS resource record",
        "HTTP": "Hypertext Transfer Protocol",
        "HTTPS": "HTTP over TLS (encrypted)",
        "TLS": "Transport Layer Security (encryption)",
        "XMPP": "Extensible Messaging and Presence Protocol (chat)",
        "SYN": "TCP synchronize flag (start connection)",
        "ACK": "TCP acknowledgement flag",
        "DHCP": "Dynamic Host Configuration Protocol (IP assignment)",
        "Bps": "Bytes per second (rate)",
        "AAAA": "IPv6 address record type in DNS",
        "MSG": "Messaging ports used by chat apps (e.g., 5222/443)",
        "AP": "Access Point (Wi‑Fi base station)",

        # Datasets / sources
        "ONU": "Optical Network Unit (edge capture dataset)",
        "BRAS": "Broadband Remote Access Server (ISP aggregation dataset)",
        "ISP": "Internet Service Provider",

        # UI / data formats
        "KPI": "Key Performance Indicator (status tile)",
        "UI": "User interface",
        "CSV": "Comma‑separated values (text table)",
        "JSON": "JavaScript Object Notation (data)",
        "PDF": "Portable Document Format",
        "URL": "Uniform Resource Locator (web address)",
        "RAM": "Random‑access memory",
        "ETL": "Extract‑Transform‑Load (data processing)",
        "MVP": "Minimum viable product",
        "WD": "Weekday (Mon..Sun)",
        "MB": "Megabytes",
        "PCA": "Principal Component Analysis (dimensionality reduction)",

        # Catalog value sources (used in metrics code)
        "AUX": "Auxiliary context value (catalog source)",
        "TODAY": "Daily base‑features value (catalog source)",
        "TR": "Today‑row value (precomputed per‑day feature)",
        "CONST": "Literal constant value",
        "FUNC": "Computed via a registered function",
        "OK": "Meets threshold (green status)",
        "N/A": "Not available / not applicable",
        "NA": "Not available / not applicable",
    }
)

def _sorted_acronyms_df(query: str | None = None) -> pd.DataFrame:
    rows = [{"Acronym": k, "Meaning": v} for k, v in ACRONYMS.items()]
    df = pd.DataFrame(rows)
    df = df.sort_values("Acronym", kind="mergesort").reset_index(drop=True)
    if query:
        q = query.strip().lower()
        if q:
            mask = df["Acronym"].str.lower().str.contains(q) | df["Meaning"].str.lower().str.contains(q)
            df = df.loc[mask].reset_index(drop=True)
    return df


# Fragment support: keep filtering interactions local without re-running whole page
try:
    _fragment = st.fragment
except Exception:
    def _fragment(fn):  # no-op fallback on older Streamlit
        return fn


@_fragment
def _render_acronyms_body():
    # Top search bar with live filtering
    query = st.text_input(
        "Search acronyms",
        key="acronyms_query",
        placeholder="Type to filter (e.g., DNS, sleep, upload)",
    )
    df = _sorted_acronyms_df(query)

    st.caption(f"{len(df)} item(s)")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Acronym": st.column_config.TextColumn("Acronym", width="small"),
            "Meaning": st.column_config.TextColumn("Meaning", width="medium"),
        },
    )


def open_acronyms_dialog():
    """Open the acronyms dialog immediately (for use from a button click)."""

    @st.dialog("Helper / Acronyms")
    def _dlg():
        st.caption("Short explanations of acronyms used in this app and metrics.")
        _render_acronyms_body()

    _dlg()


def render_acronyms_helper_in_sidebar():
    """Place a sidebar button directly under the page list to open the dialog."""
    with st.sidebar:
        if st.button("Helper / Acronyms", help="Open a quick glossary"):
            open_acronyms_dialog()
