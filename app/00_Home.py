"""
Home page with a short introduction for non‑technical users (parents, clinicians, care givers) and an
illustrative feature‑to‑criterion mapping. This file intentionally avoids any heavy logic.
"""

import os
import hashlib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------- Page config -------------------------------
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

# ------------------------------- Login -------------------------------------
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



# ------------------------------- Header ------------------------------------
with st.container(border=True):
    c1, c2, c3 = st.columns([2, 0.2, 1])
    with c1:
        st.title("PCAP‑based Behavioral Indicators aligned with DSM‑5")
        st.markdown(
            """<div class='hero'>
            This app turns everyday <b>network traffic data</b> into
            <b>human‑readable indicators</b> that indicates towards DSM‑5 symptom domains.
            It is designed for <b>non‑technical users</b> such as parents, clinicians,
            medical staff, and care givers, to support informed conversations and
            <b>earlier help‑seeking</b>.
            </div>""",
            unsafe_allow_html=True,
        )
        st.write("---")
        st.caption("🛈 This is a research tool to prompt conversation and triage. It is not a diagnostic device.")
    with c3:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/HSG_Logo_DE_RGB.svg/1024px-HSG_Logo_DE_RGB.svg.png"
        )

# ------------------------------- Audience / value ---------------------------
with st.container(border=True):
    st.subheader("Target Group")
    st.write("---")
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown(
            "<div class='kpi-card'><b>Families & Care‑Givers</b><br/><span class='small'>Spot changes in routine and sleep without reading private content.</span></div>",
            unsafe_allow_html=True,
        )
    with colB:
        st.markdown(
            "<div class='kpi-card'><b>Clinicians</b><br/><span class='small'>Complement interviews with objective, longitudinal context across DSM‑5 domains.</span></div>",
            unsafe_allow_html=True,
        )
    with colC:
        st.markdown(
            "<div class='kpi-card'><b>Medical Staff</b><br/><span class='small'>Track recovery and adherence gently, using interpretable indicators.</span></div>",
            unsafe_allow_html=True,
        )
    st.write("---")

# ------------------------------- Navigation ---------------------------------
with st.container(border=True):
    st.subheader("Navigation")
    st.caption("Jump directly to the most-used views.")
    nav1, nav2, nav3 = st.columns(3)

    with nav1:
        st.markdown("**FASL + DSM Gate**")
        st.caption("Review DSM-5 aligned indicators and daily signals.")
        if st.button("Open FASL + DSM Gate", use_container_width=True):
            try:
                st.switch_page("pages/01_FASL_DSM_Gate.py")
            except Exception:
                st.page_link("pages/01_FASL_DSM_Gate.py", label="Go to page")

    with nav2:
        st.markdown("**Network Metrics**")
        st.caption("Explore derived network metrics and trends.")
        if st.button("Open Network Metrics", use_container_width=True):
            try:
                st.switch_page("pages/02_Network_Metrics.py")
            except Exception:
                st.page_link("pages/02_Network_Metrics.py", label="Go to page")

    with nav3:
        st.markdown("**PCAP Loader**")
        st.caption("Upload PCAP files and build processed partitions.")
        if st.button("Open PCAP Loader", use_container_width=True):
            try:
                st.switch_page("pages/03_PCAP_Loader.py")
            except Exception:
                st.page_link("pages/03_PCAP_Loader.py", label="Go to page")

# with st.container(border=True):
#     st.subheader("How it works (at a glance)")
#     c1, c2, c3 = st.columns(3)
#     c1.markdown("- Capture anonymized network metadata on the home router\n- Derive activity & timing features\n- Aggregate by day")
#     c2.markdown("- Map features to DSM‑5 symptom domains\n- Build time series and baselines\n- Flag ‘OK’ vs ‘Caution’")
#     c3.markdown("- Review in the dashboard\n- Discuss patterns together\n- Use trends, not single days")

# # ------------------------------- Load Excel --------------------------------
# def load_metrics_catalog(path="metrics/metrics_catalog.xlsx") -> pd.DataFrame:
#     if not os.path.exists(path):
#         st.error(f"File not found: {path}")
#         return pd.DataFrame(columns=["Criterion", "Feature", "Label"])
#     df = pd.read_excel(path).copy()

#     # normalize columns; accept 'Label' or 'Labels'
#     label_col = None
#     for c in df.columns:
#         if str(c).strip().lower() == "label":
#             label_col = c
#             break
#         if str(c).strip().lower() == "labels":
#             label_col = c
#             break

#     df["Criterion"] = df["Criterion"].astype(str).str.strip()
#     df["Feature"] = df["Feature"].astype(str).str.strip()
#     if label_col:
#         df["Label"] = df[label_col].astype(str).str.strip()
#     else:
#         df["Label"] = df["Feature"]

#     df = df.dropna(subset=["Criterion", "Feature"]).drop_duplicates(
#         subset=["Criterion", "Feature", "Label"]
#     )
#     return df[["Criterion", "Feature", "Label"]]

# labels_df = load_metrics_catalog()

# # ------------------------------- DSM‑5 names (hard-coded sequence) ---------
# DSM5_ORDERED = [
#     ("C1", "Depressed mood"),
#     ("C2", "Anhedonia (loss of interest/pleasure)"),
#     ("C3", "Weight/appetite change"),
#     ("C4", "Sleep disturbance"),
#     ("C5", "Psychomotor change"),
#     ("C6", "Fatigue / loss of energy"),
#     ("C7", "Worthlessness / excessive guilt"),
#     ("C8", "Concentration / indecision"),
#     ("C9", "Death / suicidality"),
# ]
# DSM5_MAP = {code: name for code, name in DSM5_ORDERED}
# FINAL_SINK = "Depressed Mood"  # rightmost node

# def stable_bucket(key: str, modulo: int) -> int:
#     h = hashlib.md5(key.encode("utf-8")).hexdigest()
#     return int(h[:8], 16) % modulo

# # ------------------------------- Sankey (Plotly defaults) ------------------
# def build_mapping_sankey(df: pd.DataFrame, max_partitions: int = 8) -> go.Figure:
#     # Stages
#     router = "Home Network Router"
#     partitions = [f"PCAP 5min-Partition {i+1}" for i in range(max_partitions)]
#     users = [f"User {i+1}" for i in range(3)]

#     # Feature nodes use human-readable Label
#     feature_nodes = list(dict.fromkeys(df["Label"].tolist()))  # unique, keep order

#     # DSM nodes: always show all 9 (between Labels and final sink)
#     dsm_nodes = [name for _, name in DSM5_ORDERED]

#     # Node list (Plotly defaults for style)
#     all_nodes = [router] + partitions + users + feature_nodes + dsm_nodes + [FINAL_SINK]
#     node_index = {n: i for i, n in enumerate(all_nodes)}

#     # --- Links ---
#     src, tgt, val = [], [], []

#     # Router → Partitions
#     for p in partitions:
#         src.append(node_index[router]); tgt.append(node_index[p]); val.append(1)

#     # Partitions → Users (deterministic assignment)
#     for p in partitions:
#         u = users[stable_bucket(p, len(users))]
#         src.append(node_index[p]); tgt.append(node_index[u]); val.append(1)

#     # Users → Feature Labels (deterministic assignment)
#     feature_to_user = {feat: users[stable_bucket(feat, len(users))] for feat in feature_nodes}
#     for feat, u in feature_to_user.items():
#         src.append(node_index[u]); tgt.append(node_index[feat]); val.append(1)

#     # Feature Label → DSM‑5 criterion (from Excel rows)
#     pair_counts = {}   # (label, dsm_name) -> count
#     crit_counts = {}   # dsm_name -> total count
#     for _, row in df.iterrows():
#         feat_label = row["Label"] if row["Label"] else row["Feature"]
#         dsm_name = DSM5_MAP.get(row["Criterion"])
#         if not dsm_name:
#             continue
#         pair_counts[(feat_label, dsm_name)] = pair_counts.get((feat_label, dsm_name), 0) + 1
#         crit_counts[dsm_name] = crit_counts.get(dsm_name, 0) + 1

#     for (feat_label, dsm_name), count in pair_counts.items():
#         if feat_label in node_index and dsm_name in node_index:
#             src.append(node_index[feat_label]); tgt.append(node_index[dsm_name]); val.append(max(1, int(count)))

#     # DSM‑5 criterion → FINAL_SINK (ensure visible flow; at least 1)
#     for dsm_name in dsm_nodes:
#         count = crit_counts.get(dsm_name, 0)
#         src.append(node_index[dsm_name]); tgt.append(node_index[FINAL_SINK]); val.append(max(1, int(count)))

#     # Build figure (defaults)
#     fig = go.Figure(
#         go.Sankey(
#             node=dict(label=all_nodes),
#             link=dict(source=src, target=tgt, value=val),
#         )
#     )
#     fig.update_layout(height=640, margin=dict(l=0, r=0, t=10, b=10))
#     return fig

# # ------------------------------- Render ------------------------------------
# with st.container(border=True):
#     st.subheader("Feature‑to‑Criterion Mapping")
#     st.caption("Illustrative flow from network partitions to human‑readable features, DSM‑5 criteria, and an overall mood sink.")
#     st.plotly_chart(build_mapping_sankey(labels_df), use_container_width=True)

# with st.container(border=True):
#     st.subheader("Get started")
#     st.markdown(
#         "- Open ‘Upload and Overview’ to select datasets and inspect volume/quality.\n"
#         "- Open ‘DSM‑5 Dashboard’ to review indicators by criterion and drill into details.\n"
#         "- Use the sidebar to recompute caches if you add new data."
#     )
#     with st.popover("Privacy & data protection", use_container_width=True):
#         st.markdown(
#             "- No message/page contents are stored or analyzed.\n"
#             "- Domains are normalized using public suffix lists.\n"
#             "- Indicators are computed on your machine and cached locally.\n"
#             "- You control when caches are recalculated."
#         )

# # ------------------------------- Footer ------------------------------------
# st.markdown("---")
# st.caption("This is a research prototype and not a medical device.")
