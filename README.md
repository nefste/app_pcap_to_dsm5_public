 # PCAP Analyzer for Behavioral Research (DSM‑5 mapping)
 This app is a MVP research tool and not used for medical device. Use responsibly.

 This app transforms anonymized network metadata (no content) into interpretable, human‑readable indicators aligned with DSM‑5 symptom domains. It supports families and clinicians with longitudinal context and is intended strictly for research and early‑warning discussions — not diagnosis.
 
 ## Run With Docker (recommended)
 - Prerequisite: Docker Desktop installed
 - From this folder (`app/`), build and start:
 
 ```bash
 docker compose up --build
 ```
 
 - Open the app at: http://localhost:8501

 
 ## Local Development (without Docker)
 ```bash
 python -m venv venv
 venv\Scripts\activate     
 pip install -r ../requirements.txt
 
 streamlit run app/00_Home.py
 ```
 
 ## Project Structure
 - `app/00_Home.py`: Streamlit entry page, intro and mapping overview
 - `app/pages/01_Upload_and_Overview.py`: Upload PCAP/PCAPNG, partition into 5‑min Parquet, traffic overview
 - `app/pages/02_DSM5_Dashboard.py`: Dashboard computing per‑criterion KPIs (OK/Caution/N/A) for a selected day
 - `app/pages/03_FASL_DSM_Gate.py`: Fuzzy‑Additive Symptom Likelihood + DSM‑Gate (transparent, rule‑based prototype)
 - `app/metrics/`: Feature engineering and per‑criterion logic
   - `base_features.py`: Day‑level base features (sessions, timing, directions, volumes)
   - `common.py`: Enrichment (DNS/hostnames), curated SLD lists, helpers
   - `criterion1.py` … `criterion9.py`: Metrics mapped to DSM‑5 criteria C1–C9
   - `metrics_catalog.xlsx`: Human‑readable mapping catalog used on the home page
 - `app/processed_parquet/`: Output store for 5‑minute Parquet partitions (created on demand)
 - `app/feature_cache/`: Daily metric cache for faster dashboard loads (created on demand)
 - `app/.streamlit/`: Streamlit config and secrets (credentials)
 
 ## Architecture Overview
 - Ingestion & partitioning:
   - PCAP/PCAPNG files are read with Scapy and partitioned into 5‑minute Parquet files: `app/pages/01_Upload_and_Overview.py`.
   - Files are named with a start timestamp suffix (`__YYYYMMDD_HHMM.parquet`) to enable robust filtering by day/week.
 - Enrichment & base features:
   - DNS/hostname enrichment and curated SLDs (e.g., social, streaming) live in `metrics/common.py`.
   - Day‑level base features are computed in `metrics/base_features.py` (sessions, timing patterns, directionality, volumes, etc.).
 - Metrics per DSM‑5 criterion:
   - Each criterion (C1–C9) has a dedicated module `metrics/criterionX.py`, computing interpretable signals from base features and enriched context.
   - Output statuses are normalized to `OK`, `Caution`, or `N/A` and rendered as KPI tiles in the dashboard.
 - Caching strategy:
   - The dashboard keeps a per‑day cache (`app/feature_cache/`) to avoid recomputation when the underlying 5‑min partitions are unchanged.
   - Parquet readers are implemented defensively to bypass corrupt row groups where possible.
 - FASL + DSM‑Gate (research prototype):
   - `pages/03_FASL_DSM_Gate.py` exposes a transparent, rule‑based, interactive model that aggregates existing metrics into fuzzy likelihoods and a DSM‑Gate over a rolling window.
 
 ## Key Features
 - Upload and partition PCAP/PCAPNG into 5‑minute Parquet chunks
 - Robust Parquet read/write with fallbacks for partial corruption
 - Dataset grouping and day/week selection for analysis
 - Per‑criterion KPIs (OK/Caution/N/A) with plots and details
 - Transparent FASL + DSM‑Gate prototype leveraging existing metrics
 - Simple login gate via Streamlit secrets (username/password)
 
 ## Configuration & Credentials
 - Streamlit config: `app/.streamlit/config.toml`
 - Secrets (credentials): `app/.streamlit/secrets.toml`
   - Update the values before sharing or deploying beyond a trusted environment.
   - With Docker Compose, the entire `.streamlit` directory is mounted read‑only into the container.
 
 ## Notes
 - Scapy is configured to avoid libpcap by default (`SCAPY_USE_LIBPCAP=no`) since the app only reads from files, not from live interfaces.
 - Large PCAPs can be memory‑intensive; consider chunking uploads or running with sufficient RAM.
 - This software is a research tool and not a medical device. Use responsibly.
