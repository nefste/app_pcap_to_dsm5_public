# Router-Centric Behavioral Signals Prototype

This repository houses the end-to-end Streamlit application developed as part of the Master thesis **Digital and Environmental Signals, Passive Sensing for Early Depression Detection** by Stephan Nef at the University of St. Gallen.

**Important:** This software is a research prototype. It must not be used for diagnosis, medical decision-making, or as a substitute for professional clinical assessment.

A hosted build is available at https://unisg-nef.streamlit.app/.

## Overview

Depression often evolves gradually, and the behavioural shifts that precede a diagnosable episode are easy to miss. The project explores whether household router metadata, processed locally with full transparency, can surface interpretable indicators aligned with DSM-5 criteria. The application converts packet capture (PCAP/PCAPNG) files into daily Behaviour Observation Metric (BOM) scores using a Fuzzy Additive Symptom Likelihood (FASL) aggregation and an auditable DSM-style gate.

The primary goals are to:
- keep processing on a trusted local device;
- expose every weight, membership function, and threshold for inspection;
- communicate outputs in plain language that non-technical stakeholders can understand.

## What the application does

1. Upload one or more PCAP/PCAPNG traces captured at a router or gateway.
2. Partition each trace into five-minute Parquet windows and enrich records with Server Name Indication (SNI) and Public Suffix List lookups.
3. Engineer feature sets spanning activity timing, flow directionality, night/day ratios, domain diversity, and modality balances.
4. Map the features to per-criterion likelihoods for DSM-5 symptom domains (C1-C9) through interpretable metric modules.
5. Aggregate the likelihoods with a configurable FASL gate to produce day-level BOM statuses (`OK`, `Caution`, `N/A`) and surface them through an interactive dashboard.

## Visual walkthrough

![Overview of the application pipeline showing collection, enrichment, metrics, and assessment steps.](app/utils/overview_app_visualisations.png)

![High-level information flow from data ingestion through feature extraction to DSM-5 aligned indicators.](app/utils/information_flow.png)

![Sankey diagram linking household devices to PCAP windows, engineered metrics, BOM aggregation, DSM-5 criteria, and the user feedback loop.](app/utils/router_dsm5_sankey.png)

## Key capabilities

- Transparent DSM-5 alignment with per-criterion explanations and trend plots.
- Behaviour Observation Metric (BOM) dashboard for daily review and deeper drill-downs.
- Fuzzy Additive Symptom Likelihood gate with explicit parameters stored in `app/fasl_config.json`.
- Caching of intermediate computations for repeat analysis without reprocessing PCAP files.
- Streamlit-based authentication via `app/.streamlit/secrets.toml` for lightweight access control.

## Repository layout

- `app/`: Streamlit application, feature engineering modules, configuration, and cached artifacts.
- `app/pages/`: Multi-page dashboard including upload, DSM-5 overview, and FASL gate exploration.
- `app/metrics/`: Domain-specific feature logic for each DSM-5 criterion plus shared helpers.
- `scripts/`: Utility scripts used during experimentation and deployment.
- `requirements.txt` and `app/requirements.txt`: Python dependencies for root-level tooling and the Streamlit app respectively.

## Getting started

### Option 1 - run with Docker (recommended)

1. Ensure Docker Desktop is installed and running.
2. From the `app/` directory, build and start the container:

   ```bash
   docker compose up --build
   ```

3. Open http://localhost:8501 in a browser and sign in with the credentials defined in `app/.streamlit/secrets.toml`.

### Option 2 - local Python environment

1. Create and activate a virtual environment at the repository root:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Install the app-specific dependencies and launch Streamlit:

   ```bash
   pip install -r app/requirements.txt
   streamlit run app/00_Home.py
   ```

3. Set `SCAPY_USE_LIBPCAP=no` if libpcap is unavailable; the application only reads from files.

## Working with PCAP data

- Input traces should contain packet headers only; payload inspection is neither required nor performed.
- Processed five-minute Parquet windows are stored in `app/processed_parquet/`.
- Day-level feature caches are written to `app/feature_cache/` to accelerate repeat sessions.
- The application provides safeguards around corrupt row groups and allows uploads to be chunked when files are large.

## Transparency and configuration

- Per-criterion metric logic resides in `app/metrics/criterion*.py`, with shared enrichment helpers in `app/metrics/common.py`.
- `app/fasl_config.json` enumerates all FASL weights, membership functions, thresholds, and gate windows; adjust these to experiment with different behaviours.
- Streamlit look-and-feel and authentication are controlled by `app/.streamlit/config.toml` and `app/.streamlit/secrets.toml`.

## In-app entry points

![Early Signs Overview page with Behaviour Observation Metric tiles acting as the user's dashboard entry point.](app/utils/dsm_5_indicator_overview_as_users_entry_point.png)

![Example metric tile and detail view for DSM-5 Criterion 8 Feature 8 (C8_F8) illustrating how network signals map to interpretable indicators.](app/utils/C8_F8.png)

## Research context

The prototype was evaluated on two complementary datasets:

- **Dataset A (Jiang et al., UESTC):** Used to verify that header-derived features separate behavioural profiles without payload access.
- **Dataset B (Hjelmvik, Swedish Armed Forces CERT workshop):** Used to assess temporal stability, gate behaviour, and parameter sensitivity.

Across both datasets, flow-level indicators such as night-to-day ratios, domain diversity, and passive versus active traffic shares covary with their target DSM-5 criteria. The FASL gate produces stable daily summaries while exposing short-term excursions for further review.

## Safety, ethics, and privacy

- The tool aims to support early conversations, not clinical diagnosis or treatment decisions.
- Ensure you comply with local regulations and consent requirements before collecting network metadata.
- Keep all PCAP files and generated artifacts on trusted infrastructure; the deployed Streamlit instance performs the same computations but should still be used cautiously.

## Acknowledgements

If you use this work in research or demonstration material, please reference the Master thesis **Digital and Environmental Signals, Passive Sensing for Early Depression Detection** (University of St. Gallen, 2025) by Stephan Nef.
