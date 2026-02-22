import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, ast, warnings, json, re
from collections import Counter
from typing import List
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

os.environ["GRPC_ARG_KEEPALIVE_TIME_MS"]                            = "60000"
os.environ["GRPC_ARG_KEEPALIVE_TIMEOUT_MS"]                         = "20000"
os.environ["GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS"] = "60000"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Allocaid — Humanitarian Funding Fairness",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ISO3 → Country Name Mapping (main app)
# ============================================================
ISO_NAMES = {
    "AFG": "Afghanistan", "BFA": "Burkina Faso", "CAF": "Central African Republic",
    "COD": "DR Congo", "COL": "Colombia", "ETH": "Ethiopia", "HTI": "Haiti",
    "IRQ": "Iraq", "JOR": "Jordan", "LBN": "Lebanon", "MLI": "Mali",
    "MMR": "Myanmar", "MOZ": "Mozambique", "NER": "Niger", "NGA": "Nigeria",
    "PAK": "Pakistan", "PSE": "Palestine", "SDN": "Sudan", "SOM": "Somalia",
    "SSD": "South Sudan", "SYR": "Syria", "TCD": "Chad", "UKR": "Ukraine",
    "VEN": "Venezuela", "YEM": "Yemen"
}

# ============================================================
# CUSTOM CSS (main app)
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: transparent;
        padding: 1rem 0 0.5rem 0;
        margin-bottom: 1.5rem;
        color: #111827;
    }
    .main-header h1 { font-size: 2.2rem; margin-bottom: 0.3rem; color: #111827; }
    .main-header p { font-size: 1.05rem; opacity: 0.75; margin-bottom: 0; color: #374151; }

    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #e74c3c;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
    }
    .metric-card h3 { font-size: 1.8rem; margin: 0; color: #0f172a; }
    .metric-card p { font-size: 0.85rem; color: #666; margin: 0; }

    .insight-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    .insight-blue {
        background: #e0f2fe;
        border: 1px solid #0ea5e9;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #888;
        font-size: 0.85rem;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }

    /* ── Allocaid RAG Chat CSS ─────────────────────────────────────────── */
    :root{
        --accent:#e07b39;--accent2:#2563eb;
        --danger:#dc2626;--success:#16a34a;
        --muted:#6b7280;--text:#111827;--border:#e5e7eb;
        --surface:#ffffff;--surface2:#f9fafb;
    }
    .panel{background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;padding:1rem 1.2rem;box-shadow:0 1px 3px rgba(0,0,0,0.06);}
    .panel-title{font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:#6b7280;
        margin-bottom:0.8rem;display:flex;align-items:center;gap:0.4rem;}
    .panel-title::before{content:'';width:5px;height:5px;border-radius:50%;
        background:#e07b39;display:inline-block;}

    .chat-scroll{max-height:480px;overflow-y:auto;padding-right:4px;margin-bottom:0.6rem;}
    .msg{padding:0.65rem 0.9rem;border-radius:8px;margin-bottom:0.5rem;font-size:0.84rem;line-height:1.65;color:#111827;}
    .msg-user{background:#f3f4f6;border:1px solid #e5e7eb;}
    .msg-ai{background:#fff7f0;border:1px solid #fbd5b5;}
    .msg-role{font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;
        color:#9ca3af;margin-bottom:0.2rem;}
    .tag{display:inline-block;padding:1px 8px;border-radius:100px;font-size:0.67rem;margin:1px;font-weight:500;}
    .tag-red  {background:#fee2e2;color:#dc2626;border:1px solid #fca5a5;}
    .tag-green{background:#dcfce7;color:#16a34a;border:1px solid #86efac;}
    .tag-blue {background:#dbeafe;color:#2563eb;border:1px solid #93c5fd;}
    .tag-grey {background:#f3f4f6;color:#6b7280;border:1px solid #d1d5db;}
    .tag-orange{background:#fff7ed;color:#e07b39;border:1px solid #fed7aa;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA (main app)
# ============================================================
@st.cache_data
def load_data():
    # 1) Base panel from Excel
    panel = pd.read_excel("data/MASTER_PANEL_FINAL.xlsx")
    panel["Year"] = panel["Year"].astype(int)
    panel["actual_funding"] = pd.to_numeric(panel["Total_CBPF"], errors="coerce")

    # Pop_Used
    panel["Pop_Used"] = pd.to_numeric(panel["WB_Population"], errors="coerce")
    mask_pop = panel["Population"].notna()
    panel.loc[mask_pop, "Pop_Used"] = pd.to_numeric(panel.loc[mask_pop, "Population"], errors="coerce")

    # Need_Proxy
    pin = pd.to_numeric(panel["Total_PIN"], errors="coerce")
    risk = pd.to_numeric(panel["INFORM_Risk"], errors="coerce")
    vuln = pd.to_numeric(panel["Vulnerability"], errors="coerce")
    def _mm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)
    fallback = panel["Pop_Used"] * (0.6 * _mm(risk) + 0.4 * _mm(vuln))
    panel["Need_Proxy"] = pin.where(pin.notna(), fallback)

    # Efficiency
    panel["beneficiaries_per_million"] = np.where(
        panel["actual_funding"].notna() & (panel["actual_funding"] > 0),
        pd.to_numeric(panel["CBPF_Reached"], errors="coerce") / (panel["actual_funding"] / 1e6 + 1e-9),
        np.nan,
    )
    bpm = pd.Series(panel["beneficiaries_per_million"]).dropna()
    if len(bpm) > 0:
        med = bpm.median()
        mad = np.median(np.abs(bpm - med)) + 1e-9
        panel["efficiency_robust_z"] = 0.6745 * (panel["beneficiaries_per_million"] - med) / mad
    else:
        panel["efficiency_robust_z"] = np.nan

    # 2) Merge XGBoost predictions if CSV exists
    pred_merged = False
    for csv_name in [
        "FundingPredictionModels/outputs/scored_funding_fairness_reduced.csv",
        "FundingPredictionModels/outputs/scored_funding_fairness_full.csv",
        "data/scored_funding_fairness_reduced.csv",
        "data/scored_funding_fairness_full.csv",
    ]:
        if os.path.exists(csv_name):
            model_out = pd.read_csv(csv_name)
            merge_cols = ["ISO3", "Year"]
            for c in ["pred_funding", "funding_gap", "funding_ratio_gap", "flag_overlooked", "flag_overfunded"]:
                if c in model_out.columns:
                    merge_cols.append(c)
            model_out = model_out[merge_cols].drop_duplicates(["ISO3", "Year"])
            for c in merge_cols:
                if c not in ("ISO3", "Year") and c in panel.columns:
                    panel.drop(columns=[c], inplace=True)
            panel = panel.merge(model_out, on=["ISO3", "Year"], how="left")
            pred_merged = True
            break
    if not pred_merged:
        for c in ["pred_funding", "funding_gap", "funding_ratio_gap", "flag_overlooked", "flag_overfunded"]:
            if c not in panel.columns:
                panel[c] = np.nan

    # 3) Merge naive MODEL1
    naive_path = "data/MASTER_MODEL1_2025.xlsx"
    if os.path.exists(naive_path):
        naive = pd.read_excel(naive_path)
        naive = naive.rename(columns={
            "Dollars of CBPF funding per Person In Need in Average Program": "naive_dollars_per_pin",
            "Avg_PIN_per_Cluster": "naive_avg_pin_per_cluster",
            "Num_Active_Clusters": "naive_active_clusters",
        })
        naive["Year"] = naive["Year"].astype(int)
        keep = ["ISO3", "Year", "naive_dollars_per_pin", "naive_avg_pin_per_cluster", "naive_active_clusters"]
        keep = [c for c in keep if c in naive.columns]
        naive = naive[keep].drop_duplicates(["ISO3", "Year"])
        for c in keep:
            if c not in ("ISO3", "Year") and c in panel.columns:
                panel.drop(columns=[c], inplace=True)
        panel = panel.merge(naive, on=["ISO3", "Year"], how="left")

    # 4) Country name
    panel["Country"] = panel["ISO3"].map(ISO_NAMES).fillna(panel["ISO3"])
    scored = panel.copy()

    # 5) Compute peer benchmarking from panel (no CSV needed)
    bench = _compute_benchmarking(scored)
    return scored, bench


def _compute_benchmarking(panel: pd.DataFrame) -> pd.DataFrame:
    """Nearest-neighbor peer benchmarking computed directly from the panel."""
    BENCH_FEATURES = [
        "INFORM_Risk", "Vulnerability", "Conflict_Probability", "Food_Security",
        "Governance", "Access_Healthcare", "Uprooted_People", "Hazard_Exposure",
        "GDP_per_capita", "Pop_Used", "Density_per_km2",
    ]
    avail = [c for c in BENCH_FEATURES if c in panel.columns]
    sub = panel.dropna(subset=["actual_funding", "funding_ratio_gap"]).copy()
    sub = sub.dropna(subset=avail, thresh=len(avail) - 2)  # allow up to 2 NaN

    if len(sub) < 6 or not avail:
        return pd.DataFrame(columns=[
            "ISO3", "Year", "Country", "actual_funding", "pred_funding",
            "funding_ratio_gap", "neighbor_avg_ratio_gap",
            "neighbor_avg_funding", "relative_to_peers",
        ])

    X = sub[avail].fillna(sub[avail].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = min(5, len(sub) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)

    rows = []
    for i in range(len(sub)):
        neighbor_idx = indices[i, 1:]  # skip self
        neighbor_gaps = sub.iloc[neighbor_idx]["funding_ratio_gap"].values
        neighbor_funds = sub.iloc[neighbor_idx]["actual_funding"].values
        avg_gap = float(np.nanmean(neighbor_gaps))
        avg_fund = float(np.nanmean(neighbor_funds))
        own_gap = float(sub.iloc[i]["funding_ratio_gap"])
        rows.append({
            "ISO3": sub.iloc[i]["ISO3"],
            "Year": sub.iloc[i]["Year"],
            "Country": sub.iloc[i].get("Country", sub.iloc[i]["ISO3"]),
            "actual_funding": sub.iloc[i]["actual_funding"],
            "pred_funding": sub.iloc[i].get("pred_funding", np.nan),
            "funding_ratio_gap": own_gap,
            "neighbor_avg_ratio_gap": avg_gap,
            "neighbor_avg_funding": avg_fund,
            "relative_to_peers": own_gap - avg_gap,
        })
    return pd.DataFrame(rows)

scored_df, bench_df = load_data()

@st.cache_data
def load_cluster_highlights():
    path = "FundingPredictionModels/outputs/cluster_highlights.json"
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🔎 Allocaid</h1>
    <p>Databricks × United Nations — Where Does Humanitarian Funding Fall Short of Need?</p>
    <p style="font-size: 0.8rem; color: #6b7280; margin-top: 0.5rem;">
        Hacklytics 2026 | ML pipeline in Databricks (XGBoost + MLflow) • Visualized with Streamlit
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    available_years = sorted(scored_df["Year"].unique(), reverse=True)
    selected_year = st.selectbox("Year", available_years, index=0)
    continents = ["All"] + sorted(scored_df["Continent"].dropna().unique().tolist())
    selected_continent = st.selectbox("Region", continents)
    show_predicted_only = st.checkbox("Only show rows with model predictions", value=True)
    st.markdown("---")
    st.markdown("### 📊 How It Works")
    st.markdown(
        "Allocaid uses an **XGBoost model** trained on humanitarian indicators "
        "(INFORM Risk, Vulnerability, Population, Conflict Probability, etc.) to predict "
        "what CBPF funding **should** look like given a country's need level. "
        "The gap between actual and expected funding reveals which crises are **overlooked**."
    )
    st.markdown("---")
    st.markdown(
        "**Pipeline:** HumData → Databricks (Bronze/Silver/Gold) → MLflow → Streamlit"
    )
    st.markdown(
        "**Data:** [UN OCHA](https://data.humdata.org/) • "
        "[CBPF](https://cbpf.data.unocha.org/) • "
        "[INFORM](https://drmkc.jrc.ec.europa.eu/inform-index/)"
    )

# ============================================================
# APPLY FILTERS
# ============================================================
filtered = scored_df.copy()
if selected_year:
    filtered = filtered[filtered["Year"] == selected_year]
if selected_continent != "All":
    filtered = filtered[filtered["Continent"] == selected_continent]
if show_predicted_only:
    filtered = filtered[filtered["pred_funding"].notna()]

bench_filtered = bench_df.copy()
if selected_year:
    bench_filtered = bench_filtered[bench_filtered["Year"] == selected_year]
if show_predicted_only:
    bench_filtered = bench_filtered[bench_filtered["pred_funding"].notna()]

# ============================================================
# KEY METRICS
# ============================================================
if len(filtered) > 0:
    total_actual = filtered["actual_funding"].sum()
    total_predicted = filtered["pred_funding"].sum() if filtered["pred_funding"].notna().any() else 0
    n_overlooked = filtered["flag_overlooked"].sum()
    n_countries = filtered["ISO3"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>${total_actual / 1e6:,.0f}M</h3>
            <p>Total CBPF Funding ({selected_year})</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-color: #f39c12;">
            <h3>${total_predicted / 1e6:,.0f}M</h3>
            <p>Model-Expected Funding</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-color: #e74c3c;">
            <h3>{int(n_overlooked)}</h3>
            <p>Overlooked Crises (≥35% below expected)</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-color: #3498db;">
            <h3>{n_countries}</h3>
            <p>Countries Tracked</p>
        </div>
        """, unsafe_allow_html=True)

    if filtered["funding_ratio_gap"].notna().any():
        worst_idx = filtered["funding_ratio_gap"].idxmin()
        worst = filtered.loc[worst_idx]
        st.markdown(f"""
        <div class="insight-box">
            ⚠️ <strong>Key Finding ({selected_year}):</strong> {worst['Country']} received
            <strong>${worst['actual_funding']/1e6:.1f}M</strong> in CBPF funding —
            but our model expected <strong>${worst['pred_funding']/1e6:.1f}M</strong> based on
            humanitarian need. That's <strong>{abs(worst['funding_ratio_gap']):.0%} below</strong>
            what comparable crises receive.
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Global Map",
    "🌐 Risk Clusters",
    "⚡ Efficiency",
    "🤖 Allocaid Intelligence",
    "🧪 Model Comparison",
    "📋 Data Explorer",
    
])

# ============================================================
# TAB 1: GLOBAL MAP
# ============================================================
with tab1:
    st.markdown("### Funding Fairness Map")
    st.markdown(
        "Each country is colored by its **funding ratio gap**: how much actual CBPF funding "
        "deviates from what our model predicts based on humanitarian need. "
        "**Red = underfunded**, **Blue = overfunded** relative to expected."
    )

    map_data = filtered[filtered["funding_ratio_gap"].notna()].copy()

    if len(map_data) > 0:
        map_data["gap_display"] = map_data["funding_ratio_gap"].clip(-1, 2)
        map_data["gap_pct"] = (map_data["funding_ratio_gap"] * 100).round(1)
        map_data["actual_M"] = (map_data["actual_funding"] / 1e6).round(1)
        map_data["expected_M"] = (map_data["pred_funding"] / 1e6).round(1)

        fig_map = px.choropleth(
            map_data,
            locations="ISO3",
            color="gap_display",
            hover_name="Country",
            hover_data={
                "gap_display": False,
                "gap_pct": ":.1f",
                "actual_M": ":.1f",
                "expected_M": ":.1f",
                "INFORM_Risk": ":.1f",
                "ISO3": False,
            },
            color_continuous_scale="RdBu",
            range_color=[-1, 1],
            color_continuous_midpoint=0,
            labels={
                "gap_pct": "Funding Gap %",
                "actual_M": "Actual ($M)",
                "expected_M": "Expected ($M)",
                "INFORM_Risk": "INFORM Risk",
            },
            title=f"CBPF Funding Fairness — {selected_year}"
        )
        fig_map.update_layout(
            height=550,
            margin=dict(l=0, r=0, t=40, b=0),
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type="natural earth",
                bgcolor="rgba(0,0,0,0)"
            ),
            coloraxis_colorbar=dict(
                title="Funding Gap",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-100%", "-50%", "Fair", "+50%", "+100%+"],
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("### Bubble Map — Need vs. Funding Gap")

        fig_bubble = px.scatter_geo(
            map_data,
            lat="Latitude",
            lon="Longitude",
            size=np.sqrt(map_data["Need_Proxy"].clip(lower=1).abs()),
            color="funding_ratio_gap",
            hover_name="Country",
            color_continuous_scale="RdBu",
            range_color=[-1, 1],
            color_continuous_midpoint=0,
            size_max=25,
            hover_data={
                "actual_M": ":.1f",
                "expected_M": ":.1f",
                "gap_pct": ":.1f",
                "Latitude": False,
                "Longitude": False,
            },
            labels={
                "actual_M": "Actual ($M)",
                "expected_M": "Expected ($M)",
                "gap_pct": "Gap %",
            }
        )
        fig_bubble.update_layout(
            height=500,
            margin=dict(l=0, r=0, t=10, b=0),
            geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth")
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.info("No data with model predictions for the selected filters.")
# ============================================================
# TAB 2: FUNDING GAPS
# ============================================================













with tab2:

    # ── loaders ───────────────────────────────────────────────────────────
    @st.cache_data
    def load_cluster_highlights_tab7():
        path = "FundingPredictionModels/outputs/cluster_highlights.json"
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    @st.cache_data
    def load_risk_panel():
        """Load risk indicators for the radar chart."""
        path = "data/MASTER_PANEL_FINAL.xlsx"
        try:
            df = pd.read_excel(path)
            df["Year"] = df["Year"].astype(int)
            return df
        except Exception:
            return None

    RISK_COLS_RADAR = [
        "INFORM_Risk", "Vulnerability", "Conflict_Probability",
        "Food_Security", "Governance", "Access_Healthcare",
        "Hazard_Exposure", "Uprooted_People",
    ]
    RISK_LABELS_RADAR = [
        "INFORM Risk", "Vulnerability", "Conflict", "Food Insecurity",
        "Governance", "Healthcare Access", "Hazard Exposure", "Displacement",
    ]

    # Lat/Lon centroids for connection lines
    COUNTRY_COORDS = {
        "AFG": (33.9, 67.7), "BFA": (12.4, -1.6), "CAF": (6.6, 20.9),
        "COD": (-4.0, 21.8), "COL": (4.6, -74.1), "ETH": (9.1, 40.5),
        "HTI": (19.0, -72.4), "IRQ": (33.2, 43.7), "JOR": (31.0, 36.6),
        "LBN": (33.9, 35.5), "MLI": (17.6, -4.0), "MMR": (19.8, 96.2),
        "MOZ": (-18.7, 35.5), "NER": (17.6, 8.1), "NGA": (9.1, 8.7),
        "PAK": (30.4, 69.3), "PSE": (31.9, 35.2), "SDN": (15.5, 32.6),
        "SOM": (5.2, 46.2), "SSD": (7.9, 29.9), "SYR": (35.0, 38.0),
        "TCD": (15.5, 18.7), "UKR": (48.4, 31.2), "VEN": (6.4, -66.6),
        "YEM": (15.6, 48.5),
    }

    cluster_data = load_cluster_highlights_tab7()
    risk_panel = load_risk_panel()

    if not cluster_data:
        st.warning(
            "Cluster highlights not found at "
            "`FundingPredictionModels/outputs/cluster_highlights.json`. "
            "Run `cluster_divergence_model.py` first."
        )
    else:
        # ── Index data by country ─────────────────────────────────────────
        country_entries = {}
        for entry in cluster_data:
            iso = entry["target_ISO3"]
            country_entries.setdefault(iso, []).append(entry)

        all_isos = sorted(country_entries.keys())

        # ── Header ────────────────────────────────────────────────────────
        st.markdown(
            "<div style='margin-bottom:0.6rem;'>"
            "<span style='font-size:1.5rem;font-weight:700;'>🌐 Peer Intelligence Map</span>"
            "<span style='font-size:0.78rem;color:#9ca3af;margin-left:0.8rem;"
            "text-transform:uppercase;letter-spacing:0.08em;'>"
            "Risk-Profile Matching & Cluster Divergence</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "Countries with **similar humanitarian risk profiles** are connected as peers. "
            "Select a country to see how its **sector-level funding allocations** compare "
            "to its closest risk-matched peers — where they align, and where they diverge."
        )

        # ── Country selector ──────────────────────────────────────────────
        col_sel, col_spacer = st.columns([3, 5])
        with col_sel:
            options = [f"{ISO_NAMES.get(iso, iso)} ({iso})" for iso in all_isos]
            selected_str = st.selectbox(
                "Select a country",
                options,
                index=0,
                key="tab7_country",
                label_visibility="collapsed",
            )
            selected_iso = all_isos[options.index(selected_str)]

        entries = sorted(country_entries[selected_iso], key=lambda x: x["match_rank"])
        country_name = ISO_NAMES.get(selected_iso, selected_iso)
        target_year = entries[0]["target_Year"] if entries else "?"

        # ── CONNECTION MAP ────────────────────────────────────────────────
        fig_conn = go.Figure()

        # Background: all tracked countries as grey dots
        bg_lats = [COUNTRY_COORDS[iso][0] for iso in all_isos if iso in COUNTRY_COORDS]
        bg_lons = [COUNTRY_COORDS[iso][1] for iso in all_isos if iso in COUNTRY_COORDS]
        bg_names = [ISO_NAMES.get(iso, iso) for iso in all_isos if iso in COUNTRY_COORDS]
        fig_conn.add_trace(go.Scattergeo(
            lat=bg_lats, lon=bg_lons, text=bg_names,
            mode="markers",
            marker=dict(size=5, color="#d1d5db", line=dict(width=0.5, color="#9ca3af")),
            hoverinfo="text",
            showlegend=False,
        ))

        # Connection lines + peer dots
        t_coord = COUNTRY_COORDS.get(selected_iso)
        peer_colors = ["#e07b39", "#2563eb", "#8b5cf6"]  # rank 1, 2, 3
        peer_widths = [3, 2, 1.5]
        peer_sizes = [14, 11, 9]

        for i, entry in enumerate(entries[:3]):
            m_iso = entry["match_ISO3"]
            m_coord = COUNTRY_COORDS.get(m_iso)
            if not t_coord or not m_coord:
                continue
            m_name = ISO_NAMES.get(m_iso, m_iso)
            color = peer_colors[i] if i < len(peer_colors) else "#9ca3af"
            width = peer_widths[i] if i < len(peer_widths) else 1
            psize = peer_sizes[i] if i < len(peer_sizes) else 8

            # Arc line
            fig_conn.add_trace(go.Scattergeo(
                lat=[t_coord[0], m_coord[0]],
                lon=[t_coord[1], m_coord[1]],
                mode="lines",
                line=dict(width=width, color=color, dash="solid"),
                opacity=0.7,
                showlegend=False,
                hoverinfo="skip",
            ))

            # Peer dot
            n_div = len(entry.get("divergent_clusters", []))
            n_sim = len(entry.get("similar_clusters", []))
            fig_conn.add_trace(go.Scattergeo(
                lat=[m_coord[0]], lon=[m_coord[1]],
                text=[f"<b>#{entry['match_rank']} {m_name}</b><br>"
                      f"Distance: {entry['risk_distance']:.3f}<br>"
                      f"Similar: {n_sim} clusters | Divergent: {n_div} clusters"],
                mode="markers",
                marker=dict(size=psize, color=color,
                            line=dict(width=2, color="white")),
                hoverinfo="text",
                showlegend=False,
            ))

        # Target country dot
        if t_coord:
            fig_conn.add_trace(go.Scattergeo(
                lat=[t_coord[0]], lon=[t_coord[1]],
                text=[f"<b>{country_name}</b> (selected)"],
                mode="markers",
                marker=dict(
                    size=18, color="#dc2626",
                    line=dict(width=3, color="white"),
                    symbol="circle",
                ),
                hoverinfo="text",
                showlegend=False,
            ))

        fig_conn.update_layout(
            height=460,
            margin=dict(l=0, r=0, t=10, b=0),
            geo=dict(
                showframe=False,
                showcoastlines=True,
                coastlinecolor="#d1d5db",
                showland=True,
                landcolor="#f9fafb",
                showocean=True,
                oceancolor="#f0f7ff",
                showcountries=True,
                countrycolor="#e5e7eb",
                projection_type="natural earth",
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_conn, use_container_width=True)

        # ── Legend below map ──────────────────────────────────────────────
        st.markdown(
            f"<div style='display:flex;gap:1.5rem;justify-content:center;"
            f"font-size:0.78rem;color:#6b7280;margin:-0.5rem 0 1rem 0;'>"
            f"<span>🔴 <b>{country_name}</b> (selected)</span>"
            f"<span style='color:#e07b39;'>● #1 Peer</span>"
            f"<span style='color:#2563eb;'>● #2 Peer</span>"
            f"<span style='color:#8b5cf6;'>● #3 Peer</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── STAT CARDS ────────────────────────────────────────────────────
        rank1 = entries[0] if entries else None
        if rank1:
            r1_name = ISO_NAMES.get(rank1["match_ISO3"], rank1["match_ISO3"])
            r1_sim = rank1.get("similar_clusters", [])
            r1_div = rank1.get("divergent_clusters", [])

            # Compute a "peer alignment score" — how similar overall
            all_sim_diffs = [abs(s["frac_diff"]) for s in r1_sim]
            all_div_diffs = [abs(d["frac_diff"]) for d in r1_div]
            avg_sim = np.mean(all_sim_diffs) * 100 if all_sim_diffs else 0
            avg_div = np.mean(all_div_diffs) * 100 if all_div_diffs else 0

            # Biggest divergence headline
            biggest_div = max(r1_div, key=lambda d: abs(d["frac_diff"])) if r1_div else None

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f"<div class='metric-card' style='border-color:#e07b39;'>"
                    f"<h3>{r1_name}</h3>"
                    f"<p>Closest Risk-Profile Peer</p></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"<div class='metric-card' style='border-color:#22c55e;'>"
                    f"<h3>{rank1['risk_distance']:.3f}</h3>"
                    f"<p>Risk Distance (lower = more similar)</p></div>",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"<div class='metric-card' style='border-color:#3b82f6;'>"
                    f"<h3>{len(r1_sim)} aligned / {len(r1_div)} divergent</h3>"
                    f"<p>Cluster Allocation Match</p></div>",
                    unsafe_allow_html=True,
                )
            with c4:
                if biggest_div:
                    arrow = "▲" if "OVER" in biggest_div["direction"] else "▼"
                    st.markdown(
                        f"<div class='metric-card' style='border-color:#ef4444;'>"
                        f"<h3>{arrow} {abs(biggest_div['frac_diff']):.1%}</h3>"
                        f"<p>Biggest Divergence: {biggest_div['cluster']}</p></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div class='metric-card' style='border-color:#94a3b8;'>"
                        f"<h3>—</h3><p>No divergences found</p></div>",
                        unsafe_allow_html=True,
                    )

            # ── INSIGHT BOX ───────────────────────────────────────────────
            if biggest_div:
                over = "OVER" in biggest_div["direction"]
                verb = "allocates significantly more" if over else "allocates significantly less"
                peer_verb = "allocates less" if over else "allocates more"
                st.markdown(
                    f"<div class='insight-box'>"
                    f"⚡ <strong>Key Divergence:</strong> {country_name} {verb} to "
                    f"<strong>{biggest_div['cluster']}</strong> "
                    f"({biggest_div['target_frac']:.1%} of budget = "
                    f"${biggest_div['target_budget_usd']/1e6:.2f}M) compared to peer "
                    f"{r1_name} ({biggest_div['match_frac']:.1%} = "
                    f"${biggest_div['match_budget_usd']/1e6:.2f}M). "
                    f"This {abs(biggest_div['frac_diff']):.1%} gap is the largest "
                    f"allocation divergence between these risk-matched peers."
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── RADAR CHART: Risk Profile Comparison ──────────────────────────
        # ── CLUSTER ALLOCATION COMPARISON ─────────────────────────────────
        col_radar, col_bars = st.columns([1, 1])

        with col_radar:
            st.markdown(f"#### Risk Profile Overlay")

            radar_ok = False
            if risk_panel is not None and rank1:
                avail_cols = [c for c in RISK_COLS_RADAR if c in risk_panel.columns]
                avail_labels = [RISK_LABELS_RADAR[i] for i, c in enumerate(RISK_COLS_RADAR) if c in risk_panel.columns]

                t_row = risk_panel[
                    (risk_panel["ISO3"] == selected_iso) &
                    (risk_panel["Year"] == target_year)
                ]
                m_row = risk_panel[
                    (risk_panel["ISO3"] == rank1["match_ISO3"]) &
                    (risk_panel["Year"] == rank1["match_Year"])
                ]

                if not t_row.empty and not m_row.empty and len(avail_cols) >= 3:
                    radar_ok = True
                    t_vals = t_row[avail_cols].iloc[0].values.astype(float)
                    m_vals = m_row[avail_cols].iloc[0].values.astype(float)

                    # Normalize to 0-10 for visual comparison
                    all_vals = np.concatenate([t_vals, m_vals])
                    vmin = np.nanmin(all_vals)
                    vmax = np.nanmax(all_vals)
                    rng = vmax - vmin + 1e-9
                    t_norm = ((t_vals - vmin) / rng * 10)
                    m_norm = ((m_vals - vmin) / rng * 10)

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=np.append(t_norm, t_norm[0]),
                        theta=avail_labels + [avail_labels[0]],
                        fill="toself",
                        fillcolor="rgba(220, 38, 38, 0.15)",
                        line=dict(color="#dc2626", width=2.5),
                        name=country_name,
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=np.append(m_norm, m_norm[0]),
                        theta=avail_labels + [avail_labels[0]],
                        fill="toself",
                        fillcolor="rgba(224, 123, 57, 0.15)",
                        line=dict(color="#e07b39", width=2, dash="dash"),
                        name=ISO_NAMES.get(rank1["match_ISO3"], rank1["match_ISO3"]),
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True, range=[0, 10],
                                gridcolor="#e5e7eb", linecolor="#e5e7eb",
                                tickfont=dict(size=9, color="#9ca3af"),
                            ),
                            angularaxis=dict(
                                gridcolor="#e5e7eb", linecolor="#e5e7eb",
                                tickfont=dict(size=10, color="#374151"),
                            ),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="sans-serif", size=11, color="#111827"),
                        legend=dict(
                            orientation="h", yanchor="bottom", y=-0.15,
                            xanchor="center", x=0.5,
                            bgcolor="rgba(0,0,0,0)",
                            font=dict(size=11),
                        ),
                        height=380,
                        margin=dict(l=50, r=50, t=20, b=50),
                        showlegend=True,
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

            if not radar_ok:
                st.info(
                    "Risk indicator data not available for radar comparison. "
                    "Ensure `MASTER_PANEL_FINAL.xlsx` is in `data/`."
                )

        with col_bars:
            st.markdown(f"#### Cluster Allocation: {country_name} vs {ISO_NAMES.get(rank1['match_ISO3'], rank1['match_ISO3']) if rank1 else '?'}")

            if rank1:
                sim_clusters = rank1.get("similar_clusters", [])
                div_clusters = rank1.get("divergent_clusters", [])
                all_clusters = sim_clusters + div_clusters

                if all_clusters:
                    bar_df = pd.DataFrame(all_clusters)
                    bar_df = bar_df.sort_values("abs_diff", ascending=True)
                    bar_df["type"] = bar_df["direction"].apply(
                        lambda d: "Similar" if d == "SIMILAR" else "Divergent"
                    )

                    # Paired horizontal bar chart
                    fig_bars = go.Figure()

                    # Target country bars
                    fig_bars.add_trace(go.Bar(
                        y=bar_df["cluster"],
                        x=bar_df["target_frac"] * 100,
                        orientation="h",
                        name=country_name,
                        marker_color="#dc2626",
                        marker_line=dict(width=0),
                        opacity=0.85,
                        text=bar_df["target_frac"].apply(lambda v: f"{v:.1%}"),
                        textposition="auto",
                        textfont=dict(size=10),
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            f"{country_name}: " + "%{x:.1f}%<br>"
                            "<extra></extra>"
                        ),
                    ))

                    # Peer country bars
                    r1_name_bar = ISO_NAMES.get(rank1["match_ISO3"], rank1["match_ISO3"])
                    fig_bars.add_trace(go.Bar(
                        y=bar_df["cluster"],
                        x=bar_df["match_frac"] * 100,
                        orientation="h",
                        name=r1_name_bar,
                        marker_color="#e07b39",
                        marker_line=dict(width=0),
                        opacity=0.85,
                        text=bar_df["match_frac"].apply(lambda v: f"{v:.1%}"),
                        textposition="auto",
                        textfont=dict(size=10),
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            f"{r1_name_bar}: " + "%{x:.1f}%<br>"
                            "<extra></extra>"
                        ),
                    ))

                    fig_bars.update_layout(
                        barmode="group",
                        height=380,
                        margin=dict(l=0, r=10, t=10, b=40),
                        xaxis=dict(
                            title="% of Total CBPF Allocation",
                            gridcolor="#f3f4f6",
                            ticksuffix="%",
                        ),
                        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(249,250,251,0.6)",
                        font=dict(family="sans-serif", size=11, color="#111827"),
                        legend=dict(
                            orientation="h", yanchor="bottom", y=-0.2,
                            xanchor="center", x=0.5,
                            bgcolor="rgba(0,0,0,0)",
                        ),
                    )
                    st.plotly_chart(fig_bars, use_container_width=True)
                else:
                    st.info("No cluster comparison data for this pair.")

        st.markdown("---")

        # ── DETAILED PEER CARDS ───────────────────────────────────────────
        st.markdown(f"#### All Peer Matches for {country_name} ({target_year})")

        for entry in entries:
            m_iso = entry["match_ISO3"]
            m_name = ISO_NAMES.get(m_iso, m_iso)
            rank = entry["match_rank"]
            dist = entry["risk_distance"]
            similar = entry.get("similar_clusters", [])
            divergent = entry.get("divergent_clusters", [])

            color = peer_colors[rank - 1] if rank <= len(peer_colors) else "#9ca3af"

            with st.expander(
                f"#{rank}  {m_name}  ·  Risk Distance: {dist:.3f}  ·  "
                f"{len(similar)} similar / {len(divergent)} divergent",
                expanded=(rank == 1),
            ):
                col_s, col_d = st.columns(2)

                with col_s:
                    st.markdown("**✅ Aligned Clusters** — similar allocation share")
                    if similar:
                        for sc in similar:
                            t_pct = sc["target_frac"] * 100
                            m_pct = sc["match_frac"] * 100
                            diff_pct = abs(sc["frac_diff"]) * 100
                            st.markdown(
                                f"<div style='background:#f0fdf4;border-left:3px solid #22c55e;"
                                f"padding:0.6rem 0.8rem;margin-bottom:0.5rem;"
                                f"border-radius:0 6px 6px 0;font-size:0.84rem;'>"
                                f"<strong>{sc['cluster']}</strong><br>"
                                f"<span style='color:#dc2626;'>{country_name}:</span> "
                                f"<strong>{t_pct:.1f}%</strong>"
                                f" · ${sc['target_budget_usd']/1e6:.2f}M<br>"
                                f"<span style='color:#e07b39;'>{m_name}:</span> "
                                f"<strong>{m_pct:.1f}%</strong>"
                                f" · ${sc['match_budget_usd']/1e6:.2f}M<br>"
                                f"<span style='color:#9ca3af;font-size:0.76rem;'>"
                                f"Δ {diff_pct:.1f}pp — nearly identical allocation</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No closely aligned clusters.")

                with col_d:
                    st.markdown("**⚠️ Divergent Clusters** — allocation mismatch")
                    if divergent:
                        for dc in divergent:
                            over = "OVER" in dc["direction"]
                            bg = "#fffbeb" if over else "#fef2f2"
                            border = "#f59e0b" if over else "#ef4444"
                            icon = "🔼" if over else "🔽"
                            label = "Over-allocated" if over else "Under-allocated"
                            t_pct = dc["target_frac"] * 100
                            m_pct = dc["match_frac"] * 100
                            diff_pct = abs(dc["frac_diff"]) * 100

                            st.markdown(
                                f"<div style='background:{bg};border-left:3px solid {border};"
                                f"padding:0.6rem 0.8rem;margin-bottom:0.5rem;"
                                f"border-radius:0 6px 6px 0;font-size:0.84rem;'>"
                                f"{icon} <strong>{dc['cluster']}</strong>"
                                f" — <em>{label}</em><br>"
                                f"<span style='color:#dc2626;'>{country_name}:</span> "
                                f"<strong>{t_pct:.1f}%</strong>"
                                f" · ${dc['target_budget_usd']/1e6:.2f}M<br>"
                                f"<span style='color:#e07b39;'>{m_name}:</span> "
                                f"<strong>{m_pct:.1f}%</strong>"
                                f" · ${dc['match_budget_usd']/1e6:.2f}M<br>"
                                f"<span style='color:#9ca3af;font-size:0.76rem;'>"
                                f"Δ {diff_pct:.1f}pp — {country_name} puts "
                                f"{'more' if over else 'less'} budget here vs peer</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No major divergences.")

                # ── Mini budget comparison table ──────────────────────────
                all_entries = (
                    [{"cluster": s["cluster"], "type": "Similar",
                      f"{country_name} %": f"{s['target_frac']:.1%}",
                      f"{m_name} %": f"{s['match_frac']:.1%}",
                      f"{country_name} $": f"${s['target_budget_usd']/1e6:.2f}M",
                      f"{m_name} $": f"${s['match_budget_usd']/1e6:.2f}M",
                      "Δ": f"{abs(s['frac_diff']):.1%}"}
                     for s in similar] +
                    [{"cluster": d["cluster"], "type": d["direction"].replace("-", " ").title(),
                      f"{country_name} %": f"{d['target_frac']:.1%}",
                      f"{m_name} %": f"{d['match_frac']:.1%}",
                      f"{country_name} $": f"${d['target_budget_usd']/1e6:.2f}M",
                      f"{m_name} $": f"${d['match_budget_usd']/1e6:.2f}M",
                      "Δ": f"{abs(d['frac_diff']):.1%}"}
                     for d in divergent]
                )
                if all_entries:
                    st.markdown("**📊 Summary Table**")
                    st.dataframe(
                        pd.DataFrame(all_entries),
                        use_container_width=True,
                        hide_index=True,
                    )




















#TAB 3: PEER BENCHMARKING
# ============================================================


with tab3:
    st.markdown("### CBPF Delivery Efficiency")
    st.markdown(
        "How many beneficiaries does each dollar of CBPF funding reach? "
        "This metric helps identify which country operations are unusually efficient "
        "or inefficient at converting funding into impact."
    )
    
    eff_data = filtered[filtered["beneficiaries_per_million"].notna()].copy()
    eff_data = eff_data[eff_data["beneficiaries_per_million"] > 0]
    
    if len(eff_data) > 0:
        eff_data = eff_data.sort_values("beneficiaries_per_million", ascending=False)
        
        fig_eff = px.bar(
            eff_data,
            x="beneficiaries_per_million",
            y="Country",
            orientation="h",
            color="efficiency_robust_z",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            labels={
                "beneficiaries_per_million": "Beneficiaries per $1M",
                "efficiency_robust_z": "Efficiency Z-Score"
            },
            hover_data={
                "CBPF_Reached": ":,.0f",
                "actual_funding": ":,.0f",
            }
        )
        fig_eff.update_layout(
            height=max(400, len(eff_data) * 30),
            margin=dict(l=0, t=10),
            xaxis_title="Beneficiaries Reached per $1M CBPF",
        )
        st.plotly_chart(fig_eff, use_container_width=True)
        
        # Scatter: efficiency vs funding gap
        st.markdown("### Efficiency vs. Funding Gap")
        st.markdown("Are the most efficient operations also the most underfunded?")
        
        eff_scatter = filtered[
            (filtered["beneficiaries_per_million"].notna()) & 
            (filtered["funding_ratio_gap"].notna()) &
            (filtered["beneficiaries_per_million"] > 0)
        ].copy()
        
        if len(eff_scatter) > 0:
            fig_eff_scatter = px.scatter(
                eff_scatter,
                x="funding_ratio_gap",
                y="beneficiaries_per_million",
                text="Country",
                color="Continent",
                size="actual_funding",
                size_max=40,
                labels={
                    "funding_ratio_gap": "Funding Gap (← underfunded | overfunded →)",
                    "beneficiaries_per_million": "Beneficiaries per $1M",
                    "Continent": "Region",
                },
                hover_data={"actual_funding": ":,.0f"}
            )
            fig_eff_scatter.update_traces(textposition="top center", textfont_size=9)
            fig_eff_scatter.update_layout(
                height=500,
                margin=dict(t=10),
                xaxis_tickformat=".0%",
            )
            fig_eff_scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig_eff_scatter, use_container_width=True)
            
            st.markdown(
                "*Top-left quadrant = high efficiency but underfunded — these are the crises "
                "that deliver the most impact per dollar yet receive less than expected.*"
            )
    else:
        st.info("No efficiency data available for the selected filters.")

























# ============================================================
# TAB 4: EFFICIENCY
# ============================================================

# ── RAG config ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
RAG_CSV_PATH = "datanew.csv"
TOP_K        = 6

COUNTRY_ISO = {
    "AF":"Afghanistan","BD":"Bangladesh","ET":"Ethiopia","IQ":"Iraq",
    "ML":"Mali","NE":"Niger","PK":"Pakistan","PS":"Palestine","SD":"Sudan",
    "SY":"Syria","TD":"Chad","YE":"Yemen","SO":"Somalia","SS":"South Sudan",
    "CD":"DR Congo","CF":"Central African Republic","NG":"Nigeria",
    "MM":"Myanmar","UA":"Ukraine","LB":"Lebanon","HT":"Haiti",
    "VE":"Venezuela","MZ":"Mozambique","BF":"Burkina Faso","BI":"Burundi",
    "CM":"Cameroon","CO":"Colombia","ER":"Eritrea","LY":"Libya","ZW":"Zimbabwe",
    "KE":"Kenya","UG":"Uganda","TZ":"Tanzania","RW":"Rwanda","ZA":"South Africa",
}
CRISIS_SEVERITY = {
    "YE":10,"SY":10,"SS":9,"SD":9,"SO":9,"AF":9,"CD":8,"CF":8,"ET":8,
    "MM":8,"NG":7,"ML":7,"NE":7,"IQ":7,"LB":7,"HT":7,"TD":7,"UA":8,
    "BF":8,"MZ":6,"CM":6,"BD":5,"PK":6,"PS":8,"LY":7,"BI":6,"ER":6,
    "VE":6,"CO":5,"ZW":5,"KE":5,"UG":4,"TZ":4,"RW":4,"ZA":4,
}
TEXT_COLUMNS_RAG = [
    "title_narrative","description_narrative","sector_narrative",
    "recipient_country_narrative","participating_org_narrative",
    "result_title_narrative","policy_marker_narrative",
]

RAG_PROMPTS = [
    {"id":"bbr_outliers","icon":"⚠️","title":"Flag Budget Outliers",
     "desc":"Projects with unusually high or low budget-to-activity ratios",
     "query":"Flag projects with unusually high or low budget ratios compared to similar projects. Identify outliers.",
     "viz":"bbr"},
    {"id":"neglected","icon":"🔴","title":"Most Neglected Crises",
     "desc":"Countries where severity far exceeds funding coverage",
     "query":"Which crises are most neglected? Compare crisis severity scores against total funding.",
     "viz":"neglect"},
    {"id":"benchmarks","icon":"📐","title":"Benchmark Projects",
     "desc":"Find well-funded comparable projects for benchmarking",
     "query":"Identify the best benchmark projects — well-funded, high-impact, replicable across similar crisis contexts.",
     "viz":"benchmark"},
    {"id":"sector_gaps","icon":"📊","title":"Sector Funding Gaps",
     "desc":"Which humanitarian clusters are chronically underfunded?",
     "query":"Which sectors and humanitarian clusters receive the least funding relative to need?",
     "viz":"treemap"},
    {"id":"attention_fade","icon":"📉","title":"Fading Crisis Attention",
     "desc":"Crises that lost donor attention over time",
     "query":"Which crises experienced the sharpest decline in project funding after 2018? Show funding fatigue.",
     "viz":"heatmap"},
    {"id":"food_security","icon":"🌾","title":"Food Security Gaps",
     "desc":"Underfunded food and nutrition projects by country",
     "query":"Analyze food security and nutrition project budgets. Which countries are most underfunded in this sector?",
     "viz":"sector_bar","sector_kw":"food"},
    {"id":"health","icon":"🏥","title":"Health Cluster Analysis",
     "desc":"Health project coverage vs crisis severity",
     "query":"Analyze health cluster projects. Flag countries where health funding is lowest relative to crisis severity.",
     "viz":"sector_bar","sector_kw":"health"},
    {"id":"protection","icon":"🛡️","title":"Protection Funding",
     "desc":"Child protection and GBV project coverage",
     "query":"Analyze protection cluster funding including child protection and GBV. Which contexts are most underserved?",
     "viz":"sector_bar","sector_kw":"protection"},
    {"id":"org_efficiency","icon":"🏛️","title":"Organization Efficiency",
     "desc":"Which organizations deliver the most projects per dollar?",
     "query":"Which organizations have the best budget efficiency — most projects relative to total spend? Flag high and low performers.",
     "viz":"org_bar"},
    {"id":"emergency_response","icon":"🚨","title":"Emergency Response Coverage",
     "desc":"Emergency response project gaps by severity tier",
     "query":"Analyze emergency response project coverage. Which high-severity countries have the fewest emergency projects?",
     "viz":"scatter"},
    {"id":"policy_recs","icon":"📋","title":"Policy Recommendations",
     "desc":"AI-generated actionable recommendations from data",
     "query":"Based on funding gaps, neglect scores, and project data, generate 3 specific policy recommendations for donors and humanitarian coordinators.",
     "viz":"none"},
    {"id":"compare_sudan_yemen","icon":"⚖️","title":"Sudan vs Yemen",
     "desc":"Side-by-side funding comparison for two top crises",
     "query":"Compare Sudan and Yemen across total budget, project count, sector coverage, and funding efficiency. Which is more overlooked?",
     "viz":"compare","countries":["SD","YE"]},
]

# ── RAG data helpers ──────────────────────────────────────────────────────────
def safe_parse(val) -> str:
    if pd.isna(val) or val == "": return ""
    try:
        p = ast.literal_eval(str(val))
        if isinstance(p, list):
            return " | ".join(str(v) for v in p if str(v) not in ["nan","None",""])
        return str(p)
    except: return str(val)

def safe_list(val) -> list:
    if pd.isna(val): return []
    try:
        p = ast.literal_eval(str(val))
        return p if isinstance(p, list) else [p]
    except: return [str(val)]

def extract_max_budget(val) -> float:
    nums = [x for x in safe_list(val) if isinstance(x,(int,float)) and x>0]
    return max(nums) if nums else 0.0

def extract_years(val) -> List[int]:
    years = []
    for d in safe_list(val):
        try: years.append(int(str(d)[:4]))
        except: pass
    return [y for y in years if 1990<y<2035]

@st.cache_data(show_spinner=False)
def load_rag_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["_budget"]     = df["budget_value"].apply(extract_max_budget)
    df["_years"]      = df["activity_date_iso_date"].apply(extract_years)
    df["_start_year"] = df["_years"].apply(lambda y: min(y) if y else None)
    df["_end_year"]   = df["_years"].apply(lambda y: max(y) if y else None)
    df["_countries"]  = df["recipient_country_code"].apply(safe_list)
    df["_sectors"]    = df["sector_narrative"].apply(safe_list)
    df["_title"]      = df["title_narrative"].apply(safe_parse)
    df["_orgs"]       = df["participating_org_narrative"].apply(safe_parse)
    return df

@st.cache_data(show_spinner=False)
def get_rag_country_stats(_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iso, name in COUNTRY_ISO.items():
        sub = _df[_df["_countries"].apply(lambda c: iso in [x.upper() for x in c])]
        if not len(sub): continue
        tb  = sub["_budget"].sum()
        pc  = len(sub)
        sev = CRISIS_SEVERITY.get(iso, 3)
        gap = round(max(0, sev - min(tb/1e8,10)), 2)
        rows.append({"iso":iso,"country":name,"projects":pc,"total_budget":tb,
                     "severity":sev,"gap_score":gap})
    return pd.DataFrame(rows).sort_values("gap_score", ascending=False)

@st.cache_resource
def get_groq():
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        return Groq(api_key=GROQ_API_KEY)
    except:
        return None

def ask_groq(gc, question: str, context: str) -> str:
    if not gc: return "⚠️ Groq unavailable — check API key."
    system = """You are Allocaid, a humanitarian policy intelligence assistant.
You analyze real IATI aid project data to surface funding gaps and policy opportunities.
Respond with concise bullet points. Be specific — cite countries, budgets, organizations, sector names.
Focus on: budget ratios, neglected crises, benchmark projects, actionable policy implications.
Use bold for key figures. Keep response under 250 words."""
    try:
        resp = gc.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":f"Context from dataset:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.35, max_tokens=600,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Groq error: {e}"

def build_rag_context(df: pd.DataFrame, stats: pd.DataFrame, prompt: dict) -> str:
    lines = [f"Dataset: {len(df):,} projects across {len(stats)} countries."]
    viz = prompt.get("viz","")
    if viz == "bbr":
        budgets = df[df["_budget"]>0]["_budget"]
        median  = budgets.median()
        q25, q75 = budgets.quantile(0.25), budgets.quantile(0.75)
        high = df[df["_budget"] > q75*4].nlargest(5,"_budget")
        low  = df[(df["_budget"]>0) & (df["_budget"]<q25*0.2)].nsmallest(5,"_budget")
        lines.append(f"Median project budget: ${median:,.0f}. Q25=${q25:,.0f}, Q75=${q75:,.0f}.")
        lines.append("High-budget outliers:")
        for _,r in high.iterrows():
            lines.append(f"  - {r['_title'][:60]} | ${r['_budget']:,.0f} | {safe_parse(r.get('recipient_country_narrative',''))[:40]}")
        lines.append("Low-budget projects:")
        for _,r in low.iterrows():
            lines.append(f"  - {r['_title'][:60]} | ${r['_budget']:,.0f} | {safe_parse(r.get('recipient_country_narrative',''))[:40]}")
    elif viz == "neglect":
        for _,r in stats.head(8).iterrows():
            lines.append(f"  {r['country']}: severity={r['severity']}/10, budget=${r['total_budget']/1e6:.1f}M, neglect={r['gap_score']}/10")
    elif viz == "benchmark":
        top = df.nlargest(8,"_budget")
        for _,r in top.iterrows():
            lines.append(f"  - {r['_title'][:60]} | ${r['_budget']:,.0f} | {r['_orgs'][:40]}")
    elif viz in ("treemap","sector_bar"):
        kw = prompt.get("sector_kw","")
        sub = df if not kw else df[df["_sectors"].apply(
            lambda s: any(kw.lower() in x.lower() for x in s if isinstance(x,str)))]
        all_secs = [s for row in sub["_sectors"] for s in row
                    if isinstance(s,str) and len(s)>3 and not s.replace(".","").isdigit()]
        top_secs = Counter(all_secs).most_common(10)
        lines.append(f"Top sectors{' ('+kw+')' if kw else ''}:")
        for sec,cnt in top_secs:
            lines.append(f"  - {sec}: {cnt} projects")
        lines.append(f"Total budget in sector: ${sub['_budget'].sum():,.0f}")
    elif viz == "org_bar":
        orgs = []
        for _,r in df.iterrows():
            for o in r["_orgs"].split(" | ") if r["_orgs"] else []:
                if o.strip(): orgs.append({"org":o.strip()[:50],"budget":r["_budget"]})
        if orgs:
            odf = pd.DataFrame(orgs).groupby("org").agg(projects=("budget","count"),
                                                          total=("budget","sum")).reset_index()
            odf["per_project"] = odf["total"]/odf["projects"]
            for _,r in odf.nlargest(6,"projects").iterrows():
                lines.append(f"  {r['org']}: {r['projects']} projects, ${r['total']/1e6:.1f}M total, ${r['per_project']:,.0f}/project")
    elif viz == "scatter":
        for _,r in stats.iterrows():
            lines.append(f"  {r['country']}: {r['projects']} projects, severity={r['severity']}, budget=${r['total_budget']/1e6:.1f}M")
    elif viz == "compare":
        for iso in prompt.get("countries",["SD","YE"]):
            name = COUNTRY_ISO.get(iso,iso)
            sub  = df[df["_countries"].apply(lambda c: iso in [x.upper() for x in c])]
            secs = Counter([s for r in sub["_sectors"] for s in r
                            if isinstance(s,str) and len(s)>3]).most_common(3)
            lines.append(f"{name}: {len(sub)} projects, ${sub['_budget'].sum()/1e6:.1f}M, "
                         f"severity={CRISIS_SEVERITY.get(iso,0)}/10, "
                         f"top sectors: {', '.join([s for s,_ in secs])}")
    elif viz == "heatmap":
        for _,r in stats.head(10).iterrows():
            sub = df[df["_countries"].apply(lambda c: r['iso'] in [x.upper() for x in c])]
            early = sub[sub["_years"].apply(lambda y: any(yr<2018 for yr in y))].shape[0]
            late  = sub[sub["_years"].apply(lambda y: any(yr>=2018 for yr in y))].shape[0]
            lines.append(f"  {r['country']}: pre-2018={early} projects, post-2018={late} projects")
    return "\n".join(lines)

# ── RAG visualizations ────────────────────────────────────────────────────────
DARK = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(249,250,251,0.6)",
            font=dict(color="#111827", family="sans-serif", size=11),
            margin=dict(l=0,r=0,t=20,b=0))
GRID = dict(gridcolor="#e5e7eb", linecolor="#e5e7eb")

def rag_fig_bbr(df):
    b = df[df["_budget"]>0].copy()
    median = b["_budget"].median()
    b["ratio"] = b["_budget"] / median
    b["status"] = b["ratio"].apply(lambda r: "High outlier" if r>4 else ("Low outlier" if r<0.15 else "Normal"))
    b["country"] = b["_countries"].apply(lambda c: COUNTRY_ISO.get(c[0].upper() if c else "","Unknown") if c else "Unknown")
    b["label"]   = b["_title"].str[:40]
    cmap = {"High outlier":"#f85149","Low outlier":"#3fb950","Normal":"#9ca3af"}
    fig = px.scatter(b.head(200), x="ratio", y="label", color="status",
        color_discrete_map=cmap, size_max=12,
        hover_data={"ratio":":.2f","status":True},
        labels={"ratio":"Budget / Median","label":"Project","status":""},
        title="Budget Ratio vs Median — Outlier Detection")
    fig.add_vline(x=1, line_dash="dash", line_color="#9ca3af", opacity=0.5)
    fig.add_vline(x=4, line_dash="dot",  line_color="#f85149", opacity=0.4)
    fig.add_vline(x=0.15, line_dash="dot", line_color="#3fb950", opacity=0.4)
    fig.update_layout(**DARK, height=360, xaxis=dict(**GRID,type="log"),
                      yaxis=dict(gridcolor="rgba(0,0,0,0)",tickfont=dict(size=9)),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def rag_fig_neglect(stats):
    df = stats.head(12).sort_values("gap_score", ascending=True)
    colors = ["#f85149" if g>=8 else "#d29922" if g>=5 else "#3fb950" for g in df["gap_score"]]
    fig = go.Figure(go.Bar(
        x=df["gap_score"], y=df["country"], orientation="h",
        marker_color=colors,
        customdata=np.stack([df["total_budget"]/1e6, df["projects"], df["severity"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>Neglect: %{x:.1f}<br>Budget: $%{customdata[0]:.1f}M<br>"
                      "Projects: %{customdata[1]}<br>Severity: %{customdata[2]}<extra></extra>",
    ))
    fig.update_layout(**DARK, title="Crisis Neglect Score (Severity − Funding Coverage)",
        xaxis=dict(**GRID, title="Neglect Score", range=[0,11]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"), height=360, showlegend=False)
    return fig

def rag_fig_benchmark(df):
    top = df.nlargest(15,"_budget").copy()
    top["budget_M"] = top["_budget"]/1e6
    top["label"]    = top["_title"].str[:45]
    top["org"]      = top["_orgs"].str[:35]
    top["country"]  = top["_countries"].apply(lambda c: COUNTRY_ISO.get(c[0].upper() if c else "","?") if c else "?")
    fig = px.bar(top, x="budget_M", y="label", orientation="h", color="country",
        hover_data={"org":True,"budget_M":":.1f"},
        labels={"budget_M":"Budget (M USD equiv.)","label":"Project","country":"Country"},
        title="Top Benchmark Projects by Budget",
        color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(**DARK, height=400,
        xaxis=dict(**GRID), yaxis=dict(gridcolor="rgba(0,0,0,0)",tickfont=dict(size=9)),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=9)))
    return fig

def rag_fig_treemap(df, kw=""):
    sub = df if not kw else df[df["_sectors"].apply(
        lambda s: any(kw.lower() in x.lower() for x in s if isinstance(x,str)))]
    rows = []
    for _,row in sub.iterrows():
        for s in row["_sectors"]:
            if isinstance(s,str) and len(s)>3 and not s.replace(".","").isdigit():
                rows.append({"sector":s[:45],"budget":row["_budget"]})
    if not rows: return None
    sec = pd.DataFrame(rows).groupby("sector")["budget"].agg(["sum","count"]).reset_index()
    sec.columns = ["sector","total_budget","count"]
    sec = sec[sec["total_budget"]>0].nlargest(14,"total_budget")
    fig = px.treemap(sec, path=["sector"], values="total_budget", color="count",
        color_continuous_scale=["#fff7ed","#e07b39"],
        hover_data={"count":True,"total_budget":":,.0f"},
        title=f"Sector Funding Distribution{' — '+kw.title() if kw else ''}")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#111827",family="sans-serif",size=11),
        margin=dict(l=0,r=0,t=30,b=0), height=360)
    fig.update_traces(marker=dict(line=dict(color="#ffffff",width=1.5)))
    return fig

def rag_fig_sector_bar(df, stats, kw):
    sub = df[df["_sectors"].apply(
        lambda s: any(kw.lower() in x.lower() for x in s if isinstance(x,str)))]
    rows = []
    for iso,name in COUNTRY_ISO.items():
        s2 = sub[sub["_countries"].apply(lambda c: iso in [x.upper() for x in c])]
        if not len(s2): continue
        sev = CRISIS_SEVERITY.get(iso,3)
        tb  = s2["_budget"].sum()
        rows.append({"country":name,"projects":len(s2),"budget_M":tb/1e6,"severity":sev,
                     "gap":max(0, sev - min(tb/5e7,10))})
    if not rows: return None
    sdf = pd.DataFrame(rows).sort_values("gap", ascending=False).head(14)
    fig = px.bar(sdf, x="country", y="budget_M", color="gap",
        color_continuous_scale=["#3fb950","#d29922","#f85149"],
        hover_data={"projects":True,"severity":True,"gap":":.1f"},
        labels={"budget_M":f"{kw.title()} Budget (M)","country":"","gap":"Gap Score"},
        title=f"{kw.title()} Sector Funding by Country — Sorted by Gap")
    fig.update_layout(**DARK, xaxis=dict(**GRID,tickangle=-35),
        yaxis=dict(**GRID), height=360,
        coloraxis_colorbar=dict(title="Gap",thickness=10,tickfont=dict(color="#6b7280")))
    return fig

def rag_fig_org_bar(df):
    orgs = []
    for _,r in df.iterrows():
        for o in (r["_orgs"] or "").split(" | "):
            if o.strip() and len(o.strip())>3:
                orgs.append({"org":o.strip()[:45],"budget":r["_budget"]})
    if not orgs: return None
    odf = pd.DataFrame(orgs).groupby("org").agg(
        projects=("budget","count"), total=("budget","sum")).reset_index()
    odf["per_project"] = (odf["total"]/odf["projects"]/1e6).round(2)
    top = odf.nlargest(12,"projects")
    fig = px.scatter(top, x="projects", y="total", size="per_project", color="per_project",
        hover_name="org", hover_data={"per_project":":.2f","projects":True},
        color_continuous_scale=["#dbeafe","#2563eb","#e07b39"],
        labels={"projects":"Number of Projects","total":"Total Budget","per_project":"M$/Project"},
        title="Organization Efficiency — Projects vs Total Budget")
    fig.update_layout(**DARK, xaxis=dict(**GRID), yaxis=dict(**GRID), height=360,
        coloraxis_colorbar=dict(title="M$/proj",thickness=10,tickfont=dict(color="#6b7280")))
    return fig

def rag_fig_scatter(stats):
    fig = px.scatter(stats, x="projects", y="severity", size="gap_score",
        color="gap_score", hover_name="country",
        color_continuous_scale=["#3fb950","#d29922","#f85149"],
        hover_data={"total_budget":":,.0f","gap_score":":.1f"},
        labels={"projects":"Projects","severity":"Crisis Severity","gap_score":"Neglect"},
        title="Emergency Response Coverage — Projects vs Crisis Severity",
        size_max=30)
    fig.update_layout(**DARK, xaxis=dict(**GRID), yaxis=dict(**GRID), height=360,
        coloraxis_colorbar=dict(title="Neglect",thickness=10,tickfont=dict(color="#6b7280")))
    return fig

def rag_fig_heatmap(df):
    years = list(range(2012,2025))
    rows  = []
    for iso,name in COUNTRY_ISO.items():
        sub = df[df["_countries"].apply(lambda c: iso in [x.upper() for x in c])]
        if not len(sub): continue
        rd = {"country":name}
        for yr in years:
            rd[str(yr)] = sub[sub["_years"].apply(lambda y: yr in y)].shape[0]
        rd["total"] = len(sub)
        rows.append(rd)
    if not rows: return None
    heat = pd.DataFrame(rows).sort_values("total",ascending=False).head(16)
    mat  = heat[["country"]+[str(y) for y in years]].set_index("country")
    fig  = go.Figure(data=go.Heatmap(
        z=mat.values, x=[str(y) for y in years], y=mat.index.tolist(),
        colorscale=[[0,"#fff7ed"],[0.4,"#e07b39"],[1,"#dc2626"]],
        colorbar=dict(title="Projects",tickfont=dict(color="#6b7280"),thickness=10),
    ))
    fig.update_layout(**DARK, title="Crisis Attention Over Time — Funding Fatigue",
        xaxis=dict(**GRID), yaxis=dict(gridcolor="rgba(0,0,0,0)",tickfont=dict(size=10)),
        height=380)
    return fig

def rag_fig_compare(df, stats, countries):
    rows = []
    for iso in countries:
        name = COUNTRY_ISO.get(iso,iso)
        sub  = df[df["_countries"].apply(lambda c: iso in [x.upper() for x in c])]
        secs = Counter([s for r in sub["_sectors"] for s in r
                        if isinstance(s,str) and len(s)>3]).most_common(5)
        for sec,cnt in secs:
            rows.append({"country":name,"sector":sec[:40],"projects":cnt,
                         "budget":sub[sub["_sectors"].apply(lambda s: sec in s)]["_budget"].sum()/1e6})
    if not rows: return None
    cdf = pd.DataFrame(rows)
    fig = px.bar(cdf, x="sector", y="budget", color="country", barmode="group",
        labels={"budget":"Budget (M)","sector":"Sector","country":""},
        title=f"Sector Funding Comparison: {' vs '.join([COUNTRY_ISO.get(c,c) for c in countries])}",
        color_discrete_sequence=["#58a6ff","#e07b39"])
    fig.update_layout(**DARK, xaxis=dict(**GRID,tickangle=-30),
        yaxis=dict(**GRID), height=360,
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def get_rag_viz(prompt, df, stats):
    viz = prompt.get("viz","none")
    kw  = prompt.get("sector_kw","")
    if viz=="bbr":         return rag_fig_bbr(df)
    if viz=="neglect":     return rag_fig_neglect(stats)
    if viz=="benchmark":   return rag_fig_benchmark(df)
    if viz=="treemap":     return rag_fig_treemap(df, kw)
    if viz=="sector_bar":  return rag_fig_sector_bar(df, stats, kw)
    if viz=="org_bar":     return rag_fig_org_bar(df)
    if viz=="scatter":     return rag_fig_scatter(stats)
    if viz=="heatmap":     return rag_fig_heatmap(df)
    if viz=="compare":     return rag_fig_compare(df, stats, prompt.get("countries",["SD","YE"]))
    return None

def format_chat_content(text: str, role: str) -> str:
    """Convert markdown-style AI responses to clean styled HTML."""
    if role == "user":
        return f'<span style="font-size:0.84rem;">{text}</span>'

    import html as html_lib
    lines = text.split("\n")
    result = []
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            result.append("</ul>")
            in_ul = False
        if in_ol:
            result.append("</ol>")
            in_ol = False

    def inline_fmt(s):
        # bold
        import re
        s = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#111827;">\1</strong>', s)
        # italic
        s = re.sub(r'\*(.+?)\*', r'<em>\1</em>', s)
        # backtick code
        s = re.sub(r'`(.+?)`', r'<code style="background:#f3f4f6;color:#1d4ed8;padding:1px 5px;border-radius:3px;font-size:0.82rem;">\1</code>', s)
        return s

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.lstrip()

        # blank line
        if not stripped:
            close_lists()
            continue

        # headings
        if stripped.startswith("### "):
            close_lists()
            result.append(f'<div style="font-size:0.82rem;font-weight:700;color:#e07b39;margin:0.8rem 0 0.3rem;">{inline_fmt(stripped[4:])}</div>')
        elif stripped.startswith("## "):
            close_lists()
            result.append(f'<div style="font-size:0.88rem;font-weight:700;color:#e07b39;margin:0.8rem 0 0.3rem;">{inline_fmt(stripped[3:])}</div>')
        elif stripped.startswith("# "):
            close_lists()
            result.append(f'<div style="font-size:0.95rem;font-weight:700;color:#e07b39;margin:0.8rem 0 0.4rem;">{inline_fmt(stripped[2:])}</div>')

        # unordered bullets: *, -, +
        elif stripped.startswith(("* ", "- ", "+ ")):
            if not in_ul:
                close_lists()
                result.append('<ul style="margin:0.4rem 0 0.4rem 1.1rem;padding:0;list-style:none;">')
                in_ul = True
            bullet_text = inline_fmt(stripped[2:])
            result.append(f'<li style="position:relative;padding-left:1rem;margin-bottom:0.25rem;font-size:0.83rem;line-height:1.6;color:#374151;">'
                           f'<span style="position:absolute;left:0;color:#e07b39;">›</span>{bullet_text}</li>')

        # numbered list
        elif len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)" :
            if not in_ol:
                close_lists()
                result.append('<ol style="margin:0.4rem 0 0.4rem 1.1rem;padding:0;list-style:none;counter-reset:item;">')
                in_ol = True
            num_end = stripped.index(stripped[1]) + 1
            ol_text = inline_fmt(stripped[num_end:].lstrip())
            num = stripped[:num_end-1]
            result.append(f'<li style="padding-left:1.5rem;margin-bottom:0.25rem;font-size:0.83rem;line-height:1.6;position:relative;color:#374151;">'
                           f'<span style="position:absolute;left:0;color:#e07b39;font-weight:600;">{num}.</span>{ol_text}</li>')

        # plain paragraph
        else:
            close_lists()
            result.append(f'<p style="margin:0.3rem 0;font-size:0.83rem;line-height:1.65;color:#374151;">{inline_fmt(stripped)}</p>')

    close_lists()
    return "\n".join(result)


# ── Tab 6 render ──────────────────────────────────────────────────────────────
with tab4:
    gc = get_groq()

    if not os.path.exists(RAG_CSV_PATH):
        st.warning(f"RAG dataset not found: `{RAG_CSV_PATH}`. Place `datanew.csv` in the app directory to enable Allocaid Intelligence.")
    else:
        with st.spinner("Loading intelligence dataset..."):
            rag_df    = load_rag_data(RAG_CSV_PATH)
            rag_stats = get_rag_country_stats(rag_df)

        # Init session state
        if "rag_messages"      not in st.session_state: st.session_state.rag_messages = []
        if "rag_active_prompt" not in st.session_state: st.session_state.rag_active_prompt = None

        # Header
        st.markdown("""
        <div style="padding:0.7rem 0 1rem 0; border-bottom:1px solid #e5e7eb; margin-bottom:1rem;">
            <span style="font-size:1.4rem; font-weight:700;">🤖 Allocaid Intelligence</span>
            <span style="font-size:0.75rem; color:#9ca3af; margin-left:1rem; text-transform:uppercase; letter-spacing:0.08em;">
                Humanitarian Policy RAG Assistant
            </span>
        </div>
        """, unsafe_allow_html=True)

        left_rag = st.container()

        # ── LEFT: chat ────────────────────────────────────────────────────────
        with left_rag:
            # Chat history
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Intelligence Assistant</div>', unsafe_allow_html=True)

            chat_html = '<div class="chat-scroll">'
            if not st.session_state.rag_messages:
                chat_html += """<div style="color:#9ca3af;font-size:0.83rem;padding:0.8rem 0;line-height:1.8;">
                    Select a prompt above to generate a policy analysis,<br>
                    or type your own question below.
                </div>"""
            else:
                for msg in st.session_state.rag_messages:
                    role_lbl = "You" if msg["role"]=="user" else "Allocaid"
                    cls = "msg-user" if msg["role"]=="user" else "msg-ai"
                    rendered = format_chat_content(msg["content"], msg["role"])
                    chat_html += f'<div class="msg {cls}"><div class="msg-role">{role_lbl}</div>{rendered}</div>'
            chat_html += '</div>'
            st.markdown(chat_html, unsafe_allow_html=True)

            inp_c, btn_c, clr_c = st.columns([6,1,1])
            with inp_c:
                user_q = st.text_input("q", placeholder="Ask a custom question...",
                                       label_visibility="collapsed", key="rag_chat_input")
            with btn_c:
                send = st.button("↑", use_container_width=True, key="rag_send")
            with clr_c:
                if st.button("✕", use_container_width=True, key="rag_clear"):
                    st.session_state.rag_messages      = []
                    st.session_state.rag_active_prompt = None
                    st.rerun()

            if send and user_q.strip():
                kw = user_q.lower()

                # ── Smart viz detection from free-text ──────────────────────
                iso2_map = {
                    "afghanistan":"AF","bangladesh":"BD","ethiopia":"ET","iraq":"IQ",
                    "mali":"ML","niger":"NE","pakistan":"PK","palestine":"PS","sudan":"SD",
                    "syria":"SY","chad":"TD","yemen":"YE","somalia":"SO","south sudan":"SS",
                    "dr congo":"CD","congo":"CD","central african republic":"CF","nigeria":"NG",
                    "myanmar":"MM","ukraine":"UA","lebanon":"LB","haiti":"HT",
                    "venezuela":"VE","mozambique":"MZ","burkina faso":"BF","burundi":"BI",
                    "cameroon":"CM","colombia":"CO","eritrea":"ER","libya":"LY","zimbabwe":"ZW",
                    "kenya":"KE","uganda":"UG","tanzania":"TZ","rwanda":"RW","south africa":"ZA",
                }
                mentioned_isos = []
                for country_name, iso in iso2_map.items():
                    if country_name in kw and iso not in mentioned_isos:
                        mentioned_isos.append(iso)
                # also catch bare ISO codes
                for iso in COUNTRY_ISO:
                    if iso.lower() in kw.split() and iso not in mentioned_isos:
                        mentioned_isos.append(iso)

                if len(mentioned_isos) >= 2:
                    p = {"viz": "compare", "countries": mentioned_isos[:2]}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif len(mentioned_isos) == 1:
                    p = {"viz": "scatter"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["budget", "outlier", "ratio"]):
                    p = {"viz": "bbr"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["neglect", "ignored", "gap", "severity", "worst"]):
                    p = {"viz": "neglect"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["food", "nutrition", "hunger"]):
                    p = {"viz": "sector_bar", "sector_kw": "food"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["health", "medical", "hospital"]):
                    p = {"viz": "sector_bar", "sector_kw": "health"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["protection", "gbv", "child"]):
                    p = {"viz": "sector_bar", "sector_kw": "protection"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["sector", "cluster"]):
                    p = {"viz": "treemap"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["org", "organization", "ngo", "efficient", "efficiency"]):
                    p = {"viz": "org_bar"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["time", "trend", "year", "decline", "fade", "attention"]):
                    p = {"viz": "heatmap"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["emergency", "response", "coverage"]):
                    p = {"viz": "scatter"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                elif any(w in kw for w in ["benchmark", "best", "top", "replicable"]):
                    p = {"viz": "benchmark"}
                    ctx = build_rag_context(rag_df, rag_stats, p)

                else:
                    mask = rag_df.apply(lambda r: any(kw in str(r.get(c,"")).lower() for c in TEXT_COLUMNS_RAG), axis=1)
                    hits = rag_df[mask].head(6)
                    ctx_lines = [f"Dataset: {len(rag_df):,} projects."]
                    for _,r in hits.iterrows():
                        ctx_lines.append(f"- {r['_title'][:60]} | ${r['_budget']:,.0f} | "
                                          f"{safe_parse(r.get('recipient_country_narrative',''))[:30]}")
                    ctx = "\n".join(ctx_lines)

                st.session_state.rag_messages.append({"role":"user","content":user_q})
                reply = ask_groq(gc, user_q, ctx)
                st.session_state.rag_messages.append({"role":"assistant","content":reply})
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)












# ============================================================
# TAB 5: DATA EXPLORER
# ============================================================


with tab5:

    # ── Header ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="comparison-header">
        <h2>🧪 Model Comparison — Full vs. Reduced Feature Set</h2>
        <p>
            How does including <strong>Prior Year CBPF</strong> (historical funding) change
            what the model considers "fair"? We trained two XGBoost variants and compare
            their predictions, feature importance, and which crises they flag differently.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model descriptions ────────────────────────────────────────────
    col_desc1, col_desc2 = st.columns(2)
    with col_desc1:
        st.markdown("""
        <div class="model-card">
            <div class="model-card-title">Full Feature Model</div>
            <span class="model-badge badge-full"> </span>
            <p style="margin-top:0.7rem;font-size:0.88rem;color:#374151;">
                Includes historical funding as a predictor. This model can learn
                "countries that got funded before tend to get funded again" —
                which means it may <strong>reinforce</strong> existing allocation patterns
                rather than questioning them.
            </p>
            <p style="font-size:0.82rem;color:#6b7280;margin-top:0.4rem;">
                ⚠️ Risk: anchors predictions to the status quo
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_desc2:
        st.markdown("""
        <div class="model-card">
            <div class="model-card-title">Reduced Feature Model</div>
            <span class="model-badge badge-reduced"> Includes Risk Management Metrics, general population data, and other NGO metrics for need. excludes Prior_Year_CBPF, Google Trends, and Logistics data</span>
            <p style="margin-top:0.7rem;font-size:0.88rem;color:#374151;">
                Predicts expected funding using <strong>only humanitarian need indicators</strong>.
                By removing historical funding, this model answers: "What <em>should</em>
                this country receive based purely on crisis severity?" —
                our <strong>primary fairness model</strong>.
            </p>
            <p style="font-size:0.82rem;color:#16a34a;margin-top:0.4rem;">
                ✅ Design choice: prevents funding-inertia bias
            </p>
        </div>
        """, unsafe_allow_html=True)
    reduced_model_df = pd.read_csv("FundingPredictionModels/outputs/scored_funding_fairness_reduced.csv")
    full_model_df = pd.read_csv("FundingPredictionModels/outputs/scored_funding_fairness_full.csv")


    # ── Check data availability ───────────────────────────────────────
    if full_model_df is None or reduced_model_df is None:
        st.warning(
            "Model comparison requires both "
            "`scored_funding_fairness_full.csv` and "
            "`scored_funding_fairness_reduced.csv` in `data/` or "
            "`FundingPredictionModels/outputs/`."
        )
    else:
        # ── Merge the two model outputs ───────────────────────────────
        merged = full_model_df.merge(
            reduced_model_df,
            on=["ISO3", "Year", "actual_funding"],
            suffixes=("_full", "_reduced"),
            how="inner",
        )
        merged["Country"] = merged["ISO3"].map(ISO_NAMES).fillna(merged["ISO3"])
        merged["pred_diff"] = merged["pred_funding_full"] - merged["pred_funding_reduced"]
        merged["pred_diff_pct"] = merged["pred_diff"] / (merged["pred_funding_reduced"] + 1e-9)
        merged["gap_diff"] = merged["funding_ratio_gap_full"] - merged["funding_ratio_gap_reduced"]
        merged["flag_change"] = (
            merged["flag_overlooked_full"].astype(bool) != merged["flag_overlooked_reduced"].astype(bool)
        )

        # ── Year filter for this tab ──────────────────────────────────
        mc_years = sorted(merged["Year"].unique(), reverse=True)
        mc_year = st.selectbox("Select year", mc_years, index=0, key="mc_year")
        mc = merged[merged["Year"] == mc_year].copy()

        # ── Summary metrics ───────────────────────────────────────────
        n_flag_change = mc["flag_change"].sum()
        mae_full = (mc["actual_funding"] - mc["pred_funding_full"]).abs().mean()
        mae_reduced = (mc["actual_funding"] - mc["pred_funding_reduced"]).abs().mean()
        corr_full = mc[["actual_funding", "pred_funding_full"]].corr().iloc[0, 1]
        corr_reduced = mc[["actual_funding", "pred_funding_reduced"]].corr().iloc[0, 1]
        avg_abs_gap_full = mc["funding_ratio_gap_full"].abs().mean()
        avg_abs_gap_reduced = mc["funding_ratio_gap_reduced"].abs().mean()

        m1, m2, m3= st.columns(3)
        with m1:
            winner = "Full" if mae_full < mae_reduced else "Reduced"
            st.markdown(f"""
            <div class="metric-card" style="border-color: #6366f1;">
                <h3>${min(mae_full, mae_reduced)/1e6:.1f}M</h3>
                <p>Lower MAE ({winner} model)</p>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card" style="border-color: #8b5cf6;">
                <h3>{max(corr_full, corr_reduced):.3f}</h3>
                <p>Best Correlation (actual vs pred)</p>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card" style="border-color: #10b981;">
                <h3>{len(mc)}</h3>
                <p>Country-Year Observations</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Insight box ───────────────────────────────────────────────
        if mae_full < mae_reduced:
            st.markdown(f"""
            <div class="insight-blue">
                📊 <strong>Accuracy Note:</strong> The Full model (with Prior_Year_CBPF, Google Trends, and International Logistics Metrics) achieves
                a lower MAE (${mae_full/1e6:.1f}M vs ${mae_reduced/1e6:.1f}M) in {mc_year} —
                but this is <strong>expected</strong>. Historical funding is a strong predictor of
                future funding, highlighting how funding may partly depend on inertia rather than need. The Reduced model intentionally
                trades accuracy for <strong>fairness signal</strong>.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="insight-blue">
                📊 <strong>Accuracy Note:</strong> The Reduced model actually matches or beats
                the Full model this year (MAE: ${mae_reduced/1e6:.1f}M vs ${mae_full/1e6:.1f}M).
                Need-based features alone capture the funding landscape well in {mc_year}.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Prediction Comparison Scatter ─────────────────────────────
        st.markdown("### Prediction Scatter — Full vs. Reduced")
        st.markdown(
            "Each dot is a country. If both models agreed perfectly, all points would "
            "lie on the diagonal. Deviations show where historical funding shifts the model's "
            "expectation away from a purely need-based prediction."
        )

        fig_scatter = go.Figure()

        # Perfect agreement line
        max_val = max(mc["pred_funding_full"].max(), mc["pred_funding_reduced"].max()) / 1e6
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val * 1.1], y=[0, max_val * 1.1],
            mode="lines",
            line=dict(color="#d1d5db", width=1.5, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Flag change countries highlighted
        fig_scatter.add_trace(go.Scatter(
            x=mc["pred_funding_full"] / 1e6,
            y=mc["pred_funding_reduced"] / 1e6,
            mode="markers",
            marker=dict(
                size=10,
                color="#94a3b8",
                opacity=0.7,
                line=dict(width=1, color="white"),
            ),
            text=mc["Country"],
            hovertemplate="<b>%{text}</b><br>Full: $%{x:.1f}M<br>Reduced: $%{y:.1f}M<extra></extra>",
            name="Country predictions",
        ))

        

        fig_scatter.update_layout(
            height=520,
            xaxis_title="Full Model Prediction ($M)",
            yaxis_title="Reduced Model Prediction ($M)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=30, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(249,250,251,0.6)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        

        # ── Accuracy Comparison Table ─────────────────────────────────
        st.markdown("### 📊 Model Performance Summary")

        perf_data = []
        for yr in sorted(merged["Year"].unique()):
            yr_df = merged[merged["Year"] == yr]
            mae_f = (yr_df["actual_funding"] - yr_df["pred_funding_full"]).abs().mean()
            mae_r = (yr_df["actual_funding"] - yr_df["pred_funding_reduced"]).abs().mean()
            corr_f = yr_df[["actual_funding", "pred_funding_full"]].corr().iloc[0, 1]
            corr_r = yr_df[["actual_funding", "pred_funding_reduced"]].corr().iloc[0, 1]
            n_over_f = yr_df["flag_overlooked_full"].astype(bool).sum()
            n_over_r = yr_df["flag_overlooked_reduced"].astype(bool).sum()
            n_changed = (yr_df["flag_overlooked_full"].astype(bool) != yr_df["flag_overlooked_reduced"].astype(bool)).sum()
            perf_data.append({
                "Year": yr,
                "MAE Full ($M)": f"${mae_f/1e6:.1f}M",
                "MAE Reduced ($M)": f"${mae_r/1e6:.1f}M",
                "Corr Full": f"{corr_f:.3f}",
                "Corr Reduced": f"{corr_r:.3f}",
                "Overlooked (Full)": int(n_over_f),
                "Overlooked (Reduced)": int(n_over_r),
                "Flag Changes": int(n_changed),
            })

        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── SHAP Feature Importance ───────────────────────────────────
        st.markdown("### 🧠 SHAP Feature Importance — What Drives Each Model?")
        st.markdown(
            "SHAP (SHapley Additive exPlanations) reveals how each feature pushes a "
            "prediction higher or lower. The key difference: in the Full model, "
            "**Prior_Year_CBPF dominates** — the model anchors heavily on past funding. "
            "The Reduced model must rely entirely on humanitarian indicators, producing "
            "a more **need-aligned** fairness baseline."
        )

        shap_col1, shap_col2 = st.columns(2)

        shap_full_path = "FundingPredictionModels/outputs/shap_summary_full.png"
        shap_reduced_path = "FundingPredictionModels/outputs/shap_summary_reduced.png"

        with shap_col1:
            st.markdown("""
            <div style="text-align:center;margin-bottom:0.5rem;">
                <span class="model-badge badge-full" style="font-size:0.85rem;">Full Model — SHAP Summary</span>
            </div>
            """, unsafe_allow_html=True)
            if os.path.exists(shap_full_path):
                st.image(shap_full_path, use_container_width=True)
            else:
                st.info("Place `shap_summary_full.png` in `data/` to display.")

        with shap_col2:
            st.markdown("""
            <div style="text-align:center;margin-bottom:0.5rem;">
                <span class="model-badge badge-reduced" style="font-size:0.85rem;">Reduced Model — SHAP Summary</span>
            </div>
            """, unsafe_allow_html=True)
            if os.path.exists(shap_reduced_path):
                st.image(shap_reduced_path, use_container_width=True)
            else:
                st.info("Place `shap_summary_reduced.png` in `data/` to display.")

        # ── SHAP Interpretation ───────────────────────────────────────
        st.markdown("""
        <div class="verdict-box">
            <strong>🔑 Key Takeaway from SHAP Analysis</strong>
            <p style="margin-top:0.5rem;font-size:0.9rem;color:#374151;">
                In the <span class="model-badge badge-full">Full Model</span>,
                <code>Prior_Year_CBPF</code> is the <strong>#1 feature</strong> by importance —
                the model essentially learns "fund what was funded before."
                This creates a self-reinforcing loop where historically well-funded crises
                continue to appear "fairly funded" even if need has shifted.
            </p>
            <p style="font-size:0.9rem;color:#374151;margin-top:0.5rem;">
                The <span class="model-badge badge-reduced">Reduced Model</span> redistributes
                that importance across <strong>Conflict Probability</strong>,
                <strong>Uprooted People</strong>, <strong>Governance</strong>, and
                <strong>Food Security</strong> — all genuine humanitarian severity indicators.
                This is why we use the Reduced model as our primary fairness baseline.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Detailed Comparison Table ─────────────────────────────────
        st.markdown("### 📋 Full Country-Level Comparison")

        detail = mc[["Country", "actual_funding",
                      "pred_funding_full", "pred_funding_reduced",
                      "funding_ratio_gap_full", "funding_ratio_gap_reduced",
                      "flag_overlooked_full", "flag_overlooked_reduced",
                      "flag_change"]].copy()
        detail = detail.sort_values("Country")
        detail.columns = [
            "Country", "Actual Funding",
            "Pred (Full)", "Pred (Reduced)",
            "Gap (Full)", "Gap (Reduced)",
            "Overlooked? (Full)", "Overlooked? (Reduced)",
            "Flag Changed?"
        ]
        detail["Actual Funding"] = detail["Actual Funding"].apply(lambda x: f"${x/1e6:.1f}M")
        detail["Pred (Full)"] = detail["Pred (Full)"].apply(lambda x: f"${x/1e6:.1f}M")
        detail["Pred (Reduced)"] = detail["Pred (Reduced)"].apply(lambda x: f"${x/1e6:.1f}M")
        detail["Gap (Full)"] = detail["Gap (Full)"].apply(lambda x: f"{x:.0%}")
        detail["Gap (Reduced)"] = detail["Gap (Reduced)"].apply(lambda x: f"{x:.0%}")
        detail["Overlooked? (Full)"] = detail["Overlooked? (Full)"].apply(lambda x: "🔴 Yes" if x else "—")
        detail["Overlooked? (Reduced)"] = detail["Overlooked? (Reduced)"].apply(lambda x: "🔴 Yes" if x else "—")
        detail["Flag Changed?"] = detail["Flag Changed?"].apply(lambda x: "⚠️ Yes" if x else "—")

        st.dataframe(detail, use_container_width=True, hide_index=True)

        # ── Design Philosophy ─────────────────────────────────────────
        with st.expander("💡 Why Two Models? — Design Philosophy"):
            st.markdown("""
**The Bias Mitigation Problem**

Most humanitarian funding prediction models include prior-year allocations as a feature.
This makes sense from a pure accuracy standpoint — past funding is one of the strongest
predictors of future funding. But it creates a circular problem:

> *"Countries that received a lot of funding before are predicted to receive a lot again.
> The model then says they're 'fairly funded' — because it expected the high amount.
> Meanwhile, historically neglected crises remain invisible."*

**Our Approach: Train Both, Compare, Use the Fairer One**

We train both variants and keep them for comparison:

- The **Full Model** tells us: *"Given how funding has worked historically, is this country
  getting what we'd expect?"* — useful for detecting sudden drops in donor attention.

- The **Reduced Model** tells us: *"Given only this country's humanitarian reality,
  is the funding proportionate to need?"* — the true fairness signal.

The **gap between the two models** is itself informative: it reveals where
**funding inertia** (historical momentum) is largest, and which crises would be
re-evaluated if we stripped away path dependence.

**Bottom line:** We use the Reduced model for the main dashboard precisely because
it refuses to let history define what's "normal."
            """)
















# TAB 6

with tab6:
    st.markdown("### Explore the Data")

    display_cols = ["Country", "ISO3", "Year", "actual_funding", "pred_funding",
                    "funding_ratio_gap", "flag_overlooked", "flag_overfunded",
                    "INFORM_Risk", "Vulnerability", "Need_Proxy", "Pop_Used",
                    "Continent", "CBPF_Reached", "CBPF_Targeted",
                    "beneficiaries_per_million"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    explorer = filtered[display_cols].copy()
    money_cols = ["actual_funding", "pred_funding"]
    for c in money_cols:
        if c in explorer.columns:
            explorer[c] = explorer[c].apply(lambda x: f"${x/1e6:.1f}M" if pd.notna(x) else "—")
    if "funding_ratio_gap" in explorer.columns:
        explorer["funding_ratio_gap"] = explorer["funding_ratio_gap"].apply(
            lambda x: f"{x:.0%}" if pd.notna(x) else "—"
        )
    if "Need_Proxy" in explorer.columns:
        explorer["Need_Proxy"] = explorer["Need_Proxy"].apply(
            lambda x: f"{x/1e6:.1f}M" if pd.notna(x) else "—"
        )
    if "Pop_Used" in explorer.columns:
        explorer["Pop_Used"] = explorer["Pop_Used"].apply(
            lambda x: f"{x/1e6:.1f}M" if pd.notna(x) else "—"
        )

    st.dataframe(explorer, use_container_width=True, hide_index=True)
    csv_data = filtered.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Dataset (CSV)",
        data=csv_data,
        file_name=f"allocaid_data_{selected_year}.csv",
        mime="text/csv"
    )


























# ============================================================
# TAB 7: RISK CLUSTERS — PEER INTELLIGENCE MAP
# ============================================================
# ── PASTE THIS BLOCK to replace your existing `with tab7:` section ──
#
# Requires:
#   - FundingPredictionModels/outputs/cluster_highlights.json
#   - data/MASTER_PANEL_FINAL.xlsx   (for risk radar — optional, degrades gracefully)
#
# Adds to your existing imports (already present in your app):
#   import json, plotly.graph_objects as go, plotly.express as px, numpy as np



# ============================================================
# METHODOLOGY
# ============================================================
with st.expander("📐 Methodology — How Allocaid Works"):
    st.markdown("""
**The Core Question:**  
Given a country's humanitarian indicators, how much CBPF funding is typically allocated in similar crisis contexts? Where does actual funding diverge from this needs-aligned baseline?

**Model:**  
XGBoost regression trained with walk-forward (year-forward) validation.  
Training window expands from early years (2018–2019) with predictions evaluated on subsequent years (2020 onward).  
Target variable is log-transformed Total CBPF allocation.

**Features (needs-focused indicators):**  
- INFORM Risk  
- Vulnerability  
- Conflict Probability  
- Food Security  
- Governance  
- Healthcare Access  
- Uprooted People  
- Hazard Exposure  
- GDP per capita  
- Population (with World Bank fallback)  
- Population density  
- Urban %  
- Latitude / Longitude  
- Year-over-year change in INFORM Risk  
- Year-over-year change in Vulnerability  

**Key Design Choice (Bias Mitigation):**  
We intentionally exclude prior-year CBPF funding as a feature. Including historical funding would allow the model to learn "countries that were funded before get funded again," reinforcing historical allocation patterns instead of surfacing potential misalignments.

**Funding Gap Metric:**  
`(Actual − Expected) / Expected`  
Negative values indicate underfunding relative to similar crisis profiles; positive values indicate comparatively higher funding.

**Peer Benchmarking:**  
For each country-year, we identify the most similar crisis contexts using standardized distance across numeric indicators, then compare funding gaps to contextualize relative under- or over-funding.

**Data Pipeline:**  
- 🥉 **Bronze:** Raw datasets ingested from HumData (HNO, HRP, INFORM, CBPF)  
- 🥈 **Silver:** Cleaning, typing, feature engineering (including Need_Proxy for missing PIN), data validation  
- 🥇 **Gold:** Trained XGBoost model, scored funding gaps, peer benchmarks  
- 📦 **Export:** Gold tables → CSV → Streamlit app
""")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <strong>Allocaid</strong> — Built with Databricks + Streamlit<br>
    Databricks × United Nations Geo-Insight Challenge<br>
    Data: UN OCHA HumData • CBPF Data Hub • INFORM Risk Index
</div>
""", unsafe_allow_html=True) 
# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <strong>Allocaid</strong> — Built with Databricks + Streamlit for Hacklytics 2026<br>
    Databricks × United Nations Geo-Insight Challenge<br>
    Data: UN OCHA HumData • CBPF Data Hub • INFORM Risk Index
</div>
""", unsafe_allow_html=True)