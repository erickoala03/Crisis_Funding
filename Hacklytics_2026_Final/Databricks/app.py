import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AidLens — Humanitarian Funding Fairness",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ISO3 → Country Name Mapping
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
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { font-size: 2.2rem; margin-bottom: 0.3rem; }
    .main-header p { font-size: 1.05rem; opacity: 0.85; margin-bottom: 0; }

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
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    scored = pd.read_csv("data/scored_funding.csv")
    bench = pd.read_csv("data/benchmarking.csv")
    
    # Add country names
    scored["Country"] = scored["ISO3"].map(ISO_NAMES).fillna(scored["ISO3"])
    bench["Country"] = bench["ISO3"].map(ISO_NAMES).fillna(bench["ISO3"])
    
    return scored, bench

scored_df, bench_df = load_data()

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🔎 AidLens</h1>
    <p>Databricks × United Nations — Where Does Humanitarian Funding Fall Short of Need?</p>
    <p style="font-size: 0.8rem; opacity: 0.6; margin-top: 0.5rem;">
        Hacklytics 2026 | ML pipeline in Databricks (XGBoost + MLflow) • Visualized with Streamlit
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    
    # Year filter
    available_years = sorted(scored_df["Year"].unique(), reverse=True)
    selected_year = st.selectbox("Year", available_years, index=0)
    
    # Continent filter
    continents = ["All"] + sorted(scored_df["Continent"].dropna().unique().tolist())
    selected_continent = st.selectbox("Region", continents)
    
    # Show only predicted rows
    show_predicted_only = st.checkbox("Only show rows with model predictions", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 How It Works")
    st.markdown(
        "AidLens uses an **XGBoost model** trained on humanitarian indicators "
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

# Also filter benchmarking
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

    # Key insight
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Global Map",
    "📊 Funding Gaps", 
    "🔍 Peer Benchmarking",
    "⚡ Efficiency",
    "📋 Data Explorer"
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
        # Cap extreme values for better color scale
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
        
        # Bubble map
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
    st.markdown("### Actual vs. Expected Funding")
    st.markdown(
        "Our XGBoost model predicts what each country **should** receive based on "
        "INFORM Risk, Vulnerability, Population, Conflict Probability, and other indicators. "
        "The difference reveals where funding allocation doesn't match humanitarian need."
    )
    
    gap_data = filtered[filtered["pred_funding"].notna()].copy()
    gap_data = gap_data.sort_values("funding_ratio_gap")
    
    if len(gap_data) > 0:
        # Dumbbell chart: actual vs expected
        fig_dumbbell = go.Figure()
        
        for _, row in gap_data.iterrows():
            color = "#e74c3c" if row["funding_ratio_gap"] < -0.35 else (
                "#3b82f6" if row["funding_ratio_gap"] > 0.35 else "#94a3b8"
            )
            fig_dumbbell.add_trace(go.Scatter(
                x=[row["actual_funding"] / 1e6, row["pred_funding"] / 1e6],
                y=[row["Country"], row["Country"]],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=10),
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['Country']}</b><br>"
                    f"Actual: ${row['actual_funding']/1e6:.1f}M<br>"
                    f"Expected: ${row['pred_funding']/1e6:.1f}M<br>"
                    f"Gap: {row['funding_ratio_gap']:.0%}<extra></extra>"
                )
            ))
        
        # Legend markers
        fig_dumbbell.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(size=8, color="#e74c3c"), name="Overlooked (≥35% below)"))
        fig_dumbbell.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(size=8, color="#3b82f6"), name="Overfunded (≥35% above)"))
        fig_dumbbell.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(size=8, color="#94a3b8"), name="Within range"))
        
        fig_dumbbell.update_layout(
            height=max(400, len(gap_data) * 35),
            margin=dict(l=0, t=10, b=40),
            xaxis_title="CBPF Funding ($ Millions)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_dumbbell, use_container_width=True)
        
        st.markdown("*Left dot = actual funding, Right dot = model-expected funding. "
                     "Red lines = crises receiving ≥35% less than expected.*")
        
        # Bar chart: funding ratio gap
        st.markdown("### Funding Ratio Gap by Country")
        
        fig_gap_bar = px.bar(
            gap_data,
            x="funding_ratio_gap",
            y="Country",
            orientation="h",
            color="funding_ratio_gap",
            color_continuous_scale="RdBu",
            range_color=[-1, 1],
            color_continuous_midpoint=0,
            labels={"funding_ratio_gap": "Funding Ratio Gap", "Country": ""},
            hover_data={"actual_funding": ":.0f", "pred_funding": ":.0f"}
        )
        fig_gap_bar.update_layout(
            height=max(400, len(gap_data) * 30),
            margin=dict(l=0, t=10),
            xaxis_tickformat=".0%",
            xaxis_title="← Underfunded | Overfunded →",
            coloraxis_colorbar=dict(tickformat=".0%")
        )
        fig_gap_bar.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
        st.plotly_chart(fig_gap_bar, use_container_width=True)
        
        # Time series
        st.markdown("### Funding Gap Over Time")
        countries_list = sorted(filtered["Country"].unique())
        selected_ts_country = st.selectbox("Select country", countries_list, key="ts_country")
        
        ts_data = scored_df[
            (scored_df["Country"] == selected_ts_country) & 
            (scored_df["pred_funding"].notna())
        ].sort_values("Year")
        
        if len(ts_data) > 0:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=ts_data["Year"], y=ts_data["actual_funding"] / 1e6,
                name="Actual CBPF", mode="lines+markers",
                line=dict(color="#e74c3c", width=3),
                marker=dict(size=8)
            ))
            fig_ts.add_trace(go.Scatter(
                x=ts_data["Year"], y=ts_data["pred_funding"] / 1e6,
                name="Model Expected", mode="lines+markers",
                line=dict(color="#3b82f6", width=3, dash="dash"),
                marker=dict(size=8)
            ))
            fig_ts.update_layout(
                height=400,
                yaxis_title="Funding ($ Millions)",
                xaxis_title="Year",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=30)
            )
            st.plotly_chart(fig_ts, use_container_width=True)

# ============================================================
# TAB 3: PEER BENCHMARKING
# ============================================================
with tab3:
    st.markdown("### Peer Comparison — Is This Crisis an Outlier?")
    st.markdown(
        "For each country, our model finds the **5 most similar crises** (by humanitarian "
        "indicators like INFORM Risk, Vulnerability, Population, GDP, etc.) using nearest-neighbor "
        "matching. We then compare: is this country's funding gap unusual relative to its peers?"
    )
    
    bench_valid = bench_filtered[bench_filtered["pred_funding"].notna()].copy()
    
    if len(bench_valid) > 0:
        bench_valid["Country"] = bench_valid["ISO3"].map(ISO_NAMES).fillna(bench_valid["ISO3"])
        bench_valid = bench_valid.sort_values("relative_to_peers")
        
        fig_peers = px.bar(
            bench_valid,
            x="relative_to_peers",
            y="Country",
            orientation="h",
            color="relative_to_peers",
            color_continuous_scale="RdBu",
            range_color=[-1, 1],
            color_continuous_midpoint=0,
            labels={"relative_to_peers": "vs Peers", "Country": ""},
            hover_data={
                "funding_ratio_gap": ":.1%",
                "neighbor_avg_ratio_gap": ":.1%",
                "relative_to_peers": ":.1%",
            }
        )
        fig_peers.update_layout(
            height=max(400, len(bench_valid) * 30),
            margin=dict(l=0, t=10),
            xaxis_title="← Worse than peers | Better than peers →",
            xaxis_tickformat=".0%",
            coloraxis_colorbar=dict(title="vs Peers", tickformat=".0%"),
        )
        fig_peers.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
        st.plotly_chart(fig_peers, use_container_width=True)
        
        # Detail table
        st.markdown("#### Detailed Peer Comparison")
        display_bench = bench_valid[["Country", "actual_funding", "pred_funding", 
                                      "funding_ratio_gap", "neighbor_avg_ratio_gap", 
                                      "neighbor_avg_funding", "relative_to_peers"]].copy()
        display_bench.columns = ["Country", "Actual ($)", "Expected ($)", 
                                  "Own Gap", "Peers Avg Gap", "Peers Avg Funding ($)", "vs Peers"]
        
        display_bench["Actual ($)"] = display_bench["Actual ($)"].apply(lambda x: f"${x/1e6:.1f}M")
        display_bench["Expected ($)"] = display_bench["Expected ($)"].apply(lambda x: f"${x/1e6:.1f}M")
        display_bench["Own Gap"] = display_bench["Own Gap"].apply(lambda x: f"{x:.0%}")
        display_bench["Peers Avg Gap"] = display_bench["Peers Avg Gap"].apply(lambda x: f"{x:.0%}")
        display_bench["Peers Avg Funding ($)"] = display_bench["Peers Avg Funding ($)"].apply(lambda x: f"${x/1e6:.1f}M")
        display_bench["vs Peers"] = display_bench["vs Peers"].apply(lambda x: f"{x:+.0%}")
        
        st.dataframe(display_bench, use_container_width=True, hide_index=True)
    else:
        st.info("No benchmarking data available for the selected filters.")

# ============================================================
# TAB 4: EFFICIENCY
# ============================================================
with tab4:
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
# TAB 5: DATA EXPLORER
# ============================================================
with tab5:
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
            explorer[c] = explorer[c].apply(
                lambda x: f"${x/1e6:.1f}M" if pd.notna(x) else "—"
            )
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
        file_name=f"aidlens_data_{selected_year}.csv",
        mime="text/csv"
    )

# ============================================================
# METHODOLOGY
# ============================================================
with st.expander("📐 Methodology — How AidLens Works"):
    st.markdown("""
    **The Core Question:** Given a country's humanitarian indicators, how much CBPF funding 
    *should* it receive? Where does actual funding deviate from what the data suggests is fair?
    
    **Model:** XGBoost regression trained with walk-forward time-series validation (2018-2019 
    as initial training window, predictions from 2020 onward). Target variable is log-transformed 
    Total CBPF allocation.
    
    **Features (19 indicators):** INFORM Risk, Vulnerability, Conflict Probability, Food Security, 
    Governance, Healthcare Access, Uprooted People, Hazard Exposure, GDP per capita, Population, 
    Population Density, Urban %, Latitude/Longitude, and year-over-year changes in risk and vulnerability.
    
    **Key Design Choice:** We intentionally exclude prior-year CBPF funding as a feature. Including 
    it would let the model learn "countries that got funded before get funded again" — which bakes 
    in historical bias rather than surfacing it.
    
    **Fairness Gap:** `(Actual / Expected) - 1`. A value of -0.5 means the country received 50% 
    less than the model predicts for crises with similar humanitarian indicators.
    
    **Peer Benchmarking:** For each country-year, we find the 5 most similar observations using 
    standardized Euclidean distance across all numeric features, then compare funding gaps.
    
    **Pipeline:**
    - 🥉 **Bronze:** Raw data ingested from HumData into Delta Lake
    - 🥈 **Silver:** Cleaned, typed, Need_Proxy engineered, data quality validated
    - 🥇 **Gold:** XGBoost model trained with MLflow experiment tracking (3 hyperparameter configs), 
      SHAP explainability, nearest-neighbor benchmarking
    - 📦 **Export:** Gold tables → CSV → Streamlit Cloud
    
    All model experiments tracked in **MLflow** on Databricks. Best model selected by walk-forward R².
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <strong>AidLens</strong> — Built with Databricks + Streamlit for Hacklytics 2026<br>
    Databricks × United Nations Geo-Insight Challenge<br>
    Data: UN OCHA HumData • CBPF Data Hub • INFORM Risk Index
</div>
""", unsafe_allow_html=True)
