"""
visualize_combined.py
────────────────────────────────────────────────────────────────────────────────
Five charts from the combined IATI + Carlos crisis funding dataset.

  Chart 1 — Total funding by crisis type (horizontal bar)
  Chart 2 — Funding split by region per crisis type (stacked bar)
  Chart 3 — Project count heatmap: region × crisis type
  Chart 4 — Top 20 recipient countries by total budget
  Chart 5 — Project source comparison (IATI vs Carlos) by crisis type

Usage:
  python visualize_combined.py
────────────────────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

os.makedirs("charts", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/combined_projects.csv")
df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0)

# Normalise region labels
ORDERED_REGIONS = ["Africa", "Middle East", "Asia", "Latin America", "Europe", "Other", "Unknown"]
REGION_COLORS   = {
    "Africa":        "#E76F51",
    "Middle East":   "#2A9D8F",
    "Asia":          "#457B9D",
    "Latin America": "#E9C46A",
    "Europe":        "#6A4C93",
    "Other":         "#AAAAAA",
    "Unknown":       "#CCCCCC",
}

CRISIS_ORDER = [
    "Disease Outbreak", "Water/Sanitation", "Education Emergency",
    "Drought", "Climate/Environment", "Conflict/War", "Poverty/Economic",
    "Refugees", "Food Security", "Natural Disaster", "Mental Health", "Displacement",
]

# ── Palette ───────────────────────────────────────────────────────────────────
CRISIS_COLORS = [
    "#E63946","#457B9D","#2A9D8F","#E9C46A","#F4A261","#A8DADC",
    "#6A4C93","#48CAE4","#52B788","#F77F00","#D62828","#4CC9F0",
]

# ════════════════════════════════════════════════════════════════════════════════
# CHART 1 — Total funding by crisis type
# ════════════════════════════════════════════════════════════════════════════════
crisis_budget = (
    df.groupby("crisis_type")["budget"].sum()
    .reindex(CRISIS_ORDER)
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(12, 7))
y_pos = range(len(CRISIS_ORDER))
bars  = ax.barh(y_pos, crisis_budget.values / 1e6,
                color=CRISIS_COLORS, edgecolor="white", height=0.65)

for bar, val in zip(bars, crisis_budget.values):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
            f"${val/1e6:.0f}M", va="center", fontsize=9, color="#333333")

ax.set_yticks(y_pos)
ax.set_yticklabels(CRISIS_ORDER, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("Total Budget (USD Millions)", fontsize=11)
ax.set_title("Total Project Funding by Crisis Type\n(IATI + Carlos Dataset, 1,998 projects)",
             fontsize=14, fontweight="bold", pad=14)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, crisis_budget.max() / 1e6 * 1.18)
plt.tight_layout()
plt.savefig("charts/1_funding_by_crisis_type.png", dpi=150)
plt.close()
print("Saved: charts/1_funding_by_crisis_type.png")


# ════════════════════════════════════════════════════════════════════════════════
# CHART 2 — Funding by region per crisis type (stacked bar)
# ════════════════════════════════════════════════════════════════════════════════
pivot_fund = (
    df.groupby(["crisis_type", "region"])["budget"].sum()
    .unstack(fill_value=0)
    .reindex(index=CRISIS_ORDER, columns=ORDERED_REGIONS, fill_value=0)
) / 1e6  # → millions

fig, ax = plt.subplots(figsize=(14, 7))
bottom = np.zeros(len(CRISIS_ORDER))
x_pos  = np.arange(len(CRISIS_ORDER))

for region in ORDERED_REGIONS:
    vals = pivot_fund[region].values
    ax.bar(x_pos, vals, bottom=bottom,
           label=region, color=REGION_COLORS[region], edgecolor="white", width=0.7)
    bottom += vals

ax.set_xticks(x_pos)
ax.set_xticklabels(
    [c.replace("/", "/\n") for c in CRISIS_ORDER],
    fontsize=9, rotation=30, ha="right"
)
ax.set_ylabel("Total Budget (USD Millions)", fontsize=11)
ax.set_title("Humanitarian Funding by Crisis Type and Region\n(IATI + Carlos, stacked by region)",
             fontsize=14, fontweight="bold", pad=14)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}M"))
ax.legend(title="Region", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("charts/2_funding_by_region_and_crisis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: charts/2_funding_by_region_and_crisis.png")


# ════════════════════════════════════════════════════════════════════════════════
# CHART 3 — Project count heatmap: crisis type × region
# ════════════════════════════════════════════════════════════════════════════════
HEATMAP_REGIONS = ["Africa", "Asia", "Middle East", "Latin America", "Europe", "Other"]
pivot_count = (
    df.groupby(["crisis_type", "region"])["project_id"].count()
    .unstack(fill_value=0)
    .reindex(index=CRISIS_ORDER, columns=HEATMAP_REGIONS, fill_value=0)
)

fig, ax = plt.subplots(figsize=(11, 7))
im = ax.imshow(pivot_count.values, aspect="auto", cmap="YlOrRd")

ax.set_xticks(range(len(HEATMAP_REGIONS)))
ax.set_xticklabels(HEATMAP_REGIONS, fontsize=11)
ax.set_yticks(range(len(CRISIS_ORDER)))
ax.set_yticklabels(CRISIS_ORDER, fontsize=10)

for i in range(len(CRISIS_ORDER)):
    for j in range(len(HEATMAP_REGIONS)):
        val = pivot_count.values[i, j]
        color = "white" if val > pivot_count.values.max() * 0.55 else "#333333"
        ax.text(j, i, str(int(val)), ha="center", va="center",
                fontsize=9, fontweight="bold", color=color)

plt.colorbar(im, ax=ax, label="Number of Projects", shrink=0.7)
ax.set_title("Number of Projects by Crisis Type and Region\n(IATI + Carlos Dataset)",
             fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig("charts/3_project_count_heatmap.png", dpi=150)
plt.close()
print("Saved: charts/3_project_count_heatmap.png")


# ════════════════════════════════════════════════════════════════════════════════
# CHART 4 — Top 20 recipient countries by total budget
# ════════════════════════════════════════════════════════════════════════════════
country_budget = (
    df[df["country"] != "Unknown"]
    .groupby("country")["budget"].sum()
    .nlargest(20)
)

fig, ax = plt.subplots(figsize=(11, 8))
colors = plt.cm.tab20(np.linspace(0, 1, len(country_budget)))
ax.barh(range(len(country_budget)), country_budget.values / 1e6,
        color=colors, edgecolor="white", height=0.7)
ax.set_yticks(range(len(country_budget)))
ax.set_yticklabels(country_budget.index.tolist(), fontsize=10)
ax.invert_yaxis()

for i, val in enumerate(country_budget.values):
    ax.text(val / 1e6 + 5, i, f"${val/1e6:.0f}M", va="center", fontsize=9)

ax.set_xlabel("Total Budget (USD Millions)", fontsize=11)
ax.set_title("Top 20 Recipient Countries by Total Project Funding\n(IATI + Carlos Dataset)",
             fontsize=13, fontweight="bold", pad=14)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, country_budget.max() / 1e6 * 1.2)
plt.tight_layout()
plt.savefig("charts/4_top_countries_by_funding.png", dpi=150)
plt.close()
print("Saved: charts/4_top_countries_by_funding.png")


# ════════════════════════════════════════════════════════════════════════════════
# CHART 5 — IATI vs Carlos project count per crisis type
# ════════════════════════════════════════════════════════════════════════════════
pivot_source = (
    df.groupby(["crisis_type", "source"])["project_id"].count()
    .unstack(fill_value=0)
    .reindex(index=CRISIS_ORDER, fill_value=0)
)

x_pos = np.arange(len(CRISIS_ORDER))
w = 0.38
fig, ax = plt.subplots(figsize=(13, 6))

if "IATI" in pivot_source.columns:
    ax.bar(x_pos - w / 2, pivot_source["IATI"].values, w,
           label="IATI", color="#457B9D", edgecolor="white", alpha=0.9)
if "Carlos" in pivot_source.columns:
    ax.bar(x_pos + w / 2, pivot_source["Carlos"].values, w,
           label="Carlos Dataset", color="#E76F51", edgecolor="white", alpha=0.9)

ax.set_xticks(x_pos)
ax.set_xticklabels(
    [c.replace("/", "/\n") for c in CRISIS_ORDER],
    fontsize=9, rotation=30, ha="right"
)
ax.set_ylabel("Number of Projects", fontsize=11)
ax.set_title("Project Count by Data Source and Crisis Type\n(IATI API vs Carlos Supplementary Dataset)",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("charts/5_source_comparison.png", dpi=150)
plt.close()
print("Saved: charts/5_source_comparison.png")

print("\nAll 5 charts saved to charts/")
print(f"\nSummary:")
print(f"  Total projects : {len(df):,}")
print(f"  Total funding  : ${df['budget'].sum()/1e9:.2f}B")
print(f"  Countries      : {df[df['country'] != 'Unknown']['country'].nunique()}")
print(f"  Crisis types   : {df['crisis_type'].nunique()}")
print(f"  Regions covered: {', '.join(r for r in ORDERED_REGIONS if r not in ('Unknown','Other'))}")
