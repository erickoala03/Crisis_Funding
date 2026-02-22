"""
visualize_gaps.py
────────────────────────────────────────────────────────────────────────────────
Produce 4 charts showing humanitarian funding gaps across crisis topics/regions.

  Chart 1 — Benchmark % vs actual funding mention % per topic
  Chart 2 — Underrepresentation ratio (how many times underfunded)
  Chart 3 — Funding in matched flows by crisis topic × region
  Chart 4 — Top countries with matched flows per topic

Data sources:
  - topic_scan_results.json  (existing; used for charts 1-2)
  - FTS API sample fetch     (5 pages × 2 years ≈ 1 000 flows; used for charts 3-4)

Usage:
  python visualize_gaps.py
────────────────────────────────────────────────────────────────────────────────
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from topic_scanner import TOPICS, fetch_fts_flows

os.makedirs("charts", exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
TOPIC_COLORS = {
    "gbv":                      "#E63946",
    "mental_health":            "#457B9D",
    "education_in_emergencies": "#2A9D8F",
    "disability":               "#E9C46A",
    "elderly":                  "#A8DADC",
}
REGION_COLORS = {
    "Africa":        "#E76F51",
    "Middle East":   "#2A9D8F",
    "Asia":          "#457B9D",
    "Latin America": "#E9C46A",
    "Europe":        "#6A4C93",
    "Other":         "#CCCCCC",
}
SHORT = {
    "gbv":                      "GBV",
    "mental_health":            "Mental Health",
    "education_in_emergencies": "Education",
    "disability":               "Disability",
    "elderly":                  "Elderly",
}
BENCHMARKS_PCT = {
    "gbv": 33,
    "mental_health": 25,
    "education_in_emergencies": 20,
    "disability": 15,
    "elderly": 12,
}

# Country name → UN region (covers the main humanitarian recipient countries)
COUNTRY_REGIONS = {
    # Africa
    "South Sudan": "Africa", "Sudan": "Africa", "Ethiopia": "Africa",
    "Somalia": "Africa", "DRC": "Africa",
    "Democratic Republic of the Congo": "Africa",
    "Nigeria": "Africa", "Mali": "Africa", "Niger": "Africa",
    "Chad": "Africa", "Cameroon": "Africa",
    "Central African Republic": "Africa", "Mozambique": "Africa",
    "Zimbabwe": "Africa", "Kenya": "Africa", "Uganda": "Africa",
    "Burkina Faso": "Africa", "Libya": "Africa", "Malawi": "Africa",
    "Madagascar": "Africa", "Tanzania": "Africa", "Rwanda": "Africa",
    "Burundi": "Africa", "Eritrea": "Africa", "Guinea": "Africa",
    "Sierra Leone": "Africa", "Liberia": "Africa",
    # Middle East
    "Syria": "Middle East", "Yemen": "Middle East", "Iraq": "Middle East",
    "Palestine": "Middle East", "Lebanon": "Middle East",
    "Jordan": "Middle East", "Iran": "Middle East",
    # Asia (incl. South/Central Asia)
    "Afghanistan": "Asia", "Bangladesh": "Asia", "Myanmar": "Asia",
    "Pakistan": "Asia", "Philippines": "Asia", "Indonesia": "Asia",
    "Nepal": "Asia", "India": "Asia", "Sri Lanka": "Asia",
    "Cambodia": "Asia", "Laos": "Asia", "Thailand": "Asia",
    "Timor-Leste": "Asia", "Papua New Guinea": "Asia",
    "Kazakhstan": "Asia", "Kyrgyzstan": "Asia", "Tajikistan": "Asia",
    # Latin America
    "Colombia": "Latin America", "Venezuela": "Latin America",
    "Haiti": "Latin America", "Honduras": "Latin America",
    "Guatemala": "Latin America", "Ecuador": "Latin America",
    "Peru": "Latin America", "Bolivia": "Latin America",
    "El Salvador": "Latin America", "Nicaragua": "Latin America",
    # Europe
    "Ukraine": "Europe", "Türkiye": "Europe", "Turkey": "Europe",
    "Georgia": "Europe", "Serbia": "Europe", "Kosovo": "Europe",
    "Bosnia and Herzegovina": "Europe",
}

# ── Load existing scan results ────────────────────────────────────────────────
with open("topic_scan_results.json") as f:
    scan = json.load(f)

topic_keys = list(BENCHMARKS_PCT.keys())


# ════════════════════════════════════════════════════════════════════════════════
# CHART 1 — Benchmark % vs actual mention %
# ════════════════════════════════════════════════════════════════════════════════
bench_vals  = [BENCHMARKS_PCT[k] for k in topic_keys]
actual_vals = [scan["topics"][k]["pct_high"] for k in topic_keys]
short_labels = [SHORT[k] for k in topic_keys]

x = np.arange(len(topic_keys))
w = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(x - w / 2, bench_vals, w,
            label="Population affected (benchmark %)",
            color="#2C3E50", alpha=0.85)
b2 = ax.bar(x + w / 2, actual_vals, w,
            label="% of funding flows mentioning topic",
            color=[TOPIC_COLORS[k] for k in topic_keys], alpha=0.9)

for bar in b1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{bar.get_height():.0f}%",
            ha="center", va="bottom", fontsize=9, color="#2C3E50", fontweight="bold")
for bar in b2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%",
            ha="center", va="bottom", fontsize=9)

ax.set_title("Funding Representation vs Population Need",
             fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Percentage (%)")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=11)
ax.set_ylim(0, 42)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("charts/gap_analysis.png", dpi=150)
plt.close()
print("Saved: charts/gap_analysis.png")


# ════════════════════════════════════════════════════════════════════════════════
# CHART 2 — Underrepresentation ratio
# ════════════════════════════════════════════════════════════════════════════════
ratios = [BENCHMARKS_PCT[k] / max(scan["topics"][k]["pct_high"], 0.01)
          for k in topic_keys]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(short_labels, ratios,
              color=[TOPIC_COLORS[k] for k in topic_keys],
              edgecolor="white", width=0.55)

for bar, ratio in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{ratio:.0f}×",
            ha="center", va="bottom", fontsize=13, fontweight="bold")

ax.axhline(1, color="red", linestyle="--", linewidth=1.2,
           label="Fully represented (1×)")
ax.set_title("How Many Times Underrepresented in Funding?",
             fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Ratio: population benchmark % ÷ actual funding mention %")
ax.set_ylim(0, max(ratios) * 1.25)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("charts/underrepresentation_ratio.png", dpi=150)
plt.close()
print("Saved: charts/underrepresentation_ratio.png")


# ════════════════════════════════════════════════════════════════════════════════
# FETCH FTS SAMPLE FOR REGIONAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
print("\nFetching FTS sample (5 pages × 2 years) for regional breakdown...")
sample_flows = []
for year in [2023, 2024]:
    sample_flows.extend(fetch_fts_flows(year=year, max_pages=5, per_page=100))
print(f"Total sample flows: {len(sample_flows)}\n")

ALL_REGIONS = ["Africa", "Middle East", "Asia", "Latin America", "Europe", "Other"]

region_usd    = {k: defaultdict(float) for k in topic_keys}
country_count = {k: defaultdict(int)   for k in topic_keys}

for flow in sample_flows:
    desc   = (flow.get("description") or "").lower()
    amount = flow.get("amountUSD") or flow.get("fullParkedAmountUSD") or 0

    country = "Unknown"
    for obj in flow.get("destinationObjects", []):
        if obj.get("type") == "Location":
            country = obj.get("name", "Unknown")
            break

    region = COUNTRY_REGIONS.get(country, "Other")

    for tk, topic in TOPICS.items():
        if any(t.lower() in desc for t in topic["high_confidence"]):
            region_usd[tk][region]     += amount
            country_count[tk][country] += 1


# ════════════════════════════════════════════════════════════════════════════════
# CHART 3 — Funding by crisis topic × region (grouped bar)
# ════════════════════════════════════════════════════════════════════════════════
n_regions = len(ALL_REGIONS)
bar_w     = 0.80 / n_regions
offsets   = np.linspace(-(n_regions - 1) / 2 * bar_w,
                         (n_regions - 1) / 2 * bar_w, n_regions)
x = np.arange(len(topic_keys))

fig, ax = plt.subplots(figsize=(14, 6))
for j, (reg, offset) in enumerate(zip(ALL_REGIONS, offsets)):
    vals = [region_usd[tk].get(reg, 0) / 1e6 for tk in topic_keys]
    ax.bar(x + offset, vals, bar_w,
           label=reg, color=REGION_COLORS[reg], edgecolor="white", alpha=0.9)

ax.set_title(
    "Funding in Matched Flows by Crisis Topic and Region\n"
    "(FTS 2023–2024, ~1 000-flow sample)",
    fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("USD (millions)")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=11)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"${v:.1f}M"))
ax.legend(title="Region", bbox_to_anchor=(1.01, 1),
          loc="upper left", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("charts/regional_breakdown.png", dpi=150)
plt.close()
print("Saved: charts/regional_breakdown.png")


# ════════════════════════════════════════════════════════════════════════════════
# CHART 4 — Top countries per topic
# ════════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, len(topic_keys), figsize=(18, 5))

for ax, tk in zip(axes, topic_keys):
    cc = {c: n for c, n in country_count[tk].items() if c != "Unknown"}
    if not cc:
        ax.text(0.5, 0.5, "No matches", ha="center", va="center",
                transform=ax.transAxes, color="grey", fontsize=10)
        ax.set_title(SHORT[tk], fontsize=11, fontweight="bold",
                     color=TOPIC_COLORS[tk])
        ax.set_axis_off()
        continue

    top = sorted(cc.items(), key=lambda item: item[1], reverse=True)[:7]
    ctries, counts = zip(*top)
    y_pos = range(len(ctries))

    ax.barh(y_pos, counts, color=TOPIC_COLORS[tk], edgecolor="white", alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ctries, fontsize=9)
    ax.set_title(SHORT[tk], fontsize=11, fontweight="bold",
                 color=TOPIC_COLORS[tk])
    ax.set_xlabel("Matched flows", fontsize=8)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    "Top Countries with Matched Flows per Crisis Topic\n(FTS 2023–2024 sample)",
    fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/top_countries_per_topic.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: charts/top_countries_per_topic.png")

print("\nDone. All 4 charts saved to charts/")
