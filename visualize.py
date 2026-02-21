# visualize.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

df = pd.read_csv("data/combined_projects.csv")
df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0)

# --- Chart 1: Total funding by crisis type ---
crisis_budget = df.groupby("crisis_type")["budget"].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
crisis_budget.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Total Project Funding by Crisis Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Crisis Type")
ax.set_ylabel("Total Budget (USD)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.0f}M"))
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("charts/funding_by_crisis.png", dpi=150)
plt.show()

# --- Chart 2: Funding by region across crisis types (grouped bar) ---
REGION_MAP = {
    "289": "Africa", "380": "America", "298": "Asia",
    "89": "Europe", "589": "Middle East", "619": "Oceania"
}
df["region_name"] = df["region"].astype(str).map(REGION_MAP).fillna("Other")

pivot = df.groupby(["crisis_type", "region_name"])["budget"].sum().unstack(fill_value=0)
pivot.plot(kind="bar", figsize=(14, 7), edgecolor="white")
plt.title("Project Funding by Crisis Type and Region", fontsize=14, fontweight="bold")
plt.xlabel("Crisis Type")
plt.ylabel("Total Budget (USD)")
plt.xticks(rotation=30, ha="right")
plt.legend(title="Region", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.savefig("charts/funding_by_region_crisis.png", dpi=150)
plt.show()

# --- Chart 3: Number of projects per country (top 20) ---
top_countries = df.groupby("country")["project_id"].count().nlargest(20)
top_countries.plot(kind="barh", figsize=(10, 7), color="tomato")
plt.title("Top 20 Countries by Number of Projects", fontsize=14)
plt.xlabel("Number of Projects")
plt.tight_layout()
plt.savefig("charts/projects_by_country.png", dpi=150)
plt.show()