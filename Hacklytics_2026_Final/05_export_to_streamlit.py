# Databricks notebook source
# MAGIC %md
# MAGIC # 📦 Notebook 05 — Export to Streamlit
# MAGIC
# MAGIC **Hacklytics 2026 | Databricks × United Nations Challenge**
# MAGIC
# MAGIC Exports the Gold Delta tables as CSVs for the Streamlit web app.
# MAGIC Download these files and place them in your GitHub repo's `data/` folder.

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export 1: Scored Funding (Main Crisis Data)

# COMMAND ----------

scored = spark.table("main.default.gold_scored_funding").toPandas()

# Select columns needed by the Streamlit app
export_cols = [
    "ISO3", "Year", 
    "actual_funding", "pred_funding", 
    "funding_gap", "funding_ratio_gap",
    "flag_overlooked", "flag_overfunded",
    "Need_Proxy", "Pop_Used",
    "INFORM_Risk", "Vulnerability",
]

# Add optional columns if they exist
optional = ["Latitude", "Longitude", "Continent", "CBPF_Reached",
            "flag_low_efficiency", "flag_high_efficiency", 
            "beneficiaries_per_million", "efficiency_robust_z"]
for c in optional:
    if c in scored.columns:
        export_cols.append(c)

export_cols = [c for c in export_cols if c in scored.columns]
scored_export = scored[export_cols]

scored_export.to_csv("/tmp/scored_funding.csv", index=False)
print(f"✅ scored_funding.csv — {len(scored_export)} rows, {len(export_cols)} columns")
display(scored_export.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export 2: Benchmarking Data

# COMMAND ----------

bench = spark.table("main.default.gold_benchmarking").toPandas()
bench.to_csv("/tmp/benchmarking.csv", index=False)
print(f"✅ benchmarking.csv — {len(bench)} rows")
display(bench.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export 3: Summary Stats for Dashboard

# COMMAND ----------

# Aggregate by country (latest year)
latest_year = scored["Year"].max()
summary = (scored[scored["Year"] == latest_year]
    .groupby("ISO3")
    .agg(
        actual_funding=("actual_funding", "sum"),
        pred_funding=("pred_funding", "sum"),
        funding_gap=("funding_gap", "sum"),
        funding_ratio_gap=("funding_ratio_gap", "mean"),
        need_proxy=("Need_Proxy", "sum"),
        pop_used=("Pop_Used", "sum"),
    )
    .reset_index()
    .sort_values("funding_ratio_gap")
)

summary.to_csv("/tmp/country_summary.csv", index=False)
print(f"✅ country_summary.csv — {len(summary)} countries for year {latest_year}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Instructions
# MAGIC
# MAGIC **Option A: Copy to FileStore (download via browser)**
# MAGIC Uncomment and run the cell below, then download from the URLs.

# COMMAND ----------

# Write CSVs directly to Volumes (accessible for download)
scored = spark.table("main.default.gold_scored_funding").toPandas()
bench = spark.table("main.default.gold_benchmarking").toPandas()

scored.to_csv("/Volumes/main/default/raw_data/scored_funding.csv", index=False)
bench.to_csv("/Volumes/main/default/raw_data/benchmarking.csv", index=False)

print("✅ Files saved to Volume. Download from Catalog → main → default → Volumes → raw_data")

# COMMAND ----------

# MAGIC %md
# MAGIC **Option B: Display and copy** (for small datasets)
# MAGIC
# MAGIC Click the download icon (↓) on any displayed table above.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Summary
# MAGIC
# MAGIC ```
# MAGIC ┌───────────────────────────────────────────────────────────┐
# MAGIC │                    DATABRICKS PIPELINE                    │
# MAGIC │                                                           │
# MAGIC │  01_bronze  →  02_silver  →  03_gold_model  →  04_shap   │
# MAGIC │  (raw xlsx)    (cleaned)     (XGBoost+MLflow)  (explain)  │
# MAGIC │                                                     │     │
# MAGIC │                                              05_export    │
# MAGIC │                                                     │     │
# MAGIC └─────────────────────────────────────────────────────┼─────┘
# MAGIC                                                       │
# MAGIC                                                       ▼
# MAGIC                                              GitHub repo
# MAGIC                                              data/ folder
# MAGIC                                                       │
# MAGIC                                                       ▼
# MAGIC                                              Streamlit Cloud
# MAGIC                                              (public web app)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ Export complete!
# MAGIC
# MAGIC **Your deliverables:**
# MAGIC
# MAGIC | File | Purpose |
# MAGIC |------|---------|
# MAGIC | `scored_funding.csv` | Main data: actual vs predicted funding, flags |
# MAGIC | `benchmarking.csv` | Peer comparison data for each country-year |
# MAGIC | `country_summary.csv` | Aggregated view for map visualization |
# MAGIC
# MAGIC Place these in your Streamlit app's `data/` folder and push to GitHub.
