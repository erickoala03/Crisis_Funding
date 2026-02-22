# Databricks notebook source
# MAGIC %md
# MAGIC # 🔍 Notebook 04 — SHAP Explainability & Benchmarking
# MAGIC
# MAGIC **Hacklytics 2026 | Databricks × United Nations Challenge**
# MAGIC
# MAGIC This notebook:
# MAGIC - Generates SHAP explanations for model interpretability
# MAGIC - Builds a nearest-neighbor benchmarking engine
# MAGIC - Logs SHAP plots as MLflow artifacts

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install xgboost shap "numpy<2.0"

# COMMAND ----------

import numpy as np
import pandas as pd
import shap
import mlflow
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Gold Data & Reconstruct Model Pipeline

# COMMAND ----------

df = spark.table("main.default.gold_scored_funding").toPandas()
print(f"Loaded {len(df)} rows")

# ── Recreate feature set (must match training) ──
candidate_numeric = [
    "Need_Proxy", "Pop_Used",
    "Density_per_km2", "Land_Area_km2",
    "INFORM_Risk", "Vulnerability", "Conflict_Probability",
    "Food_Security", "Governance", "Access_Healthcare",
    "Uprooted_People", "Vulnerable_Groups", "Hazard_Exposure",
    "GDP_per_capita", "Urban_pct",
    "Latitude", "Longitude",
    "INFORM_Risk_Change", "Vulnerability_Change",
]
candidate_categorical = ["ISO3", "Continent"]

num_features = [c for c in candidate_numeric if c in df.columns]
cat_features = [c for c in candidate_categorical if c in df.columns]
X = df[num_features + cat_features]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Best Model from MLflow Registry

# COMMAND ----------

# # Load the registered model
# model_name = "crisis_funding_fairness"

# # Get latest version
# from mlflow.tracking import MlflowClient
# client = MlflowClient()
# latest = client.get_registered_model(model_name)
# latest_version = latest.latest_versions[0].version
# print(f"Loading model: {model_name} v{latest_version}")

# model_uri = f"models:/{model_name}/{latest_version}"
# loaded_pipe = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# Load best model directly from the MLflow run (skip registry)
import mlflow.sklearn

best_run_id = "<PASTE_RUN_ID_HERE>"
loaded_pipe = mlflow.sklearn.load_model("runs:/f9582cd23fed40e6ae7f424d6ef9d8e5/funding_fairness_model")
print("✅ Model loaded from MLflow run")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SHAP Analysis

# COMMAND ----------

# Transform features through the preprocessing pipeline
X_transformed = loaded_pipe.named_steps["preprocess"].transform(X)

# Get feature names after one-hot encoding
num_names = num_features
cat_encoder = loaded_pipe.named_steps["preprocess"].transformers_[1][1].named_steps["onehot"]
if hasattr(cat_encoder, "get_feature_names_out"):
    cat_names = list(cat_encoder.get_feature_names_out(cat_features))
else:
    cat_names = list(cat_encoder.get_feature_names(cat_features))

all_feature_names = num_names + cat_names

# COMMAND ----------

# Use the predict function wrapper instead of direct tree access
explainer = shap.Explainer(xgb_model.predict, X_dense, algorithm="permutation")
shap_values = explainer(X_dense)
print(f"✅ SHAP values computed")

# COMMAND ----------

# SHAP Summary Plot 
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(
    shap_values.values, 
    X_dense, 
    feature_names=all_feature_names,
    max_display=20,
    show=False
)
plt.title("SHAP Feature Importance — What Drives Funding Allocation?", fontsize=14)
plt.tight_layout()
plt.savefig("/tmp/shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# SHAP bar plot
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(
    shap_values.values, 
    X_dense, 
    feature_names=all_feature_names,
    max_display=20,
    plot_type="bar",
    show=False
)
plt.title("Mean |SHAP| — Average Feature Impact", fontsize=14)
plt.tight_layout()
plt.savefig("/tmp/shap_bar.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Nearest-Neighbor Benchmarking Engine
# MAGIC
# MAGIC For any country-year, find the 5 most similar crises by humanitarian indicators
# MAGIC and compare their actual vs. predicted funding.

# COMMAND ----------

# ── Build similarity index on numeric features ──
sim_features = [c for c in num_features if c in df.columns]
sim_df = df[sim_features].copy().replace([np.inf, -np.inf], np.nan)

sim_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

Z = sim_pipe.fit_transform(sim_df)

nn = NearestNeighbors(n_neighbors=6, metric="euclidean")
nn.fit(Z)

print(f"✅ Benchmarking index built on {len(sim_features)} features, {len(df)} rows")

# COMMAND ----------

def find_similar_crises(iso3, year, n=5):
    """Find the most similar crises to a given country-year."""
    mask = (df["ISO3"] == iso3) & (df["Year"] == year)
    if mask.sum() == 0:
        print(f"No data for {iso3} in {year}")
        return None
    
    idx = df.index[mask][0]
    row_pos = df.index.get_loc(idx)
    dists, neigh_pos = nn.kneighbors(Z[row_pos].reshape(1, -1))
    neigh_idx = df.index[neigh_pos[0]]
    neigh_idx = [i for i in neigh_idx if i != idx][:n]
    
    cols = ["ISO3", "Year", "actual_funding", "pred_funding", "funding_ratio_gap"]
    if "Continent" in df.columns:
        cols.append("Continent")
    cols += ["Need_Proxy", "INFORM_Risk"]
    cols = [c for c in cols if c in df.columns]
    
    target_row = df.loc[[idx], cols].copy()
    target_row.insert(0, "role", "TARGET")
    
    comparable = df.loc[neigh_idx, cols].copy()
    comparable.insert(0, "role", "comparable")
    
    result = pd.concat([target_row, comparable])
    return result

# COMMAND ----------

# ── Example: benchmark the most overlooked crisis ──
most_overlooked = df.loc[df["funding_ratio_gap"].idxmin()]
print(f"Most overlooked: {most_overlooked['ISO3']} in {most_overlooked['Year']}")
print(f"  Actual: ${most_overlooked['actual_funding']:,.0f}")
print(f"  Expected: ${most_overlooked['pred_funding']:,.0f}")
print(f"  Gap: {most_overlooked['funding_ratio_gap']:.1%}")

result = find_similar_crises(most_overlooked["ISO3"], int(most_overlooked["Year"]))
display(spark.createDataFrame(result))

# COMMAND ----------

# ── Build full benchmarking table for export ──
benchmark_records = []

for idx, row in df.iterrows():
    row_pos = df.index.get_loc(idx)
    dists, neigh_pos = nn.kneighbors(Z[row_pos].reshape(1, -1))
    neigh_idx = df.index[neigh_pos[0]]
    neigh_idx = [i for i in neigh_idx if i != idx][:5]
    
    neighbor_avg_ratio = df.loc[neigh_idx, "funding_ratio_gap"].mean()
    neighbor_avg_funding = df.loc[neigh_idx, "actual_funding"].mean()
    
    benchmark_records.append({
        "ISO3": row["ISO3"],
        "Year": row["Year"],
        "actual_funding": row["actual_funding"],
        "pred_funding": row["pred_funding"],
        "funding_ratio_gap": row["funding_ratio_gap"],
        "neighbor_avg_ratio_gap": neighbor_avg_ratio,
        "neighbor_avg_funding": neighbor_avg_funding,
        "relative_to_peers": row["funding_ratio_gap"] - neighbor_avg_ratio,
    })

benchmark_df = pd.DataFrame(benchmark_records)

# COMMAND ----------

# ── Save benchmarking table as Gold Delta ──
spark.createDataFrame(benchmark_df).write.format("delta").mode("overwrite").saveAsTable(
    "main.default.gold_benchmarking"
)
print(f"✅ Benchmarking table saved: main.default.gold_benchmarking ({len(benchmark_df)} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Top 10 Crises That Are Underfunded Relative to Peers

# COMMAND ----------

display(
    spark.sql("""
        SELECT ISO3, Year, 
               ROUND(actual_funding, 0) as actual_usd,
               ROUND(pred_funding, 0) as expected_usd,
               ROUND(funding_ratio_gap * 100, 1) as gap_pct,
               ROUND(relative_to_peers * 100, 1) as vs_peers_pct
        FROM main.default.gold_benchmarking
        WHERE pred_funding IS NOT NULL
        ORDER BY relative_to_peers ASC
        LIMIT 10
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ Explainability & Benchmarking complete
# MAGIC
# MAGIC - SHAP plots logged to MLflow
# MAGIC - Benchmarking table saved to `main.default.gold_benchmarking`
# MAGIC
# MAGIC **Next:** Run `05_export_to_streamlit` to export processed data for the web app.
