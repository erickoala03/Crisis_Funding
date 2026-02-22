# Databricks notebook source
# MAGIC %md
# MAGIC # 🥈 Notebook 02 — Silver Layer: Cleaning & Feature Engineering
# MAGIC
# MAGIC **Hacklytics 2026 | Databricks × United Nations Challenge**
# MAGIC
# MAGIC This notebook:
# MAGIC - Cleans and type-casts the bronze data
# MAGIC - Builds the `Need_Proxy` and `Pop_Used` features
# MAGIC - Drops leakage columns
# MAGIC - Applies data quality checks

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql.functions import col, when, lit, count, sum as spark_sum

# Load bronze data
df_bronze = spark.table("main.default.bronze_master_panel")
print(f"Bronze rows: {df_bronze.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert to Pandas for feature engineering
# MAGIC The feature engineering logic (Need_Proxy, Pop_Used) uses NumPy operations
# MAGIC that are cleaner in Pandas. We'll convert back to Spark for Delta storage.

# COMMAND ----------

df = df_bronze.toPandas()

# ── Drop leakage columns ──
drop_cols = ["CBPF_Active_Years", "Prior_Year_CBPF"]
drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)
print(f"Dropped leakage columns: {drop_cols}")

# COMMAND ----------

# ── Build Pop_Used: prefer Population, fall back to WB_Population ──
df["Pop_Used"] = pd.to_numeric(df["WB_Population"], errors="coerce")
mask_pop = df["Population"].notna()
df.loc[mask_pop, "Pop_Used"] = pd.to_numeric(
    df.loc[mask_pop, "Population"], errors="coerce"
)

print(f"Pop_Used coverage: {df['Pop_Used'].notna().sum()} / {len(df)}")

# COMMAND ----------

# ── Build Need_Proxy ──
# Priority: Total_PIN → fallback to Pop_Used * (risk + vulnerability)

def minmax_np(x):
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    return (x - mn) / (mx - mn + 1e-9)

pop_used = pd.to_numeric(df["WB_Population"], errors="coerce").to_numpy()
pop_col  = pd.to_numeric(df["Population"], errors="coerce").to_numpy()
pop_used = np.where(~np.isnan(pop_col), pop_col, pop_used)

risk = pd.to_numeric(df["INFORM_Risk"], errors="coerce").to_numpy()
vuln = pd.to_numeric(df["Vulnerability"], errors="coerce").to_numpy()
pin  = pd.to_numeric(df["Total_PIN"], errors="coerce").to_numpy()

fallback = pop_used * (0.6 * minmax_np(risk) + 0.4 * minmax_np(vuln))
need_proxy = np.where(np.isfinite(pin), pin, fallback)
need_proxy = np.where(np.isfinite(need_proxy), need_proxy, pop_used)

df["Need_Proxy"] = need_proxy

print(f"Need_Proxy coverage: {df['Need_Proxy'].notna().sum()} / {len(df)}")
print(f"\nCoverage by year:")
print(df.groupby("Year")["Need_Proxy"].apply(lambda s: s.notna().sum()))

# COMMAND ----------

# ── Cast Year to int ──
df["Year"] = df["Year"].astype(int)

# ── Build target ──
TARGET = "Total_CBPF"
df["_y"] = pd.to_numeric(df[TARGET], errors="coerce")

# Count before/after target filter
pre_count = len(df)
df = df[df["_y"].notna()].copy()
post_count = len(df)
print(f"\nTarget filter: {pre_count} → {post_count} rows ({pre_count - post_count} dropped)")

# ── Log-transform target (funding is heavy-tailed) ──
df["_y_model"] = np.log1p(df["_y"].clip(lower=0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Summary

# COMMAND ----------

# Data quality report
quality_report = pd.DataFrame({
    "column": df.columns,
    "non_null": df.notna().sum().values,
    "pct_filled": (df.notna().sum() / len(df) * 100).round(1).values,
    "dtype": df.dtypes.astype(str).values,
})
display(spark.createDataFrame(quality_report.sort_values("pct_filled")))

# COMMAND ----------

# ── Save as Silver Delta table ──
df_silver = spark.createDataFrame(df)
df_silver.write.format("delta").mode("overwrite").saveAsTable("main.default.silver_master_panel")

print(f"✅ Silver table saved: main.default.silver_master_panel")
print(f"   Rows: {len(df)}")
print(f"   Countries: {df['ISO3'].nunique()}")
print(f"   Years: {sorted(df['Year'].unique())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ Silver layer complete
# MAGIC Cleaned data with engineered features in `main.default.silver_master_panel`.
# MAGIC
# MAGIC **Next:** Run `03_gold_model_training` to train the XGBoost model with MLflow tracking.
