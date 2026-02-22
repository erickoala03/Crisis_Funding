# Databricks notebook source
# MAGIC %md
# MAGIC # 🥉 Notebook 01 — Bronze Layer: Raw Data Ingestion
# MAGIC
# MAGIC **Hacklytics 2026 | Databricks × United Nations Challenge**
# MAGIC
# MAGIC This notebook loads the raw master panel data into a Delta table.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upload Instructions (Can change based on data used)
# MAGIC 1. Upload `MASTER_PANEL_FINAL.xlsx` via **Catalog → Volumes** or the file upload UI
# MAGIC 2. Note the path (e.g., `/Volumes/main/default/raw_data/MASTER_PANEL_FINAL.xlsx`)
# MAGIC 3. Update the `PATH` variable below

# COMMAND ----------

# Install openpyxl for Excel reading (only needed once per session)
%pip install openpyxl
dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd

# ── UPDATE THIS PATH to match where you uploaded the file ──
PATH = "/Volumes/main/default/raw_data/MASTER_PANEL_FINAL.xlsx"

# Read with pandas (openpyxl handles .xlsx)
pdf = pd.read_excel(PATH)
print(f"✅ Loaded {len(pdf)} rows, {len(pdf.columns)} columns")
print(f"Columns: {list(pdf.columns)}")

# COMMAND ----------

# Preview the raw data
display(pdf.head(20))

# COMMAND ----------

# Convert to Spark DataFrame and write as Bronze Delta table
df_bronze = spark.createDataFrame(pdf)

df_bronze.write.format("delta").mode("overwrite").saveAsTable("main.default.bronze_master_panel")

print(f"✅ Bronze table saved: main.default.bronze_master_panel")
print(f"   Rows: {df_bronze.count()}")
print(f"   Columns: {len(df_bronze.columns)}")

# COMMAND ----------

# Quick data quality check on the bronze layer
display(
    spark.sql("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT ISO3) as unique_countries,
            COUNT(DISTINCT Year) as unique_years,
            MIN(Year) as min_year,
            MAX(Year) as max_year,
            SUM(CASE WHEN Total_CBPF IS NULL THEN 1 ELSE 0 END) as null_target_count
        FROM main.default.bronze_master_panel
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ Bronze layer complete
# MAGIC Raw data is preserved in `main.default.bronze_master_panel` as a Delta table.
# MAGIC
# MAGIC **Next:** Run `02_silver_cleaning` to clean and validate the data.
