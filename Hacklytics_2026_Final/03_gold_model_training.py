# Databricks notebook source
# MAGIC %md
# MAGIC # 🥇 Notebook 03 — Gold Layer: Model Training + MLflow
# MAGIC
# MAGIC **Hacklytics 2026 | Databricks × United Nations Challenge**
# MAGIC
# MAGIC This notebook:
# MAGIC - Trains an XGBoost model with walk-forward time-series validation
# MAGIC - Tracks **every experiment run** in MLflow (parameters, metrics, model artifacts)
# MAGIC - Registers the best model in the MLflow Model Registry
# MAGIC - Computes funding fairness gaps and flags overlooked crises

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install xgboost shap "numpy<2.0"

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Silver Data

# COMMAND ----------

df = spark.table("main.default.silver_master_panel").toPandas()
print(f"Loaded {len(df)} rows, {df['ISO3'].nunique()} countries, years {df['Year'].min()}-{df['Year'].max()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Features
# MAGIC
# MAGIC We intentionally **exclude** prior-year CBPF to avoid baking in historical funding politics.
# MAGIC The model predicts what funding **should** look like given humanitarian need — not what it historically was.

# COMMAND ----------

# ── Feature definitions ──
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

print(f"Numeric features ({len(num_features)}): {num_features}")
print(f"Categorical features ({len(cat_features)}): {cat_features}")

X = df[num_features + cat_features]
y = df["_y_model"].values

# COMMAND ----------

# MAGIC %md
# MAGIC ## Walk-Forward Time-Series Splits

# COMMAND ----------

years = sorted(df["Year"].unique())
min_train_years = 2

splits = []
for i in range(min_train_years, len(years)):
    train_years = years[:i]
    test_year = years[i]
    train_idx = df.index[df["Year"].isin(train_years)].to_numpy()
    test_idx  = df.index[df["Year"].eq(test_year)].to_numpy()
    splits.append((train_idx, test_idx, train_years, test_year))

print(f"Created {len(splits)} walk-forward splits:")
for tr, te, tr_yrs, te_yr in splits:
    print(f"  Train {tr_yrs[0]}-{tr_yrs[-1]} ({len(tr)} rows) → Test {te_yr} ({len(te)} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Experiment: Hyperparameter Search
# MAGIC
# MAGIC We'll try multiple XGBoost configurations and track them all in MLflow.

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")
experiment_name = "/Users/conradwyrick135@gmail.com/crisis-funding-fairness"
mlflow.set_experiment(experiment_name)
print(f"MLflow experiment: {experiment_name}")

# COMMAND ----------

# ── Preprocessing pipeline ──
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ],
    remainder="drop",
)

# COMMAND ----------

# ── Hyperparameter grid ──
param_grid = [
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.85},
    {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.80},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08, "subsample": 0.90},
]

best_r2 = -np.inf
best_run_id = None
best_pipe = None

for config_idx, params in enumerate(param_grid):
    with mlflow.start_run(run_name=f"xgb_config_{config_idx}") as run:
        
        # Log parameters
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("learning_rate", params["learning_rate"])
        mlflow.log_param("subsample", params["subsample"])
        mlflow.log_param("num_features", len(num_features))
        mlflow.log_param("cat_features", len(cat_features))
        mlflow.log_param("target", "log1p(Total_CBPF)")
        mlflow.log_param("n_splits", len(splits))
        
        # Build model
        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        
        pipe = Pipeline(steps=[
            ("preprocess", preprocess),
            ("model", model),
        ])
        
        # Walk-forward evaluation
        oof_pred = pd.Series(index=df.index, dtype=float)
        fold_metrics = []
        
        for k, (tr, te, tr_yrs, te_yr) in enumerate(splits):
            X_tr, y_tr = X.loc[tr], df.loc[tr, "_y_model"].values
            X_te, y_te = X.loc[te], df.loc[te, "_y_model"].values
            
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            oof_pred.loc[te] = pred
            
            fold_metrics.append({
                "fold": k, "test_year": te_yr,
                "MAE": mean_absolute_error(y_te, pred),
                "RMSE": mean_squared_error(y_te, pred) ** 0.5,
                "R2": r2_score(y_te, pred),
            })
        
        metrics_df = pd.DataFrame(fold_metrics)
        avg_mae  = metrics_df["MAE"].mean()
        avg_rmse = metrics_df["RMSE"].mean()
        avg_r2   = metrics_df["R2"].mean()
        
        # Log aggregate metrics
        mlflow.log_metric("avg_MAE_log", avg_mae)
        mlflow.log_metric("avg_RMSE_log", avg_rmse)
        mlflow.log_metric("avg_R2_log", avg_r2)
        
        # Dollar-space metrics
        mask = oof_pred.notna()
        y_true_usd = df.loc[mask, "_y"].values
        y_pred_usd = np.expm1(oof_pred[mask].values).clip(min=0)
        
        mae_usd = mean_absolute_error(y_true_usd, y_pred_usd)
        rmse_usd = mean_squared_error(y_true_usd, y_pred_usd) ** 0.5
        
        mlflow.log_metric("MAE_USD", mae_usd)
        mlflow.log_metric("RMSE_USD", rmse_usd)
        
        # Log the model artifact
        mlflow.sklearn.log_model(pipe, "funding_fairness_model")
        
        # Log fold-level metrics as artifact
        metrics_df.to_csv("/tmp/fold_metrics.csv", index=False)
        mlflow.log_artifact("/tmp/fold_metrics.csv")
        
        print(f"Config {config_idx}: R2={avg_r2:.4f}, MAE_log={avg_mae:.4f}, MAE_USD=${mae_usd:,.0f}")
        
        # Track best
        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_run_id = run.info.run_id
            best_pipe = pipe
            best_oof = oof_pred.copy()

print(f"\n🏆 Best config: run_id={best_run_id}, R2={best_r2:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Best Model in MLflow Registry

# COMMAND ----------

# Register the best model
model_name = "crisis_funding_fairness"
model_uri = f"runs:/{best_run_id}/funding_fairness_model"

registered = mlflow.register_model(model_uri, model_name)
print(f"✅ Model registered: {model_name}, version {registered.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Funding Fairness Gaps

# COMMAND ----------

# ── Convert predictions back to dollars ──
df["pred_log"] = best_oof
df["pred_funding"] = np.expm1(df["pred_log"]).clip(lower=0)
df["actual_funding"] = df["_y"]

# ── Funding gap metrics ──
df["funding_gap"] = df["actual_funding"] - df["pred_funding"]
df["funding_ratio_gap"] = (df["actual_funding"] / (df["pred_funding"] + 1e-9)) - 1.0

# ── Flag overlooked vs overfunded ──
df["flag_overlooked"]  = df["funding_ratio_gap"] < -0.35
df["flag_overfunded"]  = df["funding_ratio_gap"] >  0.35

n_overlooked = df["flag_overlooked"].sum()
n_overfunded = df["flag_overfunded"].sum()
print(f"Flagged: {n_overlooked} overlooked, {n_overfunded} overfunded (of {len(df)} total)")

# COMMAND ----------

# Show the most overlooked crises
overlooked = (df[df["flag_overlooked"]]
    .sort_values("funding_ratio_gap")
    [["ISO3", "Year", "actual_funding", "pred_funding", "funding_ratio_gap", "Need_Proxy"]]
    .head(20)
)
display(spark.createDataFrame(overlooked))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Efficiency Anomalies (if beneficiary data exists)

# COMMAND ----------

if "CBPF_Reached" in df.columns and "Total_CBPF" in df.columns:
    df["beneficiaries_per_million"] = df["CBPF_Reached"] / (df["Total_CBPF"] / 1_000_000 + 1e-9)
    
    med = df["beneficiaries_per_million"].median()
    mad = np.median(np.abs(df["beneficiaries_per_million"] - med)) + 1e-9
    df["efficiency_robust_z"] = 0.6745 * (df["beneficiaries_per_million"] - med) / mad
    
    df["flag_low_efficiency"]  = df["efficiency_robust_z"] < -2.5
    df["flag_high_efficiency"] = df["efficiency_robust_z"] >  2.5
    
    print(f"Efficiency flags: {df['flag_low_efficiency'].sum()} low, {df['flag_high_efficiency'].sum()} high")
else:
    print("CBPF_Reached column not found — skipping efficiency analysis")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Gold Table

# COMMAND ----------

# ── Save scored data as Gold Delta table ──
df_gold = spark.createDataFrame(df)
df_gold.write.format("delta").mode("overwrite").saveAsTable("main.default.gold_scored_funding")

print(f"✅ Gold table saved: main.default.gold_scored_funding")
print(f"   Rows: {len(df)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ Gold layer complete
# MAGIC
# MAGIC - Model tracked in MLflow with {len(param_grid)} experiment runs
# MAGIC - Best model registered in Model Registry
# MAGIC - Funding fairness gaps computed and saved to `main.default.gold_scored_funding`
# MAGIC
# MAGIC **Next:** Run `04_shap_benchmarking` for explainability and comparable projects.
