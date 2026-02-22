# pip install pandas numpy scikit-learn xgboost openpyxl

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")

# -----------------------------
# 1) Load data
# -----------------------------
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# Your data is in Crisis_Funding/data/, not FundingPredictionModels/data/
PATH = BASE_DIR.parent / "data" / "MASTER_PANEL_FINAL.xlsx"

print("Reading:", PATH)
print("Exists:", PATH.exists())

df_old = pd.read_excel(PATH)
df = df_old.drop(columns=["CBPF_Active_Years", "Prior_Year_CBPF"])

print(df.columns)
import numpy as np
import pandas as pd

# Always use a complete population signal
df["Pop_Used"] = pd.to_numeric(df["WB_Population"], errors="coerce")
df.loc[df["Population"].notna(), "Pop_Used"] = pd.to_numeric(df.loc[df["Population"].notna(), "Population"], errors="coerce")



# Build Need_Proxy (always exists)
def minmax(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

risk = df["INFORM_Risk"]
vuln = df["Vulnerability"]

df["Need_Proxy"] = pd.to_numeric(df["Total_PIN"], errors="coerce")
missing_pin = df["Need_Proxy"].isna()

df.loc[missing_pin, "Need_Proxy"] = (
    df.loc[missing_pin, "Pop_Used"] *
    (0.6 * minmax(risk) + 0.4 * minmax(vuln))
)

print("CHECK Need_Proxy non-null:", df["Need_Proxy"].notna().sum(), "of", len(df))
print("CHECK Pop_Used non-null:", df["Pop_Used"].notna().sum(), "of", len(df))


print("Columns:", list(df.columns))

for c in ["Population","WB_Population","Total_PIN","INFORM_Risk","Vulnerability","Need_Proxy","Total_CBPF"]:
    if c in df.columns:
        s = df[c]
        print(c, "non-null:", s.notna().sum(), "dtype:", s.dtype, "example:", s.dropna().head(3).tolist())
    else:
        print(c, "MISSING")

# Basic expectations
# - panel index: ISO3 + Year
# - target: Total_CBPF (or CBPF_per_capita / Log_CBPF)
# - need proxies: Population, INFORM_Risk, Vulnerability, etc.

df = df.copy()
df["Year"] = df["Year"].astype(int)

# -----------------------------
# 2) Choose target + build proxy-need
# -----------------------------
TARGET = "Total_CBPF"  # alternatives: "CBPF_per_capita", "Log_CBPF"

# If you have Total_PIN (people in need), use it; else build a proxy.
if "Total_PIN" in df.columns and df["Total_PIN"].notna().any():
    df["Need_Proxy"] = df["Total_PIN"]
else:
    # Proxy idea: population * scaled risk * scaled vulnerability
    # (cheap + usually available)
    pop = df["Population"] if "Population" in df.columns else df.get("WB_Population")
    risk = df["INFORM_Risk"] if "INFORM_Risk" in df.columns else 0
    vuln = df["Vulnerability"] if "Vulnerability" in df.columns else 0

    # Normalize risk/vuln roughly into [0,1] if they aren't already
    def minmax(s):
        s = pd.to_numeric(s, errors="coerce")
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    df["Need_Proxy"] = pd.to_numeric(pop, errors="coerce") * (0.6 * minmax(risk) + 0.4 * minmax(vuln))

# Optional: log-transform target for stability (funding is heavy-tailed)
# If your dataset already has Log_CBPF, you can just set TARGET="Log_CBPF".
df["_y"] = pd.to_numeric(df[TARGET], errors="coerce")

# Safeguard: drop rows with no target
df = df[df["_y"].notna()].copy()
import numpy as np
import pandas as pd

def minmax_np(x):
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    return (x - mn) / (mx - mn + 1e-9)

# rebuild Pop_Used safely (no index alignment bugs)
pop_used = pd.to_numeric(df["WB_Population"], errors="coerce").to_numpy()
pop_col = pd.to_numeric(df["Population"], errors="coerce").to_numpy()
pop_used = np.where(~np.isnan(pop_col), pop_col, pop_used)

risk = pd.to_numeric(df["INFORM_Risk"], errors="coerce").to_numpy()
vuln = pd.to_numeric(df["Vulnerability"], errors="coerce").to_numpy()
pin  = pd.to_numeric(df["Total_PIN"], errors="coerce").to_numpy()

fallback = pop_used * (0.6 * minmax_np(risk) + 0.4 * minmax_np(vuln))
need_proxy = np.where(np.isfinite(pin), pin, fallback)
need_proxy = np.where(np.isfinite(need_proxy), need_proxy, pop_used)

df["Need_Proxy"] = need_proxy

print("POST-FILTER Need_Proxy non-null:", df["Need_Proxy"].notna().sum(), "of", len(df))
print(df.groupby("Year")["Need_Proxy"].apply(lambda s: s.notna().sum()))

# If target is raw dollars, log1p helps a lot
use_log_target = (TARGET in ["Total_CBPF", "CBPF_per_capita"])
if use_log_target:
    df["_y_model"] = np.log1p(df["_y"].clip(lower=0))
else:
    df["_y_model"] = df["_y"]

# -----------------------------
# 3) Feature set
# -----------------------------
# Keep this list tight to avoid leakage:
# DO NOT include things like "Prior_Year_CBPF" if your goal is "fairness given need",
# because it bakes in historical funding politics and can hide unfairness.
candidate_numeric = [
    "Need_Proxy",
    "Pop_Used",
    "Density_per_km2", "Land_Area_km2",
    "INFORM_Risk", "Vulnerability", "Conflict_Probability",
    "Food_Security", "Governance", "Access_Healthcare",
    "Uprooted_People", "Vulnerable_Groups", "Hazard_Exposure",
    "GDP_per_capita", "Urban_pct",
    "Latitude", "Longitude",
    "INFORM_Risk_Change", "Vulnerability_Change",
]
candidate_categorical = [
    "ISO3",
    "Continent",
]

# Filter to columns that actually exist
num_features = [c for c in candidate_numeric if c in df.columns]
cat_features = [c for c in candidate_categorical if c in df.columns]

X = df[num_features + cat_features]
y = df["_y_model"].values

# -----------------------------
# 4) Time-series split by Year
# -----------------------------
# Train on years < test_year, test on test_year (walk-forward)
years = sorted(df["Year"].unique())

# You can adjust the minimum training window
min_train_years = 2
# right before training loop


splits = []
for i in range(min_train_years, len(years)):
    train_years = years[:i]
    test_year = years[i]
    train_idx = df.index[df["Year"].isin(train_years)].to_numpy()
    test_idx = df.index[df["Year"].eq(test_year)].to_numpy()
    splits.append((train_idx, test_idx))

# -----------------------------
# 5) Model pipeline (impute + one-hot + XGBoost)
# -----------------------------
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

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",     # faster
)
pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model),
])

# -----------------------------
# 6) Walk-forward evaluation + out-of-sample predictions
# -----------------------------
oof_pred = pd.Series(index=df.index, dtype=float)  # out-of-fold preds
print("AFTER target filter:")
print("rows:", len(df))
print("Need_Proxy non-null:", df["Need_Proxy"].notna().sum())
print(df.groupby("Year")["Need_Proxy"].apply(lambda s: s.notna().sum()))

df["Pop_Used"] = df["Pop_Used"].replace([np.inf, -np.inf], np.nan)
df["Need_Proxy"] = df["Need_Proxy"].replace([np.inf, -np.inf], np.nan)

metrics = []
for k, (tr, te) in enumerate(splits, start=1):
    X_tr = X.loc[tr]
    y_tr = df.loc[tr, "_y_model"].values
    X_te = X.loc[te]
    y_te = df.loc[te, "_y_model"].values
    # inside your fold loop, right before pipe.fit(...)
    print("train_years:", train_years, "test_year:", test_year,
      "Need_Proxy non-null in train:", X_tr["Need_Proxy"].notna().sum(),
      "of", len(X_tr))
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    oof_pred.loc[te] = pred

    mae = mean_absolute_error(y_te, pred)
    rmse = mean_squared_error(y_te, pred) ** 0.5
    r2 = r2_score(y_te, pred)

    metrics.append({
        "fold": k,
        "test_year": int(df.loc[te, "Year"].iloc[0]),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "n_test": len(te),
    })

metrics_df = pd.DataFrame(metrics)
print(metrics_df)
print("\nAvg metrics:")
print(metrics_df[["MAE", "RMSE", "R2"]].mean())

# -----------------------------
# 7) Convert predictions back to dollars + compute "fairness gap"
# -----------------------------
df["pred_log"] = oof_pred

if use_log_target:
    df["pred_funding"] = np.expm1(df["pred_log"]).clip(lower=0)
else:
    df["pred_funding"] = df["pred_log"]







df["actual_funding"] = df["_y"]

# Raw residual (Actual - Expected)
df["funding_gap"] = df["actual_funding"] - df["pred_funding"]

# Ratio residual: (Actual / Expected) - 1  (more interpretable across scales)
df["funding_ratio_gap"] = (df["actual_funding"] / (df["pred_funding"] + 1e-9)) - 1.0

# Flag "overlooked" vs "overfunded" (tune thresholds)
df["flag_overlooked"] = df["funding_ratio_gap"] < -0.35  # 35% below expected
df["flag_overfunded"] = df["funding_ratio_gap"] > 0.35

# -----------------------------
# 8) Project / efficiency-style anomalies (if you have beneficiaries & budget)
# -----------------------------
# Example: unusually high/low beneficiaries per budget
# You may have CBPF_Targeted / CBPF_Reached as "beneficiaries" proxies
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1) Metrics in model-space (log1p dollars if you used log)
y_true_model = df["_y_model"].values
y_pred_model = df["pred_log"].values  # this is oof_pred stored as pred_log in your script

mask = np.isfinite(y_true_model) & np.isfinite(y_pred_model)
y_true_model = y_true_model[mask]
y_pred_model = y_pred_model[mask]

print("\nOOF (walk-forward) REGRESSION METRICS — model space")
print("MAE :", mean_absolute_error(y_true_model, y_pred_model))
print("RMSE:", mean_squared_error(y_true_model, y_pred_model) ** 0.5)
print("R2  :", r2_score(y_true_model, y_pred_model))

# 2) Metrics in dollar space
y_true_usd = df.loc[mask, "actual_funding"].values
y_pred_usd = df.loc[mask, "pred_funding"].values

print("\nOOF (walk-forward) REGRESSION METRICS — dollar space")
print("MAE_USD :", mean_absolute_error(y_true_usd, y_pred_usd))
print("RMSE_USD:", mean_squared_error(y_true_usd, y_pred_usd) ** 0.5)

# Percent-style errors (avoid divide by tiny numbers with eps)
eps = 1e-9
mape = np.mean(np.abs((y_true_usd - y_pred_usd) / (np.abs(y_true_usd) + eps))) * 100
smape = np.mean(2*np.abs(y_true_usd - y_pred_usd)/(np.abs(y_true_usd)+np.abs(y_pred_usd)+eps)) * 100

print("MAPE%  :", mape)
print("sMAPE% :", smape)


if "CBPF_Reached" in df.columns and "Total_CBPF" in df.columns:
    df["beneficiaries_per_million"] = df["CBPF_Reached"] / (df["Total_CBPF"] / 1_000_000 + 1e-9)

    # simple robust z-score using MAD
    med = df["beneficiaries_per_million"].median()
    mad = np.median(np.abs(df["beneficiaries_per_million"] - med)) + 1e-9
    df["efficiency_robust_z"] = 0.6745 * (df["beneficiaries_per_million"] - med) / mad

    df["flag_low_efficiency"] = df["efficiency_robust_z"] < -2.5
    df["flag_high_efficiency"] = df["efficiency_robust_z"] > 2.5
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)


import shap
import matplotlib.pyplot as plt

explainer = shap.Explainer(pipe.named_steps["model"])
X_trans = pipe.named_steps["preprocess"].transform(X)
shap_values = explainer(X_trans)

shap.summary_plot(shap_values, X_trans, show=False)
plt.savefig(OUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Wrote: shap_summary.png")



# -----------------------------
# 9) "Comparable benchmarks" (nearest neighbors in feature space)
# -----------------------------
# For a given row, find similar rows by standardized numeric features
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

sim_features = [c for c in num_features if c in df.columns]
sim_df = df[sim_features].copy()
sim_df = sim_df.replace([np.inf, -np.inf], np.nan)

sim_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

Z = sim_pipe.fit_transform(sim_df)

nn = NearestNeighbors(n_neighbors=6, metric="euclidean")  # 1 self + 5 neighbors
nn.fit(Z)

def similar_projects(idx):
    row_pos = df.index.get_loc(idx)
    dists, neigh_pos = nn.kneighbors(Z[row_pos].reshape(1, -1))
    neigh_idx = df.index[neigh_pos[0]]
    # drop self
    neigh_idx = [i for i in neigh_idx if i != idx][:5]
    return df.loc[neigh_idx, ["ISO3", "Year", "actual_funding", "pred_funding", "funding_ratio_gap"] + (["Continent"] if "Continent" in df.columns else [])]

# Example usage:
example_idx = df.index[0]
print("\nSimilar benchmarks for first row:\n", similar_projects(example_idx))

# -----------------------------
# 10) Save scored table for mapping / dashboards
# -----------------------------
out_cols = ["ISO3", "Year", "actual_funding", "pred_funding", "funding_gap", "funding_ratio_gap",
            "flag_overlooked", "flag_overfunded"]
if "Latitude" in df.columns and "Longitude" in df.columns:
    out_cols += ["Latitude", "Longitude"]
if "Continent" in df.columns:
    out_cols += ["Continent"]
# -----------------------------
# 10) Save scored table for mapping / dashboards
# -----------------------------

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# --- Drop rows that were never in a test fold (no OOF prediction) ---
scored = df[df["pred_log"].notna()].copy()
print(f"\nRows with OOF predictions: {len(scored)} / {len(df)}")

# --- Build out_cols safely (only include columns that actually exist) ---
base_cols = ["ISO3", "Year", "actual_funding", "pred_funding",
             "funding_gap", "funding_ratio_gap",
             "flag_overlooked", "flag_overfunded"]
optional_cols = ["Latitude", "Longitude", "Continent"]

out_cols = [c for c in base_cols if c in scored.columns]
out_cols += [c for c in optional_cols if c in scored.columns]

fair_path = OUT_DIR / "scored_funding_fairness.csv"
scored[out_cols].to_csv(fair_path, index=False)
print("Wrote:", fair_path)
# --- Cluster divergence ---
# --- Cluster divergence ---
print("\nREACHED CLUSTER BLOCK ✅")

# 1) Load combined sectors/cluster file
projects_path = BASE_DIR.parent / "data" / "SectorsOverview_Combined.csv"
print("Reading projects:", projects_path, "Exists:", projects_path.exists())

if not projects_path.exists():
    print("⚠️  Projects file not found — skipping cluster divergence.")
else:
    projects = pd.read_csv(projects_path)
    print("Projects columns (raw):", list(projects.columns))

    # 2) Normalize column names from your sector overview export
    # Your screenshot looks like: Year, CBPF Name, Cluster, TotalAlloc..., Targeted..., Reached...
    # We only need Country, Year, Cluster, Budget
    rename_map = {}

    if "CBPF Name" in projects.columns:
        rename_map["CBPF Name"] = "Country"
    if "TotalAllocation" in projects.columns:
        rename_map["TotalAllocation"] = "Budget"
    if "Total Allocations" in projects.columns:
        rename_map["Total Allocations"] = "Budget"
    if "Total Allocation" in projects.columns:
        rename_map["Total Allocation"] = "Budget"
    if "TotalAlloc" in projects.columns:
        rename_map["TotalAlloc"] = "Budget"
    if "TotalAllocated" in projects.columns:
        rename_map["TotalAllocated"] = "Budget"

projects = projects.rename(columns=rename_map)
# 1) Validate required columns exist AFTER rename
required = {"Country", "Year", "Cluster", "Budget"}
missing = required - set(projects.columns)

if missing:
    print(f"⚠️  Projects file missing columns {missing} — skipping cluster divergence.")
else:
    # 2) Map Country -> ISO3 (ONLY after we know Country exists)
    import re
    import pycountry

    def country_to_iso3(name: str):
        if not isinstance(name, str) or not name.strip():
            return None
        n = name.strip()

        # strip RhPF suffix like "Niger (RhPF-WCA)"
        n = re.sub(r"\s*\(RhPF-[^)]+\)\s*$", "", n).strip()

        overrides = {
            "CAR": "CAF",
            "oPt": "PSE",
            "Syria Cross bordder": "SYR",
            "Syria Cross border": "SYR",
            "Democratic Republic of the Congo": "COD",
            "DRC": "COD",
        }
        if n in overrides:
            return overrides[n]

        try:
            hit = pycountry.countries.lookup(n)
            return hit.alpha_3
        except Exception:
            return None

    projects["ISO3"] = projects["Country"].apply(country_to_iso3)

    unmapped = projects.loc[projects["ISO3"].isna(), "Country"].unique()
    if len(unmapped):
        print("⚠️ Unmapped countries (dropping):", list(unmapped)[:50])

    projects = projects[projects["ISO3"].notna()].copy()

    # 3) Clean types
    projects["Year"] = pd.to_numeric(projects["Year"], errors="coerce").astype("Int64")
    projects["Budget"] = pd.to_numeric(projects["Budget"], errors="coerce")
    projects = projects.dropna(subset=["Year", "Cluster", "Budget"])

    # 4) Aggregate to ISO3-Year-Cluster
    cluster_share = (
        projects.groupby(["ISO3", "Year", "Cluster"], as_index=False)["Budget"].sum()
    )

    total_by_cy = (
        cluster_share.groupby(["ISO3", "Year"], as_index=False)["Budget"]
        .sum()
        .rename(columns={"Budget": "Total"})
    )

    cluster_share = cluster_share.merge(total_by_cy, on=["ISO3", "Year"], how="left")
    cluster_share["cluster_frac"] = cluster_share["Budget"] / (cluster_share["Total"] + 1e-9)

    # 5) Merge with model outputs
    merge_df = scored[["ISO3", "Year", "pred_funding", "actual_funding", "funding_gap"]].drop_duplicates(["ISO3", "Year"])
    cluster_share = cluster_share.merge(merge_df, on=["ISO3", "Year"], how="left")

    # 6) Compute expected vs actual per cluster
    cluster_share["expected_cluster_funding"] = cluster_share["pred_funding"] * cluster_share["cluster_frac"]
    cluster_share["actual_cluster_funding"] = cluster_share["Budget"]
    cluster_share["cluster_gap"] = cluster_share["actual_cluster_funding"] - cluster_share["expected_cluster_funding"]

    # 7) Write output
    cluster_path = OUT_DIR / "scored_cluster_divergence.csv"
    cluster_share.to_csv(cluster_path, index=False)
    print("Wrote:", cluster_path)
    print(f"Rows: {len(cluster_share)}, Clusters: {cluster_share['Cluster'].nunique()}")