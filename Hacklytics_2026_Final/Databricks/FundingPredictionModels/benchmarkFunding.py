# pip install pandas numpy scikit-learn xgboost openpyxl pycountry shap

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import shap
import re

# Try importing pycountry — only needed for cluster block
try:
    import pycountry
    HAS_PYCOUNTRY = True
except ImportError:
    HAS_PYCOUNTRY = False
    print("⚠️  pycountry not installed — cluster divergence will be skipped.")


# ======================================================================
def benchmarkPredictor(isFull):
    """
    isFull=False  →  reduced / "need-only" model  (drops ISO3, Continent, political features)
    isFull=True   →  full model                   (keeps everything except ISO3 as feature)
    """
    tag = "full" if isFull else "reduced"
    print(f"\n{'='*60}")
    print(f"  RUNNING  {tag.upper()}  MODEL")
    print(f"{'='*60}\n")

    BASE_DIR = Path(__file__).resolve().parent
    OUT_DIR = BASE_DIR / "outputs"
    OUT_DIR.mkdir(exist_ok=True)

    PATH = BASE_DIR.parent / "data" / "MASTER_PANEL_FINAL.xlsx"
    print("Reading:", PATH)
    print("Exists:", PATH.exists())

    # ------------------------------------------------------------------
    # 1) Load + drop columns
    # ------------------------------------------------------------------
    if not isFull:
        DROP_COLS = [
            "Continent",
            "Prior_Year_CBPF",
            "popularity_scaled_anchor_sudan",
            "LPI Rank", "LPI Score",
            "Customs Rank", "Customs Score",
            "Infrastructure Rank", "Infrastructure Score",
            "International shipments Rank", "International shipments Score",
            "Logistics competence Rank", "Logistics competence Score",
            "Tracking & tracing Rank", "Tracking & tracing Score",
            "Timeliness Rank", "Timeliness Score",
            "global_crisis_count_inform_ge_thresh",
        ]
    else:
        DROP_COLS = []

    df_old = pd.read_excel(PATH)
    # Always keep ISO3 in the dataframe for merging — just don't use it as a feature in reduced
    df = df_old.drop(columns=[c for c in DROP_COLS if c in df_old.columns], errors="ignore")

    print("Columns after drop:", list(df.columns))

    # ------------------------------------------------------------------
    # 2) Population signal
    # ------------------------------------------------------------------
    df["Pop_Used"] = pd.to_numeric(df["WB_Population"], errors="coerce")
    mask_pop = df["Population"].notna()
    df.loc[mask_pop, "Pop_Used"] = pd.to_numeric(
        df.loc[mask_pop, "Population"], errors="coerce"
    )

    # ------------------------------------------------------------------
    # 3) Need_Proxy
    # ------------------------------------------------------------------
    def minmax_np(x):
        x = np.asarray(x, dtype=float)
        mn, mx = np.nanmin(x), np.nanmax(x)
        return (x - mn) / (mx - mn + 1e-9)

    pop_used = pd.to_numeric(df["WB_Population"], errors="coerce").to_numpy()
    pop_col  = pd.to_numeric(df["Population"], errors="coerce").to_numpy()
    pop_used = np.where(~np.isnan(pop_col), pop_col, pop_used)

    risk = pd.to_numeric(df["INFORM_Risk"], errors="coerce").to_numpy()
    vuln = pd.to_numeric(df["Vulnerability"], errors="coerce").to_numpy()
    pin  = pd.to_numeric(df["Total_PIN"], errors="coerce").to_numpy()

    fallback   = pop_used * (0.6 * minmax_np(risk) + 0.4 * minmax_np(vuln))
    need_proxy = np.where(np.isfinite(pin), pin, fallback)
    need_proxy = np.where(np.isfinite(need_proxy), need_proxy, pop_used)

    df["Need_Proxy"] = need_proxy
    df["Pop_Used"]   = pop_used

    print("Need_Proxy non-null:", df["Need_Proxy"].notna().sum(), "/", len(df))
    print("Pop_Used non-null:  ", pd.Series(pop_used).notna().sum(), "/", len(df))

    # ------------------------------------------------------------------
    # 4) Target
    # ------------------------------------------------------------------
    df = df.copy()
    df["Year"] = df["Year"].astype(int)

    TARGET = "Total_CBPF"
    df["_y"] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df["_y"].notna()].copy()

    use_log_target = True
    df["_y_model"] = np.log1p(df["_y"].clip(lower=0))

    # Replace inf in key columns
    df["Pop_Used"]   = df["Pop_Used"].replace([np.inf, -np.inf], np.nan)
    df["Need_Proxy"] = df["Need_Proxy"].replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # 5) Features — ISO3 kept in df but only used as feature if isFull
    # ------------------------------------------------------------------
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

    if isFull:
        candidate_categorical = ["Continent"]
    else:
        candidate_categorical = []  # reduced model: no identity / political features

    num_features = [c for c in candidate_numeric if c in df.columns]
    cat_features = [c for c in candidate_categorical if c in df.columns]

    # Also add the full-model-only numeric columns if they survived the drop
    if isFull:
        extra_numeric = [
            "Prior_Year_CBPF",
            "popularity_scaled_anchor_sudan",
            "LPI Score", "Customs Score",
            "Infrastructure Score",
            "International shipments Score",
            "Logistics competence Score",
            "Tracking & tracing Score",
            "Timeliness Score",
            "global_crisis_count_inform_ge_thresh",
        ]
        for c in extra_numeric:
            if c in df.columns and c not in num_features:
                num_features.append(c)

    print(f"\n[{tag}] Numeric features ({len(num_features)}):", num_features)
    print(f"[{tag}] Categorical features ({len(cat_features)}):", cat_features)

    X = df[num_features + cat_features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # 6) Walk-forward split
    # ------------------------------------------------------------------
    years = sorted(df["Year"].unique())
    min_train_years = 2

    splits = []
    for i in range(min_train_years, len(years)):
        train_yrs = years[:i]
        test_yr   = years[i]
        train_idx = df.index[df["Year"].isin(train_yrs)].to_numpy()
        test_idx  = df.index[df["Year"].eq(test_yr)].to_numpy()
        splits.append((train_idx, test_idx))

    # ------------------------------------------------------------------
    # 7) Pipeline
    # ------------------------------------------------------------------
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = [("num", numeric_transformer, num_features)]
    if cat_features:
        transformers.append(("cat", categorical_transformer, cat_features))

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

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
        tree_method="hist",
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])

    # ------------------------------------------------------------------
    # 8) Train + predict (walk-forward)
    # ------------------------------------------------------------------
    oof_pred = pd.Series(index=df.index, dtype=float)
    metrics = []

    for k, (tr, te) in enumerate(splits, start=1):
        X_tr, y_tr = X.loc[tr], df.loc[tr, "_y_model"].values
        X_te, y_te = X.loc[te], df.loc[te, "_y_model"].values

        train_years_fold = sorted(df.loc[tr, "Year"].unique())
        test_year_fold   = int(df.loc[te, "Year"].iloc[0])

        print(f"  Fold {k}: train {train_years_fold} → test {test_year_fold}  "
              f"(train={len(tr)}, test={len(te)})")

        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        oof_pred.loc[te] = pred

        mae  = mean_absolute_error(y_te, pred)
        rmse = mean_squared_error(y_te, pred) ** 0.5
        r2   = r2_score(y_te, pred)

        metrics.append({
            "fold": k,
            "test_year": test_year_fold,
            "MAE": mae, "RMSE": rmse, "R2": r2,
            "n_test": len(te),
        })

    metrics_df = pd.DataFrame(metrics)
    print(f"\n[{tag}] Per-fold metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\n[{tag}] Average:")
    print(metrics_df[["MAE", "RMSE", "R2"]].mean())

    # ------------------------------------------------------------------
    # 9) Fairness gap
    # ------------------------------------------------------------------
    df["pred_log"] = oof_pred

    if use_log_target:
        df["pred_funding"] = np.expm1(df["pred_log"]).clip(lower=0)
    else:
        df["pred_funding"] = df["pred_log"]

    df["actual_funding"]    = df["_y"]
    df["funding_gap"]       = df["actual_funding"] - df["pred_funding"]
    df["funding_ratio_gap"] = (df["actual_funding"] / (df["pred_funding"] + 1e-9)) - 1.0
    df["flag_overlooked"]   = df["funding_ratio_gap"] < -0.35
    df["flag_overfunded"]   = df["funding_ratio_gap"] > 0.35

    # ------------------------------------------------------------------
    # 10) Dollar-space metrics
    # ------------------------------------------------------------------
    y_true_m = df["_y_model"].values
    y_pred_m = df["pred_log"].values
    mask = np.isfinite(y_true_m) & np.isfinite(y_pred_m)

    print(f"\n[{tag}] OOF METRICS — model space")
    print("  MAE :", mean_absolute_error(y_true_m[mask], y_pred_m[mask]))
    print("  RMSE:", mean_squared_error(y_true_m[mask], y_pred_m[mask]) ** 0.5)
    print("  R2  :", r2_score(y_true_m[mask], y_pred_m[mask]))

    y_true_usd = df.loc[mask, "actual_funding"].values
    y_pred_usd = df.loc[mask, "pred_funding"].values

    print(f"\n[{tag}] OOF METRICS — dollar space")
    print("  MAE_USD :", mean_absolute_error(y_true_usd, y_pred_usd))
    print("  RMSE_USD:", mean_squared_error(y_true_usd, y_pred_usd) ** 0.5)

    eps = 1e-9
    mape  = np.mean(np.abs((y_true_usd - y_pred_usd) / (np.abs(y_true_usd) + eps))) * 100
    smape = np.mean(2 * np.abs(y_true_usd - y_pred_usd) /
                    (np.abs(y_true_usd) + np.abs(y_pred_usd) + eps)) * 100
    print("  MAPE% :", mape)
    print("  sMAPE%:", smape)

    # ------------------------------------------------------------------
    # 11) Efficiency anomalies (if columns exist)
    # ------------------------------------------------------------------
    if "CBPF_Reached" in df.columns and "Total_CBPF" in df.columns:
        df["beneficiaries_per_million"] = (
            df["CBPF_Reached"] / (df["Total_CBPF"] / 1_000_000 + 1e-9)
        )
        med = df["beneficiaries_per_million"].median()
        mad = np.median(np.abs(df["beneficiaries_per_million"] - med)) + 1e-9
        df["efficiency_robust_z"] = 0.6745 * (df["beneficiaries_per_million"] - med) / mad
        df["flag_low_efficiency"]  = df["efficiency_robust_z"] < -2.5
        df["flag_high_efficiency"] = df["efficiency_robust_z"] > 2.5

    # ------------------------------------------------------------------
    # 12) SHAP
    # ------------------------------------------------------------------
    try:
        explainer   = shap.Explainer(pipe.named_steps["model"])
        X_trans     = pipe.named_steps["preprocess"].transform(X)
        shap_values = explainer(X_trans)

        # Get real feature names after preprocessing
        feature_names = pipe.named_steps["preprocess"].get_feature_names_out()

        # Wrap transformed X with column names
        X_trans_named = pd.DataFrame(X_trans, columns=feature_names)

        shap.summary_plot(shap_values, X_trans_named, show=False)
        shap_path = OUT_DIR / f"shap_summary_{tag}.png"
        plt.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Wrote: {shap_path}")
    except Exception as e:
        print(f"⚠️  SHAP failed: {e}")

    # ------------------------------------------------------------------
    # 13) Nearest-neighbor benchmarks
    # ------------------------------------------------------------------
    sim_features = [c for c in num_features if c in df.columns]
    sim_df = df[sim_features].replace([np.inf, -np.inf], np.nan)

    sim_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    Z = sim_pipe.fit_transform(sim_df)

    nn = NearestNeighbors(n_neighbors=6, metric="euclidean")
    nn.fit(Z)

    def similar_rows(idx):
        row_pos = df.index.get_loc(idx)
        dists, neigh_pos = nn.kneighbors(Z[row_pos].reshape(1, -1))
        neigh_idx = df.index[neigh_pos[0]]
        neigh_idx = [i for i in neigh_idx if i != idx][:5]
        cols = ["ISO3", "Year", "actual_funding", "pred_funding", "funding_ratio_gap"]
        if "Continent" in df.columns:
            cols.append("Continent")
        return df.loc[neigh_idx, cols]

    try:
        print(f"\n[{tag}] Similar benchmarks for first row:")
        print(similar_rows(df.index[0]))
    except Exception as e:
        print(f"⚠️  similar_rows failed: {e}")

    # ------------------------------------------------------------------
    # 14) Write fairness CSV
    # ------------------------------------------------------------------
    scored = df[df["pred_log"].notna()].copy()
    print(f"\n[{tag}] Rows with OOF predictions: {len(scored)} / {len(df)}")

    base_cols     = ["ISO3", "Year", "actual_funding", "pred_funding",
                     "funding_gap", "funding_ratio_gap",
                     "flag_overlooked", "flag_overfunded"]
    optional_cols = ["Latitude", "Longitude", "Continent"]

    out_cols = [c for c in base_cols if c in scored.columns]
    out_cols += [c for c in optional_cols if c in scored.columns]

    fair_path = OUT_DIR / f"scored_funding_fairness_{tag}.csv"
    scored[out_cols].to_csv(fair_path, index=False)
    print(f"Wrote: {fair_path}")
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  REDUCED MODEL (need-only features)")
    print("=" * 60)
    benchmarkPredictor(isFull=False)

    print("\n\n")
    print("=" * 60)
    print("  FULL MODEL (all features incl. political/logistic)")
    print("=" * 60)
    benchmarkPredictor(isFull=True)