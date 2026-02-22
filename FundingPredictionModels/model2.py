"""
Cluster Divergence Model
========================
For each CBPF country (latest year), find the most similar country-year by
risk profile (SAME YEAR ONLY), then compare where their cluster-level funding
allocations diverge. Outputs a scored table + a per-country divergence report.

Expects:
  - MASTER_PANEL_FINAL.xlsx        (risk features by ISO3 × Year)
  - SectorsOverview_Combined.csv   (cluster allocations by country × year)

Usage:
  python cluster_divergence_model.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

# ─── paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

PANEL_PATH = BASE_DIR.parent / "data" / "MASTER_PANEL_FINAL.xlsx"
SECTOR_PATH = BASE_DIR.parent / "data" / "SectorsOverview_Combined.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ─── country name → ISO3 map ─────────────────────────────────────
NAME_TO_ISO3 = {
    "Afghanistan": "AFG",
    "Burkina Faso (RhPF-WCA)": "BFA",
    "CAR": "CAF",
    "Colombia (RhPF-LAC)": "COL",
    "DRC": "COD",
    "Ethiopia": "ETH",
    "Haiti (RhPF-LAC)": "HTI",
    "Iraq": "IRQ",
    "Jordan": "JOR",
    "Lebanon": "LBN",
    "Mali (RhPF-WCA)": "MLI",
    "Myanmar": "MMR",
    "Niger (RhPF-WCA)": "NER",
    "Nigeria": "NGA",
    "Pakistan": "PAK",
    "Somalia": "SOM",
    "South Sudan": "SSD",
    "Sudan": "SDN",
    "Syria": "SYR",
    "Syria Cross border": "SYR",
    "Ukraine": "UKR",
    "Venezuela": "VEN",
    "Yemen": "YEM",
    "oPt": "PSE",
}

# ─── risk features to match on ───────────────────────────────────
RISK_FEATURES = [
    "INFORM_Risk",
    "Vulnerability",
    "Conflict_Probability",
    "Food_Security",
    "Governance",
    "Access_Healthcare",
    "Uprooted_People",
    "Vulnerable_Groups",
    "Hazard_Exposure",
    "GDP_per_capita",
]


def load_panel(path: Path) -> pd.DataFrame:
    """Load master panel and keep risk features."""
    print(f"Loading panel: {path}")
    df = pd.read_excel(path)
    df["Year"] = df["Year"].astype(int)

    keep = ["ISO3", "Year"] + [c for c in RISK_FEATURES if c in df.columns]
    df = df[keep].copy()

    for c in RISK_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_sectors(path: Path) -> pd.DataFrame:
    """Load sector overview → map to ISO3 → compute cluster fractions."""
    print(f"Loading sectors: {path}")
    sec = pd.read_csv(path)

    sec = sec.rename(
        columns={
            "CBPF Name": "Country",
            "Total Allocations": "Budget",
        }
    )

    sec["ISO3"] = sec["Country"].map(NAME_TO_ISO3)

    unmapped = sec.loc[sec["ISO3"].isna(), "Country"].unique()
    if len(unmapped):
        print(f"  ⚠️  Unmapped countries (dropped): {list(unmapped)}")

    sec = sec[sec["ISO3"].notna()].copy()

    sec["Year"] = pd.to_numeric(sec["Year"], errors="coerce").astype("Int64")
    sec["Budget"] = pd.to_numeric(sec["Budget"], errors="coerce")
    sec = sec.dropna(subset=["Year", "Cluster", "Budget"])
    sec["Year"] = sec["Year"].astype(int)

    # Aggregate Syria + Syria Cross border into one ISO3
    sec = sec.groupby(["ISO3", "Year", "Cluster"], as_index=False)["Budget"].sum()

    totals = (
        sec.groupby(["ISO3", "Year"], as_index=False)["Budget"]
        .sum()
        .rename(columns={"Budget": "Total"})
    )

    sec = sec.merge(totals, on=["ISO3", "Year"], how="left")
    sec["cluster_frac"] = sec["Budget"] / (sec["Total"] + 1e-9)

    return sec


def build_cluster_vectors(sec: pd.DataFrame) -> pd.DataFrame:
    """Pivot cluster fractions into a wide vector per country-year."""
    wide = (
        sec.pivot_table(
            index=["ISO3", "Year"],
            columns="Cluster",
            values="cluster_frac",
            fill_value=0.0,
        )
        .reset_index()
    )
    wide.columns.name = None
    return wide


def find_nearest_risk_profiles(
    panel: pd.DataFrame,
    target_year: int | None = None,
    n_neighbors: int = 3,
) -> pd.DataFrame:
    """
    SAME-YEAR ONLY.
    For each ISO3 in target_year, find the n most similar OTHER ISO3s
    in the SAME target_year by risk profile.

    Returns:
      target_ISO3, target_Year, match_ISO3, match_Year, distance, rank
    """
    feats = [c for c in RISK_FEATURES if c in panel.columns]
    if not feats:
        raise ValueError("No risk features found in panel")

    df = panel.dropna(subset=feats, how="all").copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    if target_year is None:
        target_year = int(df["Year"].max())

    # restrict to same year only (both targets + candidate pool)
    df = df[df["Year"] == target_year].copy()
    if df.empty:
        raise ValueError(f"No rows in panel for target_year={target_year}")

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    Z = pipe.fit_transform(df[feats])

    rows = []
    for pos in range(len(df)):
        t_iso3 = df.iloc[pos]["ISO3"]

        # candidates: same year, different ISO3
        cand_mask = df["ISO3"] != t_iso3
        cand_pos = np.where(cand_mask.to_numpy())[0]
        if len(cand_pos) == 0:
            continue

        k = min(n_neighbors, len(cand_pos))
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(Z[cand_pos])

        dists, neigh = nn.kneighbors(Z[pos].reshape(1, -1))

        for j in range(k):
            cpos = cand_pos[neigh[0, j]]
            rows.append(
                {
                    "target_ISO3": t_iso3,
                    "target_Year": int(target_year),
                    "match_ISO3": df.iloc[cpos]["ISO3"],
                    "match_Year": int(target_year),
                    "distance": float(dists[0, j]),
                    "rank": j + 1,
                }
            )

    return pd.DataFrame(rows)


def compute_cluster_divergence(
    matches: pd.DataFrame,
    cluster_vectors: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each target ↔ match pair, compute per-cluster divergence
    (difference in allocation fractions).
    """
    if matches.empty:
        return pd.DataFrame()

    cluster_cols = [c for c in cluster_vectors.columns if c not in ("ISO3", "Year")]

    results = []
    for _, row in matches.iterrows():
        t_vec = cluster_vectors[
            (cluster_vectors["ISO3"] == row["target_ISO3"])
            & (cluster_vectors["Year"] == row["target_Year"])
        ]
        m_vec = cluster_vectors[
            (cluster_vectors["ISO3"] == row["match_ISO3"])
            & (cluster_vectors["Year"] == row["match_Year"])
        ]

        if t_vec.empty or m_vec.empty:
            continue

        t_vals = t_vec[cluster_cols].iloc[0]
        m_vals = m_vec[cluster_cols].iloc[0]

        for cluster in cluster_cols:
            t_frac = float(t_vals[cluster])
            m_frac = float(m_vals[cluster])
            diff = t_frac - m_frac

            results.append(
                {
                    "target_ISO3": row["target_ISO3"],
                    "target_Year": int(row["target_Year"]),
                    "match_ISO3": row["match_ISO3"],
                    "match_Year": int(row["match_Year"]),
                    "match_rank": int(row["rank"]),
                    "risk_distance": float(row["distance"]),
                    "Cluster": cluster,
                    "target_frac": t_frac,
                    "match_frac": m_frac,
                    "frac_diff": diff,
                    "abs_diff": abs(diff),
                }
            )

    return pd.DataFrame(results)


def flag_divergences(div: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
    """Flag cluster allocations that differ by more than `threshold` (10 percentage points)."""
    if div.empty:
        return div

    div["divergent"] = div["abs_diff"] > threshold
    div["direction"] = np.where(
        div["frac_diff"] > threshold,
        "OVER-ALLOCATED",
        np.where(div["frac_diff"] < -threshold, "UNDER-ALLOCATED", "SIMILAR"),
    )
    return div


def summarize_divergences(div: pd.DataFrame) -> pd.DataFrame:
    """Average divergence across top matches per target country."""
    if div.empty:
        return pd.DataFrame()

    summary = (
        div.groupby(["target_ISO3", "target_Year", "Cluster"], as_index=False)
        .agg(
            mean_target_frac=("target_frac", "mean"),
            mean_match_frac=("match_frac", "mean"),
            mean_diff=("frac_diff", "mean"),
            mean_abs_diff=("abs_diff", "mean"),
            n_matches=("match_ISO3", "nunique"),
        )
    )

    summary["direction"] = np.where(
        summary["mean_diff"] > 0.05,
        "OVER-ALLOCATED",
        np.where(summary["mean_diff"] < -0.05, "UNDER-ALLOCATED", "SIMILAR"),
    )

    summary = summary.sort_values(
        ["target_ISO3", "mean_abs_diff"], ascending=[True, False]
    )
    return summary


def main():
    panel = load_panel(PANEL_PATH)
    sectors = load_sectors(SECTOR_PATH)
    cluster_vectors = build_cluster_vectors(sectors)

    latest_panel = int(panel["Year"].max())
    latest_sector = int(cluster_vectors["Year"].max())
    target_year = min(latest_panel, latest_sector)

    print(f"Panel goes to {latest_panel}, sectors to {latest_sector} → using {target_year}")

    matches = find_nearest_risk_profiles(panel, target_year=target_year, n_neighbors=3)
    print(f"\nMatches found: {len(matches)}")
    print(matches.head(10))

    div = compute_cluster_divergence(matches, cluster_vectors)
    if div.empty:
        raise RuntimeError(
            "No divergence rows produced. Likely missing cluster vectors for some ISO3-Year "
            "in the sectors file, or NAME_TO_ISO3 mapping dropped too many countries."
        )

    div = flag_divergences(div, threshold=0.10)
    print(f"\nDivergence rows: {len(div)}")

    summary = summarize_divergences(div)

    flagged = summary[summary["direction"] != "SIMILAR"].copy()
    print(f"\n{'='*70}")
    print(f"FLAGGED CLUSTER DIVERGENCES (target year = {target_year})")
    print(f"{'='*70}")

    for iso3 in sorted(flagged["target_ISO3"].unique()):
        country_flags = flagged[flagged["target_ISO3"] == iso3].head(5)
        print(f"\n  {iso3}:")
        for _, r in country_flags.iterrows():
            arrow = "▲" if r["direction"] == "OVER-ALLOCATED" else "▼"
            print(
                f"    {arrow} {r['Cluster']:40s} "
                f"target={r['mean_target_frac']:.1%}  peers={r['mean_match_frac']:.1%}  "
                f"diff={r['mean_diff']:+.1%}  [{r['direction']}]"
            )

    div_path = OUT_DIR / "cluster_divergence_detail.csv"
    div.to_csv(div_path, index=False)
    print(f"\nWrote: {div_path}")

    sum_path = OUT_DIR / "cluster_divergence_summary.csv"
    summary.to_csv(sum_path, index=False)
    print(f"Wrote: {sum_path}")

    match_path = OUT_DIR / "risk_profile_matches.csv"
    matches.to_csv(match_path, index=False)
    print(f"Wrote: {match_path}")

    return matches, div, summary


if __name__ == "__main__":
    matches, div, summary = main()