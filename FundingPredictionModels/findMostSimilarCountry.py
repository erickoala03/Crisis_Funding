"""
Cluster Divergence Model
========================
For each CBPF country (latest year), find the most similar country-year by
risk profile (SAME YEAR ONLY), then compare where their cluster-level funding
allocations diverge.

Outputs per country-match pair:
  - 2 clusters where funding allocation is MOST SIMILAR
  - 2 clusters where funding allocation MOST DIFFERS

Writes:
  - cluster_highlights.json   (nested, for web app)
  - cluster_highlights.csv    (flat, for dashboards)
  - risk_profile_matches.csv  (raw match table)

Expects:
  - MASTER_PANEL_FINAL.xlsx        (risk features by ISO3 × Year)
  - SectorsOverview_Combined.csv   (cluster allocations by country × year)

Usage:
  python cluster_divergence_model.py
"""

import json
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


# ══════════════════════════════════════════════════════════════════
# Loaders
# ══════════════════════════════════════════════════════════════════

def load_panel(path: Path) -> pd.DataFrame:
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
    print(f"Loading sectors: {path}")
    sec = pd.read_csv(path)

    sec = sec.rename(columns={
        "CBPF Name": "Country",
        "Total Allocations": "Budget",
    })

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
        .sum().rename(columns={"Budget": "Total"})
    )

    sec = sec.merge(totals, on=["ISO3", "Year"], how="left")
    sec["cluster_frac"] = sec["Budget"] / (sec["Total"] + 1e-9)

    return sec


def build_cluster_vectors(sec: pd.DataFrame) -> pd.DataFrame:
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


# ══════════════════════════════════════════════════════════════════
# Risk-profile matching (same year only)
# ══════════════════════════════════════════════════════════════════

def find_nearest_risk_profiles(
    panel: pd.DataFrame,
    target_year: int | None = None,
    n_neighbors: int = 3,
) -> pd.DataFrame:
    feats = [c for c in RISK_FEATURES if c in panel.columns]
    if not feats:
        raise ValueError("No risk features found in panel")

    df = panel.dropna(subset=feats, how="all").copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    if target_year is None:
        target_year = int(df["Year"].max())

    df = df[df["Year"] == target_year].copy()
    if df.empty:
        raise ValueError(f"No rows in panel for target_year={target_year}")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    Z = pipe.fit_transform(df[feats])

    rows = []
    for pos in range(len(df)):
        t_iso3 = df.iloc[pos]["ISO3"]

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
            rows.append({
                "target_ISO3": t_iso3,
                "target_Year": int(target_year),
                "match_ISO3": df.iloc[cpos]["ISO3"],
                "match_Year": int(target_year),
                "distance": float(dists[0, j]),
                "rank": j + 1,
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# Core: 2 similar + 2 divergent per pair
# ══════════════════════════════════════════════════════════════════

def compute_pair_highlights(
    matches: pd.DataFrame,
    cluster_vectors: pd.DataFrame,
    sectors: pd.DataFrame,
    n_similar: int = 2,
    n_divergent: int = 2,
) -> list[dict]:
    """
    For each target ↔ match pair, find:
      - top n_similar clusters with smallest abs(frac_diff)
      - top n_divergent clusters with largest abs(frac_diff)

    Only considers clusters where AT LEAST ONE side has >1% allocation
    (filters out noise from near-zero clusters).

    Returns a list of dicts ready for JSON / web app.
    """
    if matches.empty:
        return []

    cluster_cols = [c for c in cluster_vectors.columns if c not in ("ISO3", "Year")]

    # Pre-build budget lookup: (ISO3, Year) → {Cluster: Budget}
    budget_lookup: dict[tuple, dict] = {}
    for _, r in sectors.iterrows():
        key = (r["ISO3"], int(r["Year"]))
        if key not in budget_lookup:
            budget_lookup[key] = {}
        budget_lookup[key][r["Cluster"]] = float(r["Budget"])

    results = []

    for _, row in matches.iterrows():
        t_iso3 = row["target_ISO3"]
        t_year = int(row["target_Year"])
        m_iso3 = row["match_ISO3"]
        m_year = int(row["match_Year"])

        t_vec = cluster_vectors[
            (cluster_vectors["ISO3"] == t_iso3) &
            (cluster_vectors["Year"] == t_year)
        ]
        m_vec = cluster_vectors[
            (cluster_vectors["ISO3"] == m_iso3) &
            (cluster_vectors["Year"] == m_year)
        ]

        if t_vec.empty or m_vec.empty:
            continue

        t_vals = t_vec[cluster_cols].iloc[0]
        m_vals = m_vec[cluster_cols].iloc[0]

        t_budgets = budget_lookup.get((t_iso3, t_year), {})
        m_budgets = budget_lookup.get((m_iso3, m_year), {})

        # Build per-cluster comparison — skip if both sides <1%
        comparisons = []
        for cluster in cluster_cols:
            t_frac = float(t_vals[cluster])
            m_frac = float(m_vals[cluster])

            if t_frac < 0.01 and m_frac < 0.01:
                continue

            diff = t_frac - m_frac
            comparisons.append({
                "cluster": cluster,
                "target_frac": round(t_frac, 4),
                "match_frac": round(m_frac, 4),
                "target_budget_usd": round(t_budgets.get(cluster, 0), 2),
                "match_budget_usd": round(m_budgets.get(cluster, 0), 2),
                "frac_diff": round(diff, 4),
                "abs_diff": round(abs(diff), 4),
            })

        if not comparisons:
            continue

        comp_df = pd.DataFrame(comparisons).sort_values("abs_diff")

        # ── Most similar: smallest abs_diff ──
        similar = comp_df.head(n_similar).to_dict("records")
        for s in similar:
            s["direction"] = "SIMILAR"

        # ── Most divergent: largest abs_diff ──
        divergent = (
            comp_df.tail(n_divergent)
            .sort_values("abs_diff", ascending=False)
            .to_dict("records")
        )
        for d in divergent:
            if d["frac_diff"] > 0:
                d["direction"] = "OVER-ALLOCATED"
            elif d["frac_diff"] < 0:
                d["direction"] = "UNDER-ALLOCATED"
            else:
                d["direction"] = "SIMILAR"

        results.append({
            "target_ISO3": t_iso3,
            "target_Year": t_year,
            "match_ISO3": m_iso3,
            "match_Year": m_year,
            "match_rank": int(row["rank"]),
            "risk_distance": round(float(row["distance"]), 4),
            "similar_clusters": similar,
            "divergent_clusters": divergent,
        })

    return results


# ══════════════════════════════════════════════════════════════════
# Flatten for CSV / dashboards
# ══════════════════════════════════════════════════════════════════

def flatten_for_csv(highlights: list[dict]) -> pd.DataFrame:
    """
    Each row = one cluster entry (SIMILAR or DIVERGENT) for a country pair.
    """
    rows = []
    for h in highlights:
        base = {
            "target_ISO3": h["target_ISO3"],
            "target_Year": h["target_Year"],
            "match_ISO3": h["match_ISO3"],
            "match_Year": h["match_Year"],
            "match_rank": h["match_rank"],
            "risk_distance": h["risk_distance"],
        }

        for s in h["similar_clusters"]:
            rows.append({
                **base,
                "comparison_type": "SIMILAR",
                "cluster": s["cluster"],
                "target_frac": s["target_frac"],
                "match_frac": s["match_frac"],
                "target_budget_usd": s["target_budget_usd"],
                "match_budget_usd": s["match_budget_usd"],
                "frac_diff": s["frac_diff"],
                "abs_diff": s["abs_diff"],
                "direction": "SIMILAR",
            })

        for d in h["divergent_clusters"]:
            rows.append({
                **base,
                "comparison_type": "DIVERGENT",
                "cluster": d["cluster"],
                "target_frac": d["target_frac"],
                "match_frac": d["match_frac"],
                "target_budget_usd": d["target_budget_usd"],
                "match_budget_usd": d["match_budget_usd"],
                "frac_diff": d["frac_diff"],
                "abs_diff": d["abs_diff"],
                "direction": d["direction"],
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

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

    # ── Core output: 2 similar + 2 divergent per pair ──
    highlights = compute_pair_highlights(
        matches, cluster_vectors, sectors,
        n_similar=2, n_divergent=2,
    )

    print(f"Highlight entries: {len(highlights)}")

    # ── Print summary ──
    print(f"\n{'='*70}")
    print(f"  CLUSTER COMPARISON HIGHLIGHTS  (year = {target_year})")
    print(f"{'='*70}")

    for h in highlights:
        print(f"\n  {h['target_ISO3']} vs {h['match_ISO3']}  "
              f"(rank {h['match_rank']}, dist={h['risk_distance']:.3f})")

        print("    SIMILAR FUNDING:")
        for s in h["similar_clusters"]:
            print(f"      • {s['cluster']:35s}  "
                  f"target={s['target_frac']:.1%}  match={s['match_frac']:.1%}  "
                  f"Δ={s['frac_diff']:+.1%}")

        print("    DIVERGENT FUNDING:")
        for d in h["divergent_clusters"]:
            arrow = "▲" if d["direction"] == "OVER-ALLOCATED" else "▼"
            print(f"      {arrow} {d['cluster']:35s}  "
                  f"target={d['target_frac']:.1%}  match={d['match_frac']:.1%}  "
                  f"Δ={d['frac_diff']:+.1%}  [{d['direction']}]")

    # ── Write outputs ──

    # 1) JSON for web app (nested structure)
    json_path = OUT_DIR / "cluster_highlights.json"
    with open(json_path, "w") as f:
        json.dump(highlights, f, indent=2)
    print(f"\nWrote: {json_path}")

    # 2) Flat CSV for analysis / dashboards
    flat = flatten_for_csv(highlights)
    csv_path = OUT_DIR / "cluster_highlights.csv"
    flat.to_csv(csv_path, index=False)
    print(f"Wrote: {csv_path}")

    # 3) Risk profile matches
    match_path = OUT_DIR / "risk_profile_matches.csv"
    matches.to_csv(match_path, index=False)
    print(f"Wrote: {match_path}")

    return matches, highlights, flat


if __name__ == "__main__":
    matches, highlights, flat = main()