# Crisis Funding Analysis

A Python data analysis tool that maps humanitarian project funding across **12 crisis types** and **129 countries**, revealing how aid is distributed across different regions and crisis categories.

## What This Does

- Pulls live project data from the **IATI (International Aid Transparency Initiative) API** — the global standard for aid transparency
- Merges it with a supplementary curated dataset (`carlos.json`) of 100 additional projects
- Generates **5 visualizations** showing how $15.3B in documented funding flows across crisis types and regions
- Identifies which crisis types and regions are most and least represented

## Crisis Types Covered (12)

| Crisis Type | Sector Code |
|---|---|
| Food Security | 52010 |
| Refugees | 93010 |
| Natural Disaster | 74010 |
| Conflict/War | 15220 |
| Disease Outbreak | 12110 |
| Water/Sanitation | 14010 |
| Mental Health | 12220 |
| Education Emergency | 11120 |
| Poverty/Economic | 16010 |
| Climate/Environment | 41010 |
| Drought | 43040 |
| Displacement | 93020 |

## Charts Generated

| Chart | File | Description |
|---|---|---|
| 1 | `charts/1_funding_by_crisis_type.png` | Total documented funding per crisis type |
| 2 | `charts/2_funding_by_region_and_crisis.png` | Stacked funding breakdown: which regions get aid for each crisis |
| 3 | `charts/3_project_count_heatmap.png` | Heatmap of project count — crisis type × region |
| 4 | `charts/4_top_countries_by_funding.png` | Top 20 recipient countries by total budget |
| 5 | `charts/5_source_comparison.png` | IATI vs supplementary dataset — project coverage comparison |

## Quick Start

```bash
# 1. Install dependencies
pip install requests pandas matplotlib

# 2. Fetch IATI data (one-time, ~40 seconds)
python fetch_iati_robust.py

# 3. Merge IATI + Carlos datasets
python combine_data.py

# 4. Generate all 5 charts
python visualize_combined.py
```

Charts are saved to the `charts/` folder.

## Project Structure

```
crisis_funding/
├── fetch_iati_robust.py     # Fetches IATI API data with retry logic
├── fetch_iati.py            # Original simple fetcher
├── combine_data.py          # Merges IATI + carlos.json → combined_projects.csv
├── visualize_combined.py    # Generates 5 regional analysis charts  ← main viz
├── visualize.py             # Original 3-chart version
├── visualize_gaps.py        # FTS-based funding gap analysis (5 topics)
├── topic_scanner.py         # FTS API scanner for vulnerable populations
├── data/
│   ├── carlos.json          # Supplementary project dataset (100 projects)
│   ├── iati_raw/
│   │   └── all_crises.json  # Fetched IATI data (auto-generated)
│   └── combined_projects.csv # Merged output (auto-generated)
└── charts/                  # All chart outputs (auto-generated)
```

## Data Sources

- **IATI Datastore API** — https://developer.iatistandard.org/ — open humanitarian aid database covering 1,800+ projects
- **carlos.json** — 100 curated projects manually compiled across all 12 crisis types
- **FTS API** — UN OCHA Financial Tracking Service (used in `topic_scanner.py` for gap analysis)

## Dataset Summary

| Metric | Value |
|---|---|
| Total projects | 1,998 |
| Total documented funding | $15.3B |
| Countries represented | 129 |
| Crisis types | 12 |
| Regions covered | Africa, Middle East, Asia, Latin America, Europe |

## Key Findings

- **Disease Outbreak** has the highest documented funding ($8.1B) driven by large global health programs
- **Africa** receives the largest share of humanitarian project funding across all crisis types
- **Mental Health** and **Displacement** are the most under-documented crisis types by project count
- **Water/Sanitation** shows broad regional coverage — projects in all 5 major regions

## Known Issues / Limitations

- The IATI API key in `fetch_iati.py` is hardcoded — move to an environment variable before sharing
- ~525 of 1,998 projects have no documented budget (reported as $0)
- `Displacement` sector (93020) returned 0 IATI results — covered only by carlos.json
- The carlos.json dataset is a curated supplement, not an official data source

## Requirements

```
requests
pandas
matplotlib
```

Install with: `pip install requests pandas matplotlib`
