# combine_data.py
# Merges IATI project data with the carlos.json supplementary dataset.
# Outputs: data/combined_projects.csv

import json
import pandas as pd

# ── ISO-2 country code → UN macro-region ─────────────────────────────────────
COUNTRY_REGION = {
    # Africa
    "AO":"Africa","BF":"Africa","BI":"Africa","BJ":"Africa","BW":"Africa",
    "CD":"Africa","CF":"Africa","CG":"Africa","CI":"Africa","CM":"Africa",
    "CV":"Africa","DJ":"Africa","DZ":"Africa","EG":"Africa","ER":"Africa",
    "ET":"Africa","GA":"Africa","GH":"Africa","GM":"Africa","GN":"Africa",
    "GQ":"Africa","GW":"Africa","KE":"Africa","KM":"Africa","LR":"Africa",
    "LS":"Africa","LY":"Africa","MA":"Africa","MG":"Africa","ML":"Africa",
    "MR":"Africa","MU":"Africa","MW":"Africa","MZ":"Africa","NA":"Africa",
    "NE":"Africa","NG":"Africa","RW":"Africa","SC":"Africa","SD":"Africa",
    "SL":"Africa","SN":"Africa","SO":"Africa","SS":"Africa","ST":"Africa",
    "SZ":"Africa","TD":"Africa","TG":"Africa","TN":"Africa","TZ":"Africa",
    "UG":"Africa","ZA":"Africa","ZM":"Africa","ZW":"Africa",
    # Middle East
    "AE":"Middle East","BH":"Middle East","IQ":"Middle East","IR":"Middle East",
    "JO":"Middle East","KW":"Middle East","LB":"Middle East","OM":"Middle East",
    "PS":"Middle East","QA":"Middle East","SA":"Middle East","SY":"Middle East",
    "YE":"Middle East",
    # Asia
    "AF":"Asia","AM":"Asia","AZ":"Asia","BD":"Asia","BT":"Asia","CN":"Asia",
    "GE":"Asia","ID":"Asia","IN":"Asia","KG":"Asia","KH":"Asia","KI":"Asia",
    "KP":"Asia","KZ":"Asia","LA":"Asia","LK":"Asia","MM":"Asia","MN":"Asia",
    "MV":"Asia","MY":"Asia","NP":"Asia","PG":"Asia","PH":"Asia","PK":"Asia",
    "SB":"Asia","TH":"Asia","TJ":"Asia","TL":"Asia","TM":"Asia","TR":"Asia",
    "UZ":"Asia","VN":"Asia","WS":"Asia",
    # Latin America
    "BO":"Latin America","BR":"Latin America","CL":"Latin America",
    "CO":"Latin America","CR":"Latin America","CU":"Latin America",
    "DO":"Latin America","EC":"Latin America","GT":"Latin America",
    "GY":"Latin America","HN":"Latin America","HT":"Latin America",
    "JM":"Latin America","MX":"Latin America","NI":"Latin America",
    "PA":"Latin America","PE":"Latin America","PY":"Latin America",
    "SV":"Latin America","TT":"Latin America","UY":"Latin America",
    "VE":"Latin America",
    # Europe
    "AL":"Europe","BA":"Europe","BY":"Europe","HR":"Europe","KV":"Europe",
    "MD":"Europe","ME":"Europe","MK":"Europe","RS":"Europe","RU":"Europe",
    "UA":"Europe","XK":"Europe",
    # North America
    "CA":"North America","US":"North America",
}


def get_region(codes):
    """Return macro-region for a list of ISO-2 country codes."""
    if not codes:
        return "Unknown"
    code = codes[0] if isinstance(codes, list) else codes
    return COUNTRY_REGION.get(str(code).upper(), "Other")


# ── Load IATI data ────────────────────────────────────────────────────────────
with open("data/iati_raw/all_crises.json") as f:
    iati_data = json.load(f)

rows = []
for crisis_type, projects in iati_data.items():
    for p in projects:
        country_codes = p.get("recipient_country_code", [])
        country = country_codes[0] if country_codes else "Unknown"
        rows.append({
            "crisis_type": crisis_type,
            "project_id":  p.get("iati_identifier"),
            "title":       (p.get("title_narrative") or [""])[0][:120],
            "country":     country,
            "region":      get_region(country_codes),
            "budget":      sum(p.get("budget_value") or [0]),
            "source":      "IATI",
        })

# ── Load carlos supplementary dataset ────────────────────────────────────────
with open("data/carlos.json") as f:
    carlos_data = json.load(f)

for item in carlos_data:
    rows.append({
        "crisis_type": item.get("crisis_type", "Unknown"),
        "project_id":  item.get("id"),
        "title":       item.get("title", "")[:120],
        "country":     item.get("country", "Unknown"),
        "region":      item.get("region", "Unknown"),
        "budget":      item.get("budget", 0),
        "source":      "Carlos",
    })

df = pd.DataFrame(rows)
df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0)
df.to_csv("data/combined_projects.csv", index=False)

print(f"Combined dataset: {df.shape[0]} projects, {df.shape[1]} columns")
print(f"\nProjects by source:\n{df['source'].value_counts().to_string()}")
print(f"\nProjects by crisis type:\n{df.groupby('crisis_type')['project_id'].count().sort_values(ascending=False).to_string()}")
print(f"\nProjects by region:\n{df.groupby('region')['project_id'].count().sort_values(ascending=False).to_string()}")
