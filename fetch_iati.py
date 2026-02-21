# fetch_iati.py
import requests
import json
import time

API_KEY = "156133b7ccee4e2eb6731fdaf1279290"
BASE_URL = "https://api.iatistandard.org/datastore/activity/select"

CRISIS_SECTORS = {
    "Food Security":       "52010",
    "Refugees":            "93010",
    "Natural Disaster":    "74010",
    "Conflict/War":        "15220",
    "Disease Outbreak":    "12110",
    "Water/Sanitation":    "14010",
    "Mental Health":       "12220",
    "Education Emergency": "11120",
    "Poverty/Economic":    "16010",
    "Climate/Environment": "41010",
    "Drought":             "43040",
    "Displacement":        "93020",
}

def fetch_crisis_projects(sector_code, rows=200):
    params = {
        "q": f"sector_code:{sector_code} AND humanitarian:1",
        "fl": "iati_identifier,title_narrative,recipient_country_code,"
              "recipient_region_code,budget_value,transaction_value,"
              "activity_status_code,sector_code,sector_narrative",
        "rows": rows,
        "wt": "json"
    }
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    r = requests.get(BASE_URL, params=params, headers=headers)
    r.raise_for_status()
    return r.json()

all_data = {}
for crisis_name, code in CRISIS_SECTORS.items():
    print(f"Fetching: {crisis_name} (sector {code})")
    result = fetch_crisis_projects(code)
    all_data[crisis_name] = result.get("response", {}).get("docs", [])
    time.sleep(0.5)  # be polite to the API

with open("data/iati_raw/all_crises.json", "w") as f:
    json.dump(all_data, f)
print("Done.")