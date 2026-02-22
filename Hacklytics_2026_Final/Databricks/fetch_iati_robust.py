"""
fetch_iati_robust.py
Fetches IATI humanitarian project data with retry + backoff.
Saves results incrementally so a crash doesn't lose work.
"""
import requests
import json
import time
import os

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

OUTPUT_FILE = "data/iati_raw/all_crises.json"
os.makedirs("data/iati_raw", exist_ok=True)

# Load any partial results already saved
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        all_data = json.load(f)
    print(f"Resuming: {list(all_data.keys())} already fetched.")
else:
    all_data = {}


def fetch_with_retry(sector_code, rows=200, max_retries=5):
    params = {
        "q": f"sector_code:{sector_code} AND humanitarian:1",
        "fl": "iati_identifier,title_narrative,recipient_country_code,"
              "recipient_region_code,budget_value,transaction_value,"
              "activity_status_code,sector_code,sector_narrative",
        "rows": rows,
        "wt": "json",
    }
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    for attempt in range(max_retries):
        try:
            r = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            wait = 5 * (attempt + 1)
            print(f"  Error: {e}. Retry in {wait}s...")
            time.sleep(wait)
    print(f"  Failed after {max_retries} attempts.")
    return None


for crisis_name, code in CRISIS_SECTORS.items():
    if crisis_name in all_data:
        print(f"Skipping (cached): {crisis_name}")
        continue
    print(f"Fetching: {crisis_name} (sector {code})")
    result = fetch_with_retry(code)
    if result:
        docs = result.get("response", {}).get("docs", [])
        all_data[crisis_name] = docs
        print(f"  Got {len(docs)} projects")
    else:
        all_data[crisis_name] = []
        print(f"  Stored 0 (fetch failed)")
    # Save after every sector so we don't lose progress
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_data, f)
    time.sleep(3)  # polite delay between sectors

print(f"\nDone. All data saved to {OUTPUT_FILE}")
totals = {k: len(v) for k, v in all_data.items()}
print(json.dumps(totals, indent=2))
