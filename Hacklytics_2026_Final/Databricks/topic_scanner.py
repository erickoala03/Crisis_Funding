"""
=============================================================================
TOPIC SCANNER — Which "overlooked crisis" has the most dramatic data gap?
=============================================================================
Run this against FTS API data to make a DATA-DRIVEN decision on your topic.

Usage:
  pip install requests pandas --break-system-packages
  python topic_scanner.py

This will:
1. Pull 2000+ flows from FTS API (2023 + 2024)
2. Scan descriptions for each candidate topic
3. Show you exactly which topic produces the most shocking stat
4. Help you pick your hackathon angle in 20 minutes
=============================================================================
"""

import requests
import json
import re
import time
from collections import defaultdict

# =============================================================================
# CANDIDATE TOPICS — each with keyword lists and benchmark stats
# =============================================================================

TOPICS = {
    "mental_health": {
        "label": "Mental Health & Psychosocial Support",
        "benchmark": "20-30% of crisis-affected populations have PTSD/depression/anxiety (WHO)",
        "benchmark_pct": 0.25,  # 25% midpoint estimate
        "published_funding_gap": "MHPSS receives ~0.3% of international health aid (WHO/Lancet)",
        "high_confidence": [
            "mental health",
            "mhpss",
            "psychosocial support",
            "psychosocial services",
            "psychological support",
            "psychological first aid",
            "ptsd",
            "post-traumatic",
            "trauma counseling",
            "trauma counselling",
            "psychiatric",
            "mental well-being",
            "mental wellbeing",
            "mental wellness",
            "psychological trauma",
            "psychological counseling",
            "psychological counselling",
            "santé mentale",          # French
            "soutien psychosocial",   # French
            "salud mental",           # Spanish
            "apoyo psicosocial",      # Spanish
        ],
        "medium_confidence": [
            "psychosocial",
            "counseling",
            "counselling",
            "trauma",
            "psychological",
            "well-being",
            "wellbeing",
            "stress management",
            "emotional support",
            "grief",
            "bereavement",
            "suicide prevention",
            "depression",
            "anxiety",
        ]
    },

    "elderly": {
        "label": "Older Persons / Aging Populations",
        "benchmark": "~12% of crisis-affected populations are over 60 (UN Pop Division)",
        "benchmark_pct": 0.12,
        "published_funding_gap": "<1% of humanitarian funding targets older people (HelpAge Intl)",
        "high_confidence": [
            "older persons",
            "older people",
            "older adults",
            "older men and women",
            "older women and men",
            "elderly",
            "ageing population",
            "aging population",
            "aged population",
            "geriatric",
            "old age",
            "older refugees",
            "older displaced",
            "personnes âgées",        # French
            "personas mayores",       # Spanish
            "tercera edad",           # Spanish
        ],
        "medium_confidence": [
            "age-friendly",
            "age-inclusive",
            "60 years",
            "over 60",
            "65 years",
            "over 65",
            "senior citizens",
            "seniors",
            "dementia",
            "age-disaggregated",
            "intergenerational",
            "helpage",
        ]
    },

    "gbv": {
        "label": "Gender-Based Violence",
        "benchmark": "1 in 3 women experience violence; rates spike 3-5x in crises (UN Women)",
        "benchmark_pct": 0.33,
        "published_funding_gap": "GBV receives ~0.2% of humanitarian funding (UNFPA)",
        "high_confidence": [
            "gender-based violence",
            "gender based violence",
            "sexual violence",
            "sexual exploitation",
            "sexual abuse",
            "sgbv",
            "gbv prevention",
            "gbv response",
            "gbv services",
            "violence against women",
            "violence against girls",
            "intimate partner violence",
            "domestic violence",
            "rape",
            "sexual assault",
            "violences basées sur le genre",   # French
            "violencia de género",              # Spanish
            "violencia sexual",                 # Spanish
        ],
        "medium_confidence": [
            "gbv",
            "gender violence",
            "protection from violence",
            "women's safety",
            "safe spaces for women",
            "women-friendly spaces",
            "dignity kits",
            "clinical management of rape",
            "cmr",
            "survivor support",
            "survivor assistance",
            "referral pathways",
        ]
    },

    "education_in_emergencies": {
        "label": "Education in Emergencies",
        "benchmark": "~50% of crisis-affected children are out of school (UNESCO/ECW)",
        "benchmark_pct": 0.20,  # ~20% of crisis population are school-age children affected
        "published_funding_gap": "Education gets 2-4% of humanitarian funding (ECW)",
        "high_confidence": [
            "education in emergencies",
            "emergency education",
            "education cannot wait",
            "ecw",
            "school in crisis",
            "education for refugees",
            "refugee education",
            "displaced children education",
            "temporary learning",
            "temporary schools",
            "éducation en situation d'urgence",  # French
            "educación en emergencias",           # Spanish
        ],
        "medium_confidence": [
            "education",
            "school",
            "learning",
            "teacher training",
            "school supplies",
            "classroom",
            "literacy",
            "numeracy",
            "back to school",
            "child-friendly spaces",
            "early childhood development",
            "ecd",
        ]
    },

    "disability": {
        "label": "Persons with Disabilities",
        "benchmark": "15% of global population; higher in crisis zones (WHO)",
        "benchmark_pct": 0.15,
        "published_funding_gap": "<1% of humanitarian funding is disability-inclusive (IDA)",
        "high_confidence": [
            "persons with disabilities",
            "people with disabilities",
            "disability inclusion",
            "disability-inclusive",
            "disabled persons",
            "disabled people",
            "wheelchair",
            "assistive technology",
            "assistive devices",
            "sign language",
            "visual impairment",
            "hearing impairment",
            "physical disability",
            "intellectual disability",
            "psychosocial disability",
            "crpd",
            "personnes handicapées",     # French
            "personas con discapacidad", # Spanish
        ],
        "medium_confidence": [
            "disability",
            "disabilities",
            "impairment",
            "accessible",
            "accessibility",
            "inclusive programming",
            "universal design",
            "mobility challenges",
            "special needs",
            "prosthetic",
            "orthotic",
            "rehabilitation",
            "washington group",
        ]
    },
}


# =============================================================================
# FTS API PULLER — Pages through all flows
# =============================================================================

def fetch_fts_flows(year=2024, max_pages=20, per_page=100):
    """
    Pull flows from FTS API with pagination.
    Returns list of flow dicts.
    """
    all_flows = []
    base_url = "https://api.hpc.tools/v1/public/fts/flow"

    for page in range(1, max_pages + 1):
        params = {
            "year": year,
            "limit": per_page,
            "page": page,
        }

        try:
            print(f"  Fetching page {page} for {year}...", end=" ")
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            flows = data.get("data", {}).get("flows", [])
            if not flows:
                print("no more data")
                break

            all_flows.extend(flows)
            print(f"got {len(flows)} flows (total: {len(all_flows)})")

            # Check if there's a next page
            meta = data.get("meta", {})
            if not meta.get("nextLink"):
                break

            # Be nice to the API
            time.sleep(0.3)

        except Exception as e:
            print(f"ERROR: {e}")
            break

    return all_flows


# =============================================================================
# TOPIC SCANNER — Runs all topics against flow descriptions
# =============================================================================

def scan_flows(flows, topics=TOPICS):
    """
    Scan all flow descriptions for each candidate topic.
    Returns results dict with counts and funding amounts.
    """
    results = {}

    for topic_key, topic in topics.items():
        results[topic_key] = {
            "label": topic["label"],
            "benchmark": topic["benchmark"],
            "benchmark_pct": topic["benchmark_pct"],
            "published_funding_gap": topic["published_funding_gap"],
            "high_confidence_matches": 0,
            "any_match": 0,
            "high_confidence_funding": 0,
            "any_match_funding": 0,
            "sample_matches": [],
            "matched_countries": defaultdict(float),
            "matched_sectors": defaultdict(int),
        }

    total_flows = 0
    total_funding = 0
    flows_with_text = 0
    flows_with_funding = 0

    for flow in flows:
        desc = (flow.get("description") or "").strip()
        amount = flow.get("amountUSD") or 0
        parked = flow.get("fullParkedAmountUSD") or 0
        effective_amount = amount if amount > 0 else parked

        total_flows += 1
        total_funding += effective_amount

        if desc:
            flows_with_text += 1
        if effective_amount > 0:
            flows_with_funding += 1

        if not desc:
            continue

        desc_lower = desc.lower()

        # Extract country from destination objects
        country = "Unknown"
        sector = "Unknown"
        for obj in flow.get("destinationObjects", []):
            if obj.get("type") == "Location":
                country = obj.get("name", "Unknown")
            if obj.get("type") == "GlobalCluster":
                sector = obj.get("name", "Unknown")

        # Scan each topic
        for topic_key, topic in topics.items():
            r = results[topic_key]

            high_hit = any(term.lower() in desc_lower for term in topic["high_confidence"])
            med_hit = any(term.lower() in desc_lower for term in topic["medium_confidence"])

            if high_hit:
                r["high_confidence_matches"] += 1
                r["high_confidence_funding"] += effective_amount
                r["matched_countries"][country] += effective_amount
                r["matched_sectors"][sector] += 1

                if len(r["sample_matches"]) < 5:
                    r["sample_matches"].append({
                        "desc": desc[:200],
                        "amount": effective_amount,
                        "country": country,
                    })

            if high_hit or med_hit:
                r["any_match"] += 1
                r["any_match_funding"] += effective_amount

    return results, total_flows, total_funding, flows_with_text, flows_with_funding


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def display_results(results, total_flows, total_funding, flows_with_text, flows_with_funding):
    """
    Show the comparison table and recommend the best topic.
    """
    print("\n" + "=" * 80)
    print("FTS DATA OVERVIEW")
    print("=" * 80)
    print(f"Total flows analyzed:      {total_flows:,}")
    print(f"Flows with description:    {flows_with_text:,} ({flows_with_text/total_flows*100:.1f}%)")
    print(f"Flows with funding > $0:   {flows_with_funding:,}")
    print(f"Total funding tracked:     ${total_funding:,.0f}")

    print("\n" + "=" * 80)
    print("TOPIC COMPARISON — WHICH GAP IS MOST DRAMATIC?")
    print("=" * 80)

    # Sort by "gap score" — bigger gap between benchmark and actual = better story
    scored = []
    for key, r in results.items():
        pct_high = (r["high_confidence_matches"] / total_flows * 100) if total_flows > 0 else 0
        pct_any = (r["any_match"] / total_flows * 100) if total_flows > 0 else 0
        funding_pct = (r["high_confidence_funding"] / total_funding * 100) if total_funding > 0 else 0

        # Gap score = benchmark percentage - actual percentage (bigger = more dramatic)
        gap_score = r["benchmark_pct"] * 100 - pct_high

        scored.append((key, r, pct_high, pct_any, funding_pct, gap_score))

    # Sort by gap score descending
    scored.sort(key=lambda x: x[5], reverse=True)

    for rank, (key, r, pct_high, pct_any, funding_pct, gap_score) in enumerate(scored, 1):
        print(f"\n{'─' * 80}")
        print(f"  #{rank}  {r['label']}")
        print(f"{'─' * 80}")
        print(f"  Benchmark:              {r['benchmark']}")
        print(f"  Published funding gap:  {r['published_funding_gap']}")
        print(f"")
        print(f"  HIGH-CONFIDENCE matches:  {r['high_confidence_matches']:,} / {total_flows:,} flows  ({pct_high:.2f}%)")
        print(f"  ANY mention matches:      {r['any_match']:,} / {total_flows:,} flows  ({pct_any:.2f}%)")
        print(f"  Funding in matched flows: ${r['high_confidence_funding']:,.0f}  ({funding_pct:.2f}% of total)")
        print(f"")
        print(f"  >>> GAP SCORE: Population is {r['benchmark_pct']*100:.0f}% but funding mentions are {pct_high:.2f}%")
        print(f"  >>> RATIO: {r['benchmark_pct']*100/max(pct_high, 0.01):.0f}x underrepresented")
        print(f"")

        if r["sample_matches"]:
            print(f"  Sample matches:")
            for m in r["sample_matches"][:3]:
                print(f"    [{m['country']}] ${m['amount']:,.0f}")
                print(f"    \"{m['desc'][:150]}...\"")
                print()

        if r["matched_countries"]:
            top_countries = sorted(r["matched_countries"].items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top countries with mentions:")
            for c, amt in top_countries:
                print(f"    {c}: ${amt:,.0f}")

    # RECOMMENDATION
    winner = scored[0]
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"\n  GO WITH: {winner[1]['label']}")
    print(f"")
    print(f"  Why: {winner[1]['benchmark_pct']*100:.0f}% of population affected,")
    print(f"       but only {winner[2]:.2f}% of funding flows mention it.")
    print(f"       That's a {winner[5]:.1f} percentage-point gap.")
    print(f"       Ratio: {winner[1]['benchmark_pct']*100/max(winner[2], 0.01):.0f}x underrepresented.")
    print(f"")
    print(f"  Your one-liner for judges:")
    print(f"  \"We analyzed ${total_funding/1e9:.0f}B in humanitarian funding and found that")
    print(f"   {winner[1]['label'].lower()} — affecting {winner[1]['benchmark_pct']*100:.0f}% of crisis")
    print(f"   populations — appears in only {winner[2]:.1f}% of project descriptions.\"")
    print(f"")
    print(f"  Published validation: {winner[1]['published_funding_gap']}")
    print("=" * 80)

    return scored


# =============================================================================
# BONUS: Test against the sample data you already have
# =============================================================================

def test_with_sample():
    """
    Quick test using the 5 flows you already pulled.
    Just to verify the scanner works before hitting the API.
    """
    sample_flows = [
        {
            "description": "The project is expected to improve food security and increase the resilience of livelihood of livestock farmers to animal disease threats in Pakistan.",
            "amountUSD": 6598917,
            "destinationObjects": [{"type": "Location", "name": "Pakistan"}]
        },
        {
            "description": "Vulnerable women and Men in targeted refugee and Lebanese communities sustainably improve their skills, capacities and livelihood opportunities",
            "amountUSD": 5513367,
            "destinationObjects": [{"type": "Location", "name": "Lebanon"}]
        },
        {
            "description": "Syria 3RP Livelihoods Promoting Decent Work for Syrian Refugees and Host Communities in Türkiye through Investment in Skills",
            "amountUSD": 10631439,
            "destinationObjects": [{"type": "Location", "name": "Türkiye"}]
        },
        {
            "description": "Appui au renforcement des capacités de résilience des populations vulnérables du Batha",
            "amountUSD": 1878460,
            "destinationObjects": [{"type": "Location", "name": "Chad"}]
        },
        {
            "description": "Emergency health and nutrition response including mental health and psychosocial support services for conflict-affected populations",
            "amountUSD": 3500000,
            "destinationObjects": [{"type": "Location", "name": "Sudan"}, {"type": "GlobalCluster", "name": "Health"}]
        },
    ]

    print("=" * 80)
    print("QUICK TEST WITH SAMPLE DATA (5 flows)")
    print("=" * 80)
    results, total, funding, with_text, with_funding = scan_flows(sample_flows)
    display_results(results, total, funding, with_text, with_funding)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        test_with_sample()
        sys.exit(0)

    print("=" * 80)
    print("DISABILITY FUNDING GAP — TOPIC SCANNER")
    print("Which overlooked crisis has the most dramatic data gap?")
    print("=" * 80)

    # Pull data from FTS API
    all_flows = []

    for year in [2023, 2024]:
        print(f"\n--- Fetching {year} ---")
        flows = fetch_fts_flows(year=year, max_pages=15, per_page=100)
        all_flows.extend(flows)
        print(f"Total so far: {len(all_flows)} flows")

    if not all_flows:
        print("\nERROR: Could not fetch FTS data. Check your internet connection.")
        print("Running test with sample data instead...\n")
        test_with_sample()
        sys.exit(1)

    # Scan all topics
    print(f"\nScanning {len(all_flows)} flows across {len(TOPICS)} topics...")
    results, total, funding, with_text, with_funding = scan_flows(all_flows)

    # Display comparison
    scored = display_results(results, total, funding, with_text, with_funding)

    # Save raw results
    output = {
        "total_flows": total,
        "total_funding": funding,
        "flows_with_text": with_text,
        "topics": {}
    }
    for key, r, pct_high, pct_any, funding_pct, gap_score in scored:
        output["topics"][key] = {
            "label": r["label"],
            "high_confidence_matches": r["high_confidence_matches"],
            "any_match": r["any_match"],
            "pct_high": pct_high,
            "pct_any": pct_any,
            "funding_pct": funding_pct,
            "gap_score": gap_score,
        }

    with open("topic_scan_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nRaw results saved to topic_scan_results.json")
