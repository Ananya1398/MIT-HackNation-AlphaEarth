"""
NFIP + Open-Meteo ERA5 Climate Markers (Event-Year) + Event Type Classification (Filtered)
-----------------------------------------------------------------------------------------
Removes rows with missing floodEvent and adds event_type column right after it.
Now uses year-specific Open-Meteo daily ERA5 reanalysis data instead of NASA POWER.
"""

import requests
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime

CURRENT_YEAR = datetime.now().year

# FEMA NFIP endpoint
NFIP_URL = "https://www.fema.gov/api/open/v2/FimaNfipClaims"

# Output folder
OUT_DIR = os.path.join(os.getcwd(), "insurance")
os.makedirs(OUT_DIR, exist_ok=True)

# Coordinates (approximate centers of states)
STATE_COORDS = {
    "AL": (32.8, -86.8), "AR": (34.9, -92.3), "AZ": (33.7, -111.4),
    "CA": (36.1, -119.7), "CO": (39.0, -105.3), "CT": (41.6, -72.7),
    "DE": (39.3, -75.5), "FL": (27.7, -81.6), "GA": (33.0, -83.6),
    "HI": (21.1, -157.5), "IA": (42.0, -93.2), "ID": (44.2, -114.5),
    "IL": (40.3, -88.9), "IN": (39.8, -86.2), "KS": (38.5, -96.7),
    "KY": (37.6, -84.6), "LA": (31.1, -91.8), "MA": (42.2, -71.5),
    "MD": (39.0, -76.8), "ME": (44.7, -69.3), "MI": (43.3, -84.5),
    "MN": (45.6, -93.9), "MO": (38.4, -92.2), "MS": (32.7, -89.6),
    "MT": (46.9, -110.4), "NC": (35.6, -79.8), "ND": (47.5, -99.7),
    "NE": (41.1, -98.2), "NH": (43.4, -71.5), "NJ": (40.2, -74.5),
    "NM": (34.8, -106.2), "NV": (38.3, -117.0), "NY": (42.1, -74.9),
    "OH": (40.3, -82.7), "OK": (35.5, -96.9), "OR": (44.5, -122.0),
    "PA": (40.5, -77.2), "RI": (41.6, -71.5), "SC": (33.8, -80.9),
    "SD": (44.2, -99.4), "TN": (35.7, -86.6), "TX": (31.0, -97.5),
    "UT": (40.1, -111.8), "VA": (37.7, -78.1), "VT": (44.0, -72.7),
    "WA": (47.4, -121.4), "WI": (44.2, -89.6), "WV": (38.4, -80.9),
    "WY": (42.7, -107.3), "PR": (18.2, -66.5)
}


# -------------------------------
# STEP 1: Fetch FEMA NFIP claims
# -------------------------------
def fetch_nfip(top=1000, max_pages=40):
    """Fetch all NFIP records with payouts, state, and floodEvent."""
    all_rows, skip = [], 0
    for i in range(max_pages):
        params = {
            "$top": top,
            "$skip": skip,
            "$select": "state,yearOfLoss,floodEvent,"
                       "amountPaidOnBuildingClaim,amountPaidOnContentsClaim"
        }
        r = requests.get(NFIP_URL, params=params, timeout=30)
        r.raise_for_status()
        rows = r.json().get("FimaNfipClaims", [])
        if not rows:
            break
        all_rows.extend(rows)
        skip += top
        print(f"Fetched {len(all_rows)} rows...")

    df = pd.DataFrame(all_rows)

    # Clean numeric columns
    df["amountPaidOnBuildingClaim"] = pd.to_numeric(df["amountPaidOnBuildingClaim"], errors="coerce").fillna(0)
    df["amountPaidOnContentsClaim"] = pd.to_numeric(df["amountPaidOnContentsClaim"], errors="coerce").fillna(0)
    df["totalPaid"] = df["amountPaidOnBuildingClaim"] + df["amountPaidOnContentsClaim"]
    df["yearOfLoss"] = pd.to_numeric(df["yearOfLoss"], errors="coerce").fillna(0).astype(int)

    # Drop rows with no floodEvent
    df = df.dropna(subset=["floodEvent"])
    df = df[df["floodEvent"].astype(str).str.strip() != ""]

    return df.dropna(subset=["state"])


# ---------------------------------------
# STEP 2: Fetch Open-Meteo ERA5 (event-year daily)
# ---------------------------------------
def get_weather_openmeteo(lat, lon, year):
    """Return event-year mean precip, max wind, mean temp using Open-Meteo ERA5."""
    try:
        url = (
            f"https://archive-api.open-meteo.com/v1/era5?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={year}-01-01&end_date={year}-12-31"
            f"&daily=precipitation_sum,windspeed_10m_max,temperature_2m_max"
            f"&timezone=UTC"
        )
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json().get("daily", {})

        precip = max(data.get("precipitation_sum", [0]))
        wind_max = max(data.get("windspeed_10m_max", [0]))
        temp_mean = sum(data.get("temperature_2m_max", [])) / max(1,
                                                                  len(data.get("temperature_2m_max",
                                                                               [])))
        return {"precip_mm": precip, "wind_max_m_s": wind_max, "temp_mean_c": temp_mean}
    except Exception as e:
        print(f"Open-Meteo failed for {lat},{lon},{year}: {e}")
        return {"precip_mm": 0, "wind_max_m_s": 0, "temp_mean_c": 0}


# -----------------------------------------
# STEP 3: Classify event type (Hurricane/Flood/etc.)
# -----------------------------------------
def classify_event_type(event_name):
    if not isinstance(event_name, str) or not event_name.strip():
        return None
    name = event_name.lower()
    if "hurricane" in name:
        return "Hurricane"
    elif "flood" in name or "flooding" in name:
        return "Flood"
    elif "storm" in name or "tropical" in name or "nor'easter" in name:
        return "Storm"
    elif "tornado" in name or "twister" in name:
        return "Tornado"
    elif "blizzard" in name or "snow" in name:
        return "Blizzard"
    elif "rain" in name or "torrential" in name:
        return "Heavy Rain"
    else:
        return "Other"


# -------------------------------
# STEP 4: Main pipeline
# -------------------------------
def main():
    print("Fetching NFIP claims...")
    df = fetch_nfip()
    print(f"\nNFIP dataset after filtering empty floodEvent: {len(df)} rows")

    print("\nFetching Open-Meteo daily weather per state-year...")
    climate_rows = []
    for state, group in tqdm(df.groupby("state")):
        if state not in STATE_COORDS:
            continue
        lat, lon = STATE_COORDS[state]

        for year in group["yearOfLoss"].unique():
            if year <= 0 or year > CURRENT_YEAR - 1:
                # Skip invalid or incomplete years
                continue
            markers = get_weather_openmeteo(lat, lon, year)
            climate_rows.append({"state": state, "yearOfLoss": year, **markers})

    climate_df = pd.DataFrame(climate_rows)

    # Merge and add event type
    merged = df.merge(climate_df, on=["state", "yearOfLoss"], how="left")
    merged["event_type"] = merged["floodEvent"].apply(classify_event_type)

    # Reorder columns: floodEvent → event_type → rest
    base_cols = list(merged.columns)
    cols_ordered = (
        base_cols[:base_cols.index("floodEvent") + 1]
        + ["event_type"]
        + [c for c in base_cols if c not in ("floodEvent", "event_type")]
    )
    merged = merged[cols_ordered]

    # Save single CSV here
    out_path = os.path.join(OUT_DIR, "nfip_with_eventtype.csv")
    merged.to_csv(out_path, index=False)
    print(f"\nSaved {len(merged)} NFIP records with event-year weather + event type → {out_path}")


if __name__ == "__main__":
    main()
