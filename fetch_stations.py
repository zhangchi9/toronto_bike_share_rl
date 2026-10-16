"""Fetch Bike Share Toronto station metadata (lat/lon) from the GBFS API."""
import json
import requests

GBFS_URL = "https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information"
OUT_PATH = "data/stations.json"


def main():
    resp = requests.get(GBFS_URL, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    stations = payload["data"]["stations"]
    result = [
        {
            "station_id": int(s["station_id"]),
            "name": s["name"],
            "lat": s["lat"],
            "lon": s["lon"],
            "capacity": s.get("capacity", 0),
        }
        for s in stations
    ]

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved {len(result)} stations to {OUT_PATH}")


if __name__ == "__main__":
    main()
