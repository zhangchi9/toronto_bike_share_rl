"""Load a single day's trips from the Parquet file, joined with station coords."""
import json
import pandas as pd
import duckdb

PARQUET_PATH = "data/bikeshare-ridership-2026.parquet"
STATIONS_PATH = "data/stations.json"


def load_stations() -> pd.DataFrame:
    with open(STATIONS_PATH) as f:
        stations = json.load(f)
    return pd.DataFrame(stations)


def load_day(date_str: str, stations_df: pd.DataFrame) -> pd.DataFrame:
    """Return all trips for date_str (YYYY-MM-DD), with start/end lat/lon."""
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT * FROM read_parquet('{PARQUET_PATH}')
        WHERE Start_Time::DATE = ?
        """,
        [date_str],
    ).df()

    coords = stations_df[["station_id", "lat", "lon"]]

    df = df.merge(
        coords.rename(columns={"station_id": "Start_Station_Id", "lat": "start_lat", "lon": "start_lon"}),
        on="Start_Station_Id",
        how="left",
    )
    df = df.merge(
        coords.rename(columns={"station_id": "End_Station_Id", "lat": "end_lat", "lon": "end_lon"}),
        on="End_Station_Id",
        how="left",
    )
    return df
