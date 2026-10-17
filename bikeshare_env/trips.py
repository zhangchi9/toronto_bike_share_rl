"""Load real trips for a simulation period as minute-indexed numpy event arrays."""
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from data_loader import PARQUET_PATH


@dataclass
class TripEvents:
    """Trips sorted by start minute. Minutes count from 00:00 of the period start."""
    start_min: np.ndarray   # int64
    end_min: np.ndarray     # int64
    start_idx: np.ndarray   # int64 station row index
    end_idx: np.ndarray     # int64 station row index
    rent_min: np.ndarray    # int64 billed minutes, >= 1
    casual: np.ndarray | None = None   # int8 1 = Casual user (display only)

    def __len__(self) -> int:
        return len(self.start_min)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "TripEvents":
        """Build from columns start_min, end_min, start_idx, end_idx, rent_min
        (and optional casual)."""
        df = df.sort_values("start_min", kind="stable")
        casual = df["casual"].to_numpy(dtype=np.int8) if "casual" in df else None
        return cls(*(df[c].to_numpy(dtype=np.int64)
                     for c in ("start_min", "end_min", "start_idx", "end_idx", "rent_min")),
                   casual=casual)


def load_period_trips(start_date: str, end_date: str,
                      station_ids: np.ndarray) -> TripEvents:
    """Trips whose Start_Time falls on [start_date, end_date] (inclusive).

    Trips touching stations not in `station_ids` are dropped. End times past the
    period are kept; the env simply never reaches them.
    """
    t0 = pd.Timestamp(start_date)
    t1 = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    df = duckdb.connect().execute(
        f"""
        SELECT Start_Station_Id, End_Station_Id, Start_Time, End_Time,
               TRY_CAST(Trip_Duration AS DOUBLE) AS dur, User_Type
        FROM read_parquet('{PARQUET_PATH}')
        WHERE Start_Time >= ? AND Start_Time < ?
          AND Start_Station_Id IS NOT NULL AND End_Station_Id IS NOT NULL
          AND End_Time IS NOT NULL
        """,
        [t0.to_pydatetime(), t1.to_pydatetime()],
    ).df()

    idx_of = pd.Series(np.arange(len(station_ids)), index=station_ids)
    df["start_idx"] = df["Start_Station_Id"].map(idx_of)
    df["end_idx"] = df["End_Station_Id"].map(idx_of)
    df = df.dropna(subset=["start_idx", "end_idx"])

    df["start_min"] = ((df["Start_Time"] - t0).dt.total_seconds() // 60).astype(np.int64)
    df["end_min"] = ((df["End_Time"] - t0).dt.total_seconds() // 60).astype(np.int64)
    df["end_min"] = np.maximum(df["end_min"], df["start_min"])
    secs = df["dur"].fillna((df["End_Time"] - df["Start_Time"]).dt.total_seconds())
    df["rent_min"] = np.maximum(1, np.ceil(secs / 60)).astype(np.int64)
    df["casual"] = (df["User_Type"] == "Casual").astype(np.int8)
    return TripEvents.from_frame(df)
