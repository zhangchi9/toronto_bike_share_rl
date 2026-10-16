"""Convert the 2026 ridership CSV to Parquet for fast per-day queries."""
import duckdb

CSV_PATH = "data/bikeshare-ridership-2026/bikeshare-ridership-2026.csv"
PARQUET_PATH = "data/bikeshare-ridership-2026.parquet"


def main():
    print(f"Reading {CSV_PATH} ...")
    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT
                Trip_Id,
                Trip_Duration,
                TRY_CAST(Start_Station_Id AS INTEGER) AS Start_Station_Id,
                strptime(Start_Time::VARCHAR, '%Y-%m-%d %H:%M:%S') AS Start_Time,
                Start_Station_Name,
                TRY_CAST(End_Station_Id AS DOUBLE)::INTEGER AS End_Station_Id,
                strptime(End_Time::VARCHAR, '%Y-%m-%d %H:%M:%S') AS End_Time,
                End_Station_Name,
                Bike_Id,
                User_Type,
                Bike_Model
            FROM read_csv_auto('{CSV_PATH}', header=true, all_varchar=true)
        ) TO '{PARQUET_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    count = con.execute(f"SELECT COUNT(*) FROM '{PARQUET_PATH}'").fetchone()[0]
    print(f"Written {count:,} rows to {PARQUET_PATH}")


if __name__ == "__main__":
    main()
