"""
Background ingestion client for the Nigeria Urban Air Monitor.

CORE REQUIREMENT 1 — AUTOMATED DATA INGESTION
This script:
1. Calls the active Open-Meteo Air Quality API.
2. Validates the response.
3. Appends new observations to DuckDB.
4. Records each run in ingestion_log.
5. Repeats every 60 seconds when run in continuous mode.

Run in Colab:
    python ingestion_client.py --once
or:
    python ingestion_client.py --interval 60
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests


API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
DB_PATH = Path("air_quality.duckdb")

CITIES = {
    "Lagos": {"location_id": 1, "latitude": 6.5244, "longitude": 3.3792},
    "Abuja": {"location_id": 2, "latitude": 9.0765, "longitude": 7.3986},
    "Port Harcourt": {"location_id": 3, "latitude": 4.8156, "longitude": 7.0498},
}


def connection():
    return duckdb.connect(str(DB_PATH))


def initialize_database():
    con = connection()

    con.execute("""
        CREATE TABLE IF NOT EXISTS locations (
            location_id INTEGER PRIMARY KEY,
            location_name VARCHAR NOT NULL UNIQUE,
            latitude DOUBLE NOT NULL,
            longitude DOUBLE NOT NULL,
            country VARCHAR NOT NULL,
            timezone VARCHAR NOT NULL,
            source VARCHAR NOT NULL
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS air_quality_observations (
            observation_id BIGINT PRIMARY KEY,
            location_id INTEGER NOT NULL,
            observed_at TIMESTAMP NOT NULL,
            pm2_5 DOUBLE,
            pm10 DOUBLE,
            nitrogen_dioxide DOUBLE,
            ozone DOUBLE,
            sulphur_dioxide DOUBLE,
            carbon_monoxide DOUBLE,
            dust DOUBLE,
            european_aqi DOUBLE,
            us_aqi DOUBLE,
            ingested_at_utc TIMESTAMP NOT NULL,
            UNIQUE(location_id, observed_at)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_log (
            ingestion_id BIGINT PRIMARY KEY,
            started_at_utc TIMESTAMP NOT NULL,
            completed_at_utc TIMESTAMP,
            locations_requested INTEGER NOT NULL,
            rows_received INTEGER DEFAULT 0,
            rows_inserted INTEGER DEFAULT 0,
            status VARCHAR NOT NULL,
            error_message VARCHAR
        )
    """)

    for city, info in CITIES.items():
        con.execute("""
            INSERT INTO locations
            VALUES (?, ?, ?, ?, 'Nigeria', 'Africa/Lagos', 'Open-Meteo')
            ON CONFLICT (location_id) DO NOTHING
        """, [
            info["location_id"],
            city,
            info["latitude"],
            info["longitude"],
        ])

    con.close()


def fetch_current(city, info):
    """Fetch one current observation from Open-Meteo."""
    params = {
        "latitude": info["latitude"],
        "longitude": info["longitude"],
        "current": (
            "pm2_5,pm10,nitrogen_dioxide,ozone,"
            "sulphur_dioxide,carbon_monoxide,dust,"
            "european_aqi,us_aqi"
        ),
        "timezone": "Africa/Lagos",
    }

    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()
    current = payload.get("current")

    if not current:
        raise ValueError(f"No current data returned for {city}.")

    return {
        "location_id": info["location_id"],
        "location_name": city,
        "observed_at": pd.to_datetime(current["time"]),
        "pm2_5": current.get("pm2_5"),
        "pm10": current.get("pm10"),
        "nitrogen_dioxide": current.get("nitrogen_dioxide"),
        "ozone": current.get("ozone"),
        "sulphur_dioxide": current.get("sulphur_dioxide"),
        "carbon_monoxide": current.get("carbon_monoxide"),
        "dust": current.get("dust"),
        "european_aqi": current.get("european_aqi"),
        "us_aqi": current.get("us_aqi"),
    }


def insert_rows(rows):
    """Insert only timestamps not already present in DuckDB."""
    if not rows:
        return 0

    df = pd.DataFrame(rows)
    df["ingested_at_utc"] = datetime.now(timezone.utc).replace(tzinfo=None)
    df = df.drop_duplicates(["location_id", "observed_at"])

    con = connection()

    existing = con.execute("""
        SELECT location_id, observed_at
        FROM air_quality_observations
    """).df()

    if not existing.empty:
        df = df.merge(
            existing.assign(_exists=1),
            on=["location_id", "observed_at"],
            how="left",
        )
        df = df[df["_exists"].isna()].drop(columns="_exists")

    if df.empty:
        con.close()
        return 0

    max_id = con.execute("""
        SELECT COALESCE(MAX(observation_id), 0)
        FROM air_quality_observations
    """).fetchone()[0]

    df = df.reset_index(drop=True)
    df["observation_id"] = np.arange(
        max_id + 1,
        max_id + 1 + len(df),
        dtype=np.int64,
    )

    columns = [
        "observation_id", "location_id", "observed_at",
        "pm2_5", "pm10", "nitrogen_dioxide", "ozone",
        "sulphur_dioxide", "carbon_monoxide", "dust",
        "european_aqi", "us_aqi", "ingested_at_utc"
    ]

    con.register("new_rows", df[columns])
    con.execute("""
        INSERT INTO air_quality_observations
        SELECT * FROM new_rows
    """)

    inserted = len(df)
    con.close()
    return inserted


def run_once():
    initialize_database()

    started = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    errors = []

    for city, info in CITIES.items():
        try:
            rows.append(fetch_current(city, info))
        except Exception as exc:
            errors.append(f"{city}: {exc}")

    inserted = insert_rows(rows)

    con = connection()
    next_id = con.execute("""
        SELECT COALESCE(MAX(ingestion_id), 0) + 1
        FROM ingestion_log
    """).fetchone()[0]

    status = "SUCCESS" if not errors else "PARTIAL"

    con.execute("""
        INSERT INTO ingestion_log
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        next_id,
        started,
        datetime.now(timezone.utc).replace(tzinfo=None),
        len(CITIES),
        len(rows),
        inserted,
        status,
        "; ".join(errors) if errors else None,
    ])

    con.close()

    print(
        f"✅ {status} | API rows received: {len(rows)} | "
        f"new rows inserted: {inserted} | "
        f"database: {DB_PATH}"
    )

    if errors:
        print("⚠️ " + " | ".join(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    if args.once:
        run_once()
        return

    print(f"🚀 Background ingestion started. Interval: {args.interval} seconds.")

    while True:
        run_once()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
