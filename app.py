"""
Nigeria Urban Air Monitor
Real-Time Air Quality Intelligence

CAPSTONE REQUIREMENTS
1. Automated Data Ingestion
   - Open-Meteo Air Quality API
   - 60-second Streamlit fragment refresh
   - DuckDB local database
2. In-Memory Analytical Queries
   - DuckDB SQL for current values, baselines, hourly patterns and pollutant profiles
3. Visual Uncertainty Forecasts
   - Adaptive forecast engine: seasonal-naive baseline vs SARIMAX
   - Point forecast + empirical/model confidence intervals
4. Live Public Deployment
   - Streamlit-ready entry point for Streamlit Community Cloud
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Nigeria Urban Air Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DB_PATH = Path("air_quality.duckdb")
API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
LOCAL_TZ = ZoneInfo("Africa/Lagos")

CITIES = {
    "Lagos": {"location_id": 1, "latitude": 6.5244, "longitude": 3.3792},
    "Abuja": {"location_id": 2, "latitude": 9.0765, "longitude": 7.3986},
    "Port Harcourt": {
        "location_id": 3,
        "latitude": 4.8156,
        "longitude": 7.0498,
    },
}

POLLUTANTS = {
    "PM2.5": "pm2_5",
    "PM10": "pm10",
    "NO₂": "nitrogen_dioxide",
    "O₃": "ozone",
    "SO₂": "sulphur_dioxide",
    "CO": "carbon_monoxide",
    "Dust": "dust",
}

DISPLAY_NAMES = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "nitrogen_dioxide": "NO₂",
    "ozone": "O₃",
    "sulphur_dioxide": "SO₂",
    "carbon_monoxide": "CO",
    "dust": "Dust",
}

POLLUTANT_AQI_COLUMNS = {
    "PM2.5": "us_aqi_pm2_5",
    "PM10": "us_aqi_pm10",
    "NO₂": "us_aqi_nitrogen_dioxide",
    "O₃": "us_aqi_ozone",
    "SO₂": "us_aqi_sulphur_dioxide",
    "CO": "us_aqi_carbon_monoxide",
    "Dust": None,
}


DB_LOCK = threading.RLock()
LAST_INGESTION_UTC: datetime | None = None


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f6f8fb;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #173b57 100%);
        padding: 2.1rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
    }

    .hero h1 {
        color: white;
        margin: 0;
        font-size: 2.25rem;
        letter-spacing: 0.02em;
    }

    .hero p {
        color: #dbeafe;
        margin: 0.35rem 0 0;
        font-size: 1.05rem;
    }

    .section-title {
        color: #0f172a;
        font-size: 1.35rem;
        font-weight: 800;
        margin-top: 0.8rem;
        margin-bottom: 0.15rem;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 0.92rem;
        margin-bottom: 0.9rem;
    }

    .status-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        min-height: 122px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
    }

    .status-city {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .status-value {
        color: #0f172a;
        font-size: 1.7rem;
        font-weight: 850;
        margin-top: 0.25rem;
    }

    .status-label {
        font-weight: 750;
        font-size: 0.9rem;
        margin-top: 0.15rem;
    }

    .insight {
        background: white;
        border-left: 5px solid #2563eb;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
        color: #334155;
        line-height: 1.55;
    }

    .pipeline {
        background: #0f172a;
        color: white;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
    }

    .pipeline span {
        margin-right: 0.8rem;
        white-space: nowrap;
    }

    .small-note {
        color: #64748b;
        font-size: 0.78rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 0.75rem 0.9rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_connection():
    """Open a short-lived DuckDB connection for thread-safe operations."""
    return duckdb.connect(str(DB_PATH))


def ensure_sequence(con, sequence_name: str, table_name: str, id_column: str) -> None:
    """Create a DuckDB sequence above the existing maximum ID if needed."""
    exists = con.execute(
        """
        SELECT COUNT(*)
        FROM duckdb_sequences()
        WHERE schema_name = 'main' AND sequence_name = ?
        """,
        [sequence_name],
    ).fetchone()[0]

    if not exists:
        max_id = con.execute(
            f"SELECT COALESCE(MAX({id_column}), 0) FROM {table_name}"
        ).fetchone()[0]
        con.execute(
            f"CREATE SEQUENCE IF NOT EXISTS {sequence_name} START {int(max_id) + 1}"
        )


def allocate_sequence_ids(
    con, sequence_name: str, table_name: str, id_column: str, count: int
) -> list[int]:
    """Allocate unique IDs from a database-managed DuckDB sequence.

    The small catch-up loop also repairs an older sequence that was created
    below the table's existing MAX(id).
    """
    if count <= 0:
        return []

    ensure_sequence(con, sequence_name, table_name, id_column)
    max_id = int(
        con.execute(
            f"SELECT COALESCE(MAX({id_column}), 0) FROM {table_name}"
        ).fetchone()[0]
    )

    ids = []
    for _ in range(count):
        candidate = int(
            con.execute(
                f"SELECT nextval('{sequence_name}')",
            ).fetchone()[0]
        )
        while candidate <= max_id:
            candidate = int(
                con.execute(
                    f"SELECT nextval('{sequence_name}')",
                ).fetchone()[0]
            )
        ids.append(candidate)
        max_id = max(max_id, candidate)

    return ids


def initialize_database() -> None:
    """
    Create the database schema and seed the three monitored cities.
    """
    con = get_connection()

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS locations (
            location_id INTEGER PRIMARY KEY,
            location_name VARCHAR NOT NULL UNIQUE,
            latitude DOUBLE NOT NULL,
            longitude DOUBLE NOT NULL,
            country VARCHAR NOT NULL,
            timezone VARCHAR NOT NULL,
            source VARCHAR NOT NULL
        )
        """
    )

    con.execute(
        """
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
            us_aqi_pm2_5 DOUBLE,
            us_aqi_pm10 DOUBLE,
            us_aqi_nitrogen_dioxide DOUBLE,
            us_aqi_ozone DOUBLE,
            us_aqi_sulphur_dioxide DOUBLE,
            us_aqi_carbon_monoxide DOUBLE,
            ingested_at_utc TIMESTAMP NOT NULL,
            UNIQUE(location_id, observed_at)
        )
        """
    )

    con.execute(
        """
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
        """
    )

    # Automatically upgrade older DuckDB files.
    for column_name in [
        "us_aqi_pm2_5",
        "us_aqi_pm10",
        "us_aqi_nitrogen_dioxide",
        "us_aqi_ozone",
        "us_aqi_sulphur_dioxide",
        "us_aqi_carbon_monoxide",
    ]:
        con.execute(
            f"ALTER TABLE air_quality_observations "
            f"ADD COLUMN IF NOT EXISTS {column_name} DOUBLE"
        )

    # Database-managed IDs prevent duplicate primary-key errors when Streamlit
    # reruns the app or more than one browser session writes at the same time.
    ensure_sequence(
        con,
        "observation_id_seq",
        "air_quality_observations",
        "observation_id",
    )
    ensure_sequence(
        con,
        "ingestion_id_seq",
        "ingestion_log",
        "ingestion_id",
    )

    for city, info in CITIES.items():
        con.execute(
            """
            INSERT INTO locations
            VALUES (?, ?, ?, ?, 'Nigeria', 'Africa/Lagos', 'Open-Meteo')
            ON CONFLICT (location_id) DO NOTHING
            """,
            [
                info["location_id"],
                city,
                info["latitude"],
                info["longitude"],
            ],
        )

    con.close()


# ============================================================
# 1. AUTOMATED DATA INGESTION
# ============================================================

def api_request(params: dict) -> dict:
    """Call Open-Meteo and raise a useful error for failed requests."""
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_hourly_response(
    response: dict,
    city: str,
    location_id: int,
    cutoff_local: datetime | None = None,
) -> pd.DataFrame:
    """Convert an Open-Meteo hourly response to our long-format schema."""
    hourly = response.get("hourly", {})

    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)

    rename_map = {
        "time": "observed_at",
        "pm2_5": "pm2_5",
        "pm10": "pm10",
        "nitrogen_dioxide": "nitrogen_dioxide",
        "ozone": "ozone",
        "sulphur_dioxide": "sulphur_dioxide",
        "carbon_monoxide": "carbon_monoxide",
        "dust": "dust",
        "european_aqi": "european_aqi",
        "us_aqi": "us_aqi",
        "us_aqi_pm2_5": "us_aqi_pm2_5",
        "us_aqi_pm10": "us_aqi_pm10",
        "us_aqi_nitrogen_dioxide": "us_aqi_nitrogen_dioxide",
        "us_aqi_ozone": "us_aqi_ozone",
        "us_aqi_sulphur_dioxide": "us_aqi_sulphur_dioxide",
        "us_aqi_carbon_monoxide": "us_aqi_carbon_monoxide",
    }

    df = df.rename(columns=rename_map)

    expected_columns = list(rename_map.values())
    for column in expected_columns:
        if column not in df.columns:
            df[column] = np.nan

    df["observed_at"] = pd.to_datetime(df["observed_at"], errors="coerce")
    df = df.dropna(subset=["observed_at"])

    # Open-Meteo returns Africa/Lagos local timestamps because timezone is set.
    # We store them as naive local timestamps in DuckDB.
    if cutoff_local is not None:
        cutoff_naive = cutoff_local.replace(tzinfo=None)
        df = df[df["observed_at"] <= cutoff_naive]

    df["location_id"] = location_id
    df["location_name"] = city

    columns = [
        "location_id",
        "location_name",
        "observed_at",
        "pm2_5",
        "pm10",
        "nitrogen_dioxide",
        "ozone",
        "sulphur_dioxide",
        "carbon_monoxide",
        "dust",
        "european_aqi",
        "us_aqi",
        "us_aqi_pm2_5",
        "us_aqi_pm10",
        "us_aqi_nitrogen_dioxide",
        "us_aqi_ozone",
        "us_aqi_sulphur_dioxide",
        "us_aqi_carbon_monoxide",
    ]

    return df[columns].copy()


def insert_observations(df: pd.DataFrame) -> int:
    """
    Insert only new (location_id, observed_at) records.

    Observation IDs are allocated once per batch so duplicate primary-key
    collisions cannot occur during historical backfills.
    """
    if df.empty:
        return 0

    con = get_connection()

    work = df.copy()
    work["ingested_at_utc"] = datetime.now(timezone.utc).replace(tzinfo=None)

    # De-duplicate within the incoming batch first.
    work = work.drop_duplicates(subset=["location_id", "observed_at"])

    # Keep existing timestamps in the batch because current Open-Meteo
    # observations may use the same hourly timestamp. In that case we need
    # to UPDATE the existing row so newly added pollutant-specific AQI fields
    # are populated.
    if work.empty:
        con.close()
        return 0

    work = work.reset_index(drop=True)
    work["observation_id"] = allocate_sequence_ids(
        con,
        "observation_id_seq",
        "air_quality_observations",
        "observation_id",
        len(work),
    )

    insert_columns = [
        "observation_id",
        "location_id",
        "observed_at",
        "pm2_5",
        "pm10",
        "nitrogen_dioxide",
        "ozone",
        "sulphur_dioxide",
        "carbon_monoxide",
        "dust",
        "european_aqi",
        "us_aqi",
        "us_aqi_pm2_5",
        "us_aqi_pm10",
        "us_aqi_nitrogen_dioxide",
        "us_aqi_ozone",
        "us_aqi_sulphur_dioxide",
        "us_aqi_carbon_monoxide",
        "ingested_at_utc",
    ]

    con.register("incoming_observations", work[insert_columns])

    con.execute(
        """
        INSERT INTO air_quality_observations
        SELECT *
        FROM incoming_observations
        ON CONFLICT (location_id, observed_at) DO UPDATE SET
            pm2_5 = EXCLUDED.pm2_5,
            pm10 = EXCLUDED.pm10,
            nitrogen_dioxide = EXCLUDED.nitrogen_dioxide,
            ozone = EXCLUDED.ozone,
            sulphur_dioxide = EXCLUDED.sulphur_dioxide,
            carbon_monoxide = EXCLUDED.carbon_monoxide,
            dust = EXCLUDED.dust,
            european_aqi = EXCLUDED.european_aqi,
            us_aqi = EXCLUDED.us_aqi,
            us_aqi_pm2_5 = EXCLUDED.us_aqi_pm2_5,
            us_aqi_pm10 = EXCLUDED.us_aqi_pm10,
            us_aqi_nitrogen_dioxide = EXCLUDED.us_aqi_nitrogen_dioxide,
            us_aqi_ozone = EXCLUDED.us_aqi_ozone,
            us_aqi_sulphur_dioxide = EXCLUDED.us_aqi_sulphur_dioxide,
            us_aqi_carbon_monoxide = EXCLUDED.us_aqi_carbon_monoxide,
            ingested_at_utc = EXCLUDED.ingested_at_utc
        """
    )

    # Count inserted/updated observations for the pipeline status.
    inserted = len(work)
    con.close()

    return inserted


def ingest_current_data(force: bool = False) -> dict:
    """
    Fetch current air quality for all three cities.

    The lock prevents multiple browser sessions from writing to DuckDB
    simultaneously. A 50-second guard also prevents duplicate API calls.
    """
    global LAST_INGESTION_UTC

    now_utc = datetime.now(timezone.utc)

    with DB_LOCK:
        if (
            not force
            and LAST_INGESTION_UTC is not None
            and (now_utc - LAST_INGESTION_UTC).total_seconds() < 50
        ):
            return {
                "status": "SKIPPED",
                "rows_received": 0,
                "rows_inserted": 0,
                "message": "Recent ingestion already completed.",
            }

        initialize_database()

        started = datetime.now(timezone.utc).replace(tzinfo=None)

        received = 0
        inserted = 0
        errors = []

        current_time_local = datetime.now(LOCAL_TZ).replace(
            minute=0, second=0, microsecond=0
        )

        for city, info in CITIES.items():
            params = {
                "latitude": info["latitude"],
                "longitude": info["longitude"],
                "current": (
                    "pm2_5,pm10,nitrogen_dioxide,ozone,"
                    "sulphur_dioxide,carbon_monoxide,dust,"
                    "european_aqi,us_aqi,"
                    "us_aqi_pm2_5,us_aqi_pm10,us_aqi_nitrogen_dioxide,"
                    "us_aqi_ozone,us_aqi_sulphur_dioxide,"
                    "us_aqi_carbon_monoxide"
                ),
                "timezone": "Africa/Lagos",
            }

            try:
                response = api_request(params)
                current = response.get("current", {})

                if not current:
                    raise ValueError("Open-Meteo returned no current data.")

                row = pd.DataFrame(
                    [
                        {
                            "location_id": info["location_id"],
                            "location_name": city,
                            "observed_at": pd.to_datetime(
                                current["time"]
                            ),
                            "pm2_5": current.get("pm2_5"),
                            "pm10": current.get("pm10"),
                            "nitrogen_dioxide": current.get(
                                "nitrogen_dioxide"
                            ),
                            "ozone": current.get("ozone"),
                            "sulphur_dioxide": current.get(
                                "sulphur_dioxide"
                            ),
                            "carbon_monoxide": current.get(
                                "carbon_monoxide"
                            ),
                            "dust": current.get("dust"),
                            "european_aqi": current.get("european_aqi"),
                            "us_aqi": current.get("us_aqi"),
                            "us_aqi_pm2_5": current.get("us_aqi_pm2_5"),
                            "us_aqi_pm10": current.get("us_aqi_pm10"),
                            "us_aqi_nitrogen_dioxide": current.get(
                                "us_aqi_nitrogen_dioxide"
                            ),
                            "us_aqi_ozone": current.get("us_aqi_ozone"),
                            "us_aqi_sulphur_dioxide": current.get(
                                "us_aqi_sulphur_dioxide"
                            ),
                            "us_aqi_carbon_monoxide": current.get(
                                "us_aqi_carbon_monoxide"
                            ),
                        }
                    ]
                )

                received += len(row)
                inserted += insert_observations(row)

            except Exception as exc:
                errors.append(f"{city}: {exc}")

        completed = datetime.now(timezone.utc).replace(tzinfo=None)

        status = "SUCCESS" if not errors else "PARTIAL"

        con = get_connection()
        ingestion_id = allocate_sequence_ids(
            con,
            "ingestion_id_seq",
            "ingestion_log",
            "ingestion_id",
            1,
        )[0]
        con.execute(
            """
            INSERT INTO ingestion_log
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ingestion_id,
                started,
                completed,
                len(CITIES),
                received,
                inserted,
                status,
                "; ".join(errors) if errors else None,
            ],
        )
        con.close()

        LAST_INGESTION_UTC = now_utc

        return {
            "status": status,
            "rows_received": received,
            "rows_inserted": inserted,
            "message": "; ".join(errors) if errors else "All cities updated.",
        }


def backfill_history(days: int = 90) -> dict:
    """
    Create an initial historical dataset for trend analysis and forecasting.

    Open-Meteo supports historical hourly data through past_days. We explicitly
    discard any future timestamps so forecast hours are never stored as history.
    """
    initialize_database()

    con = get_connection()
    counts = con.execute(
        """
        SELECT location_id, COUNT(*) AS n
        FROM air_quality_observations
        GROUP BY location_id
        """
    ).df()
    con.close()

    # If every city already has enough history, skip the expensive backfill.
    if (
        len(counts) == len(CITIES)
        and counts["n"].min() >= max(24 * 30, days * 24 * 0.80)
    ):
        return {
            "status": "SKIPPED",
            "rows_inserted": 0,
            "message": "Historical backfill already exists.",
        }

    total_inserted = 0
    errors = []

    cutoff_local = datetime.now(LOCAL_TZ).replace(
        minute=0, second=0, microsecond=0
    )

    for city, info in CITIES.items():
        params = {
            "latitude": info["latitude"],
            "longitude": info["longitude"],
            "hourly": (
                "pm2_5,pm10,nitrogen_dioxide,ozone,"
                "sulphur_dioxide,carbon_monoxide,dust,"
                "european_aqi,us_aqi,"
                "us_aqi_pm2_5,us_aqi_pm10,us_aqi_nitrogen_dioxide,"
                "us_aqi_ozone,us_aqi_sulphur_dioxide,"
                "us_aqi_carbon_monoxide"
            ),
            "past_days": min(days, 90),
            "forecast_days": 0,
            "timezone": "Africa/Lagos",
        }

        try:
            response = api_request(params)
            df = parse_hourly_response(
                response,
                city,
                info["location_id"],
                cutoff_local=cutoff_local,
            )
            total_inserted += insert_observations(df)

        except Exception as exc:
            errors.append(f"{city}: {exc}")

    return {
        "status": "SUCCESS" if not errors else "PARTIAL",
        "rows_inserted": total_inserted,
        "message": "; ".join(errors) if errors else "Historical data ready.",
    }


# ============================================================
# 2. IN-MEMORY ANALYTICAL QUERIES — DUCKDB SQL
# ============================================================

@st.cache_data(ttl=30, show_spinner=False)
def query_current_pollution() -> pd.DataFrame:
    """Latest observation for every monitored city."""
    con = get_connection()
    df = con.execute(
        """
        WITH latest AS (
            SELECT
                location_id,
                MAX(observed_at) AS latest_time
            FROM air_quality_observations
            GROUP BY location_id
        )
        SELECT
            l.location_name,
            o.observed_at,
            o.pm2_5,
            o.pm10,
            o.nitrogen_dioxide,
            o.ozone,
            o.sulphur_dioxide,
            o.carbon_monoxide,
            o.dust,
            o.european_aqi,
            o.us_aqi,
            o.us_aqi_pm2_5,
            o.us_aqi_pm10,
            o.us_aqi_nitrogen_dioxide,
            o.us_aqi_ozone,
            o.us_aqi_sulphur_dioxide,
            o.us_aqi_carbon_monoxide
        FROM air_quality_observations o
        JOIN latest x
          ON o.location_id = x.location_id
         AND o.observed_at = x.latest_time
        JOIN locations l
          ON o.location_id = l.location_id
        ORDER BY o.pm2_5 DESC
        """
    ).df()
    con.close()
    return df


@st.cache_data(ttl=30, show_spinner=False)
def query_trend(days: int = 7) -> pd.DataFrame:
    """Hourly PM2.5 history for the trend chart."""
    con = get_connection()
    df = con.execute(
        """
        WITH latest AS (
            SELECT MAX(observed_at) AS latest_time
            FROM air_quality_observations
        )
        SELECT
            l.location_name,
            o.observed_at,
            o.pm2_5
        FROM air_quality_observations o
        JOIN locations l
          ON o.location_id = l.location_id
        CROSS JOIN latest
        WHERE o.observed_at >= latest.latest_time - INTERVAL '1 day' * ?
          AND o.pm2_5 IS NOT NULL
        ORDER BY o.observed_at, l.location_name
        """,
        [days],
    ).df()
    con.close()
    return df


@st.cache_data(ttl=30, show_spinner=False)
def query_hourly_pattern() -> pd.DataFrame:
    """Average PM2.5 by hour of day."""
    con = get_connection()
    df = con.execute(
        """
        SELECT
            l.location_name,
            EXTRACT(HOUR FROM o.observed_at) AS hour_of_day,
            AVG(o.pm2_5) AS avg_pm2_5,
            COUNT(*) AS observations
        FROM air_quality_observations o
        JOIN locations l
          ON o.location_id = l.location_id
        WHERE o.pm2_5 IS NOT NULL
        GROUP BY l.location_name, hour_of_day
        ORDER BY l.location_name, hour_of_day
        """
    ).df()
    con.close()
    return df


@st.cache_data(ttl=30, show_spinner=False)
def query_baseline() -> pd.DataFrame:
    """Current PM2.5 versus each city's seven-day PM2.5 average."""
    con = get_connection()
    df = con.execute(
        """
        WITH latest AS (
            SELECT
                location_id,
                MAX(observed_at) AS latest_time
            FROM air_quality_observations
            GROUP BY location_id
        ),
        current_values AS (
            SELECT
                o.location_id,
                l.location_name,
                o.observed_at,
                o.pm2_5 AS current_pm2_5
            FROM air_quality_observations o
            JOIN latest x
              ON o.location_id = x.location_id
             AND o.observed_at = x.latest_time
            JOIN locations l
              ON o.location_id = l.location_id
        ),
        baseline_values AS (
            SELECT
                o.location_id,
                AVG(o.pm2_5) AS seven_day_avg_pm2_5
            FROM air_quality_observations o
            CROSS JOIN (
                SELECT MAX(observed_at) AS latest_time
                FROM air_quality_observations
            ) latest
            WHERE o.observed_at >= latest.latest_time - INTERVAL '7 days'
            GROUP BY o.location_id
        )
        SELECT
            c.location_name,
            c.observed_at,
            ROUND(c.current_pm2_5, 2) AS current_pm2_5,
            ROUND(b.seven_day_avg_pm2_5, 2) AS seven_day_avg_pm2_5,
            ROUND(
                (
                    (c.current_pm2_5 - b.seven_day_avg_pm2_5)
                    / NULLIF(b.seven_day_avg_pm2_5, 0)
                ) * 100,
                2
            ) AS percent_vs_baseline
        FROM current_values c
        JOIN baseline_values b
          ON c.location_id = b.location_id
        ORDER BY percent_vs_baseline DESC
        """
    ).df()
    con.close()
    return df


@st.cache_data(ttl=30, show_spinner=False)
def query_pollutant_profile() -> pd.DataFrame:
    """Latest concentrations of all tracked pollutants."""
    con = get_connection()
    df = con.execute(
        """
        WITH latest AS (
            SELECT
                location_id,
                MAX(observed_at) AS latest_time
            FROM air_quality_observations
            GROUP BY location_id
        )
        SELECT
            l.location_name,
            o.observed_at,
            o.pm2_5,
            o.pm10,
            o.nitrogen_dioxide,
            o.ozone,
            o.sulphur_dioxide,
            o.carbon_monoxide,
            o.dust
        FROM air_quality_observations o
        JOIN latest x
          ON o.location_id = x.location_id
         AND o.observed_at = x.latest_time
        JOIN locations l
          ON o.location_id = l.location_id
        ORDER BY l.location_name
        """
    ).df()
    con.close()
    return df


@st.cache_data(ttl=30, show_spinner=False)
def query_record_count() -> int:
    """Total observations currently stored."""
    con = get_connection()
    count = con.execute(
        "SELECT COUNT(*) FROM air_quality_observations"
    ).fetchone()[0]
    con.close()
    return int(count)


@st.cache_data(ttl=30, show_spinner=False)
def query_last_ingestion():
    """Most recent ingestion log entry."""
    con = get_connection()
    result = con.execute(
        """
        SELECT
            completed_at_utc,
            status,
            rows_received,
            rows_inserted,
            error_message
        FROM ingestion_log
        ORDER BY ingestion_id DESC
        LIMIT 1
        """
    ).df()
    con.close()
    return result


# ============================================================
# 3. VISUAL UNCERTAINTY FORECASTS
# ============================================================

def seasonal_naive_forecast(series: pd.Series, steps: int, season: int = 24):
    """Forecast by repeating the last complete daily cycle."""
    values = series.dropna().values
    if len(values) < season:
        raise ValueError("Not enough history for seasonal-naive forecast.")

    forecast = np.array(
        [values[-season + (i % season)] for i in range(steps)],
        dtype=float,
    )
    return forecast


def fit_sarimax_forecast(
    train: pd.Series,
    steps: int,
    seasonal_period: int = 24,
):
    """Fit a compact SARIMAX model and return point forecast + confidence interval."""
    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, seasonal_period),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(disp=False)
    result = fitted.get_forecast(steps=steps)

    point = result.predicted_mean
    interval = result.conf_int(alpha=0.10)

    lower = interval.iloc[:, 0].to_numpy(dtype=float)
    upper = interval.iloc[:, 1].to_numpy(dtype=float)

    return point.to_numpy(dtype=float), lower, upper


def forecast_with_validation(
    city: str,
    horizon_hours: int = 72,
) -> dict:
    """
    Select the better of:
      A) seasonal-naive baseline
      B) SARIMAX

    The final model is selected using a hidden 24-hour validation window.
    This prevents us from presenting a more complex model simply because it
    sounds more sophisticated.
    """
    con = get_connection()
    df = con.execute(
        """
        SELECT
            o.observed_at,
            o.pm2_5
        FROM air_quality_observations o
        JOIN locations l
          ON o.location_id = l.location_id
        WHERE l.location_name = ?
          AND o.pm2_5 IS NOT NULL
        ORDER BY o.observed_at
        """,
        [city],
    ).df()
    con.close()

    if len(df) < 24 * 14:
        raise ValueError(f"Not enough history to forecast {city}.")

    series = (
        df.set_index("observed_at")["pm2_5"]
        .sort_index()
        .asfreq("h")
        .interpolate(limit_direction="both")
    )

    validation_hours = 24

    train = series.iloc[:-validation_hours]
    test = series.iloc[-validation_hours:]

    baseline_validation = seasonal_naive_forecast(
        train, validation_hours, season=24
    )

    baseline_mae = mean_absolute_error(test, baseline_validation)
    baseline_rmse = np.sqrt(
        mean_squared_error(test, baseline_validation)
    )

    sarimax_ok = True

    try:
        sarimax_validation, _, _ = fit_sarimax_forecast(
            train, validation_hours
        )

        sarimax_mae = mean_absolute_error(test, sarimax_validation)
        sarimax_rmse = np.sqrt(
            mean_squared_error(test, sarimax_validation)
        )

    except Exception:
        sarimax_ok = False
        sarimax_mae = np.inf
        sarimax_rmse = np.inf

    if sarimax_ok and sarimax_mae < baseline_mae:
        selected_model = "SARIMAX"
    else:
        selected_model = "Seasonal Naive"

    future_index = pd.date_range(
        series.index[-1] + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq="h",
    )

    if selected_model == "SARIMAX":
        point, lower, upper = fit_sarimax_forecast(
            series, horizon_hours
        )
    else:
        point = seasonal_naive_forecast(
            series, horizon_hours, season=24
        )

        # Empirical uncertainty from one-step/day-season errors.
        residuals = (
            series.iloc[24:].to_numpy()
            - series.shift(24).iloc[24:].to_numpy()
        )
        residuals = residuals[np.isfinite(residuals)]

        if len(residuals) < 20:
            residual_scale = float(series.std())
        else:
            residual_scale = float(np.std(residuals))

        lower = point - 1.645 * residual_scale
        upper = point + 1.645 * residual_scale

    forecast_df = pd.DataFrame(
        {
            "observed_at": future_index,
            "forecast": np.maximum(point, 0),
            "lower": np.maximum(lower, 0),
            "upper": np.maximum(upper, 0),
        }
    )

    historical = series.tail(24 * 7).reset_index()
    historical.columns = ["observed_at", "pm2_5"]

    return {
        "city": city,
        "model": selected_model,
        "historical": historical,
        "forecast": forecast_df,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "sarimax_mae": sarimax_mae,
        "sarimax_rmse": sarimax_rmse,
    }


# ============================================================
# DASHBOARD PRESENTATION HELPERS
# ============================================================

def classify_us_aqi(aqi) -> tuple[str, str]:
    """
    Dashboard uses a simple three-tier presentation:
      Good     = US AQI 0–50
      Moderate = US AQI 51–100
      Poor     = US AQI >100

    This intentionally compresses the full US AQI scale into the three
    dashboard labels requested in the mockup.
    """
    if pd.isna(aqi):
        return "Unknown", "⚪"

    aqi = float(aqi)

    if aqi <= 50:
        return "Good", "🟢"
    if aqi <= 100:
        return "Moderate", "🟡"
    return "Poor", "🔴"


def status_color(status: str) -> str:
    return {
        "Good": "#16a34a",
        "Moderate": "#eab308",
        "Poor": "#dc2626",
        "Unknown": "#64748b",
    }.get(status, "#64748b")


def format_local_time(value) -> str:
    if pd.isna(value):
        return "—"
    timestamp = pd.Timestamp(value)
    return timestamp.strftime("%d %b %Y • %H:%M")


def build_status_cards(current: pd.DataFrame):
    """Return current city cards ordered worst → best using US AQI."""
    cards = []

    for _, row in current.sort_values(
        "us_aqi", ascending=False, na_position="last"
    ).iterrows():
        status, icon = classify_us_aqi(row["us_aqi"])
        cards.append(
            {
                "city": row["location_name"],
                "pm25": row["pm2_5"],
                "status": status,
                "icon": icon,
                "aqi": row["us_aqi"],
            }
        )

    return cards


def chart_layout(fig, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
    return fig


# ============================================================
# INITIALIZE DATABASE + HISTORY + FIRST LIVE INGESTION
# ============================================================

initialize_database()

with st.spinner("Preparing the live air-quality data pipeline..."):
    history_result = backfill_history(days=90)
    ingestion_result = ingest_current_data(force=False)

# Clear cached SQL results after ingestion so the dashboard sees new rows.
if ingestion_result["status"] in {"SUCCESS", "PARTIAL"}:
    query_current_pollution.clear()
    query_trend.clear()
    query_hourly_pattern.clear()
    query_baseline.clear()
    query_pollutant_profile.clear()
    query_record_count.clear()
    query_last_ingestion.clear()


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>REAL-TIME AIR QUALITY INTELLIGENCE</h1>
        <p>Nigeria Urban Air Monitor</p>
        <p>Live monitoring • Analytics • Forecasting</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LIVE DASHBOARD — AUTO-REFRESHES EVERY 60 SECONDS
# ============================================================

@st.fragment(run_every="60s")
def live_dashboard():
    """
    The fragment is the live layer of the application.

    Every 60 seconds it:
      1. Calls Open-Meteo.
      2. Writes the latest observation to DuckDB.
      3. Re-runs analytical SQL.
      4. Refreshes the dashboard visuals.
    """
    # ========================================================
    # 1. AUTOMATED DATA INGESTION — LIVE REFRESH
    # ========================================================
    live_ingestion = ingest_current_data(force=False)

    if live_ingestion["status"] in {"SUCCESS", "PARTIAL"}:
        query_current_pollution.clear()
        query_trend.clear()
        query_hourly_pattern.clear()
        query_baseline.clear()
        query_pollutant_profile.clear()
        query_record_count.clear()
        query_last_ingestion.clear()

    current = query_current_pollution()

    if current.empty:
        st.error("No air-quality observations are currently available.")
        return

    # ========================================================
    # DYNAMIC STATUS CARDS — WORST → BEST
    # ========================================================
    cards = build_status_cards(current)

    card_columns = st.columns(4)

    for column, card in zip(card_columns[:3], cards[:3]):
        with column:
            color = status_color(card["status"])
            st.markdown(
                f"""
                <div class="status-card">
                    <div class="status-city">
                        {card["icon"]} {card["city"]}
                    </div>
                    <div class="status-value">
                        {card["pm25"]:.1f} <span style="font-size:0.9rem;">µg/m³</span>
                    </div>
                    <div class="status-label" style="color:{color};">
                        Status: {card["status"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with card_columns[3]:
        latest_time = current["observed_at"].max()
        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-city">🔄 LAST UPDATED</div>
                <div class="status-value" style="font-size:1.25rem;">
                    {format_local_time(latest_time)}
                </div>
                <div class="status-label" style="color:#2563eb;">
                    Africa/Lagos
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">CURRENT AIR QUALITY</div>'
        '<div class="section-subtitle">'
        "Select a city and pollutant to inspect current conditions."
        "</div>",
        unsafe_allow_html=True,
    )

    filter_col1, filter_col2 = st.columns([1, 1])

    with filter_col1:
        selected_city = st.selectbox(
            "City",
            ["All Cities"] + sorted(current["location_name"].unique()),
        )

    with filter_col2:
        selected_pollutant = st.selectbox(
            "Pollutant",
            list(POLLUTANTS.keys()),
        )

    selected_column = POLLUTANTS[selected_pollutant]

    if selected_city == "All Cities":
        filtered_current = current.copy()
    else:
        filtered_current = current[
            current["location_name"] == selected_city
        ].copy()

    metric_columns = st.columns(len(filtered_current))

    selected_aqi_column = POLLUTANT_AQI_COLUMNS[selected_pollutant]

    for column, (_, row) in zip(
        metric_columns, filtered_current.iterrows()
    ):
        with column:
            concentration = row[selected_column]

            if selected_aqi_column is None:
                aqi_text = "US AQI not available"
                category_text = "⚪ Not available"
            else:
                pollutant_aqi = row[selected_aqi_column]
                status, icon = classify_us_aqi(pollutant_aqi)

                if pd.notna(pollutant_aqi):
                    aqi_text = f"US AQI {float(pollutant_aqi):.0f}"
                    category_text = f"{icon} {status}"
                else:
                    aqi_text = "US AQI N/A"
                    category_text = "⚪ Unknown"

            st.markdown(
                f'''
                <div class="status-card">
                    <div class="status-city">{row["location_name"]}</div>
                    <div class="status-value">
                        {f"{concentration:.1f}" if pd.notna(concentration) else "N/A"}
                        <span style="font-size:0.9rem;">µg/m³</span>
                    </div>
                    <div style="margin-top:0.45rem; font-weight:750; color:#334155;">
                        {aqi_text}
                    </div>
                    <div class="status-label">
                        {category_text}
                    </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

    with st.expander("ℹ️ How to read US AQI"):
        st.markdown(
            '''
            **US AQI categories**

            | US AQI | Category |
            |---:|---|
            | **0–50** | 🟢 Good |
            | **51–100** | 🟡 Moderate |
            | **101–150** | 🟠 Unhealthy for Sensitive Groups |
            | **151–200** | 🔴 Unhealthy |
            | **201–300** | 🟣 Very Unhealthy |
            | **301–500** | 🟤 Hazardous |

            The concentration is the amount of the selected pollutant.
            The pollutant-specific US AQI is the AQI for that pollutant.
            The category shown on each card is based on that pollutant-specific
            AQI, not the city's overall US AQI.

            **Example:** CO → `379.0 µg/m³` → `US AQI 45` → 🟢 Good.
            '''
        )

    # ========================================================
    # ① CITY POLLUTION COMPARISON
    # ========================================================
    st.markdown(
        '<div class="section-title">① CITY POLLUTION COMPARISON</div>'
        '<div class="section-subtitle">'
        "Which city currently has the highest PM2.5 concentration?"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        city_chart = px.bar(
            current.sort_values("pm2_5", ascending=True),
            x="pm2_5",
            y="location_name",
            orientation="h",
            text="pm2_5",
            labels={
                "pm2_5": "PM2.5 (µg/m³)",
                "location_name": "",
            },
            title="Current PM2.5 by City",
        )
        city_chart.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside",
        )
        st.plotly_chart(
            chart_layout(city_chart, 390),
            width="stretch",
        )

    # ========================================================
    # ② POLLUTION TREND
    # ========================================================
    with col2:
        trend = query_trend(days=7)

        trend_chart = px.line(
            trend,
            x="observed_at",
            y="pm2_5",
            color="location_name",
            markers=False,
            labels={
                "observed_at": "Time",
                "pm2_5": "PM2.5 (µg/m³)",
                "location_name": "",
            },
            title="Seven-Day PM2.5 Trend",
        )

        st.plotly_chart(
            chart_layout(trend_chart, 390),
            width="stretch",
        )

    # ========================================================
    # ③ WHEN IS POLLUTION WORST?
    # ========================================================
    st.markdown(
        '<div class="section-title">③ WHEN IS POLLUTION WORST?</div>'
        '<div class="section-subtitle">'
        "Average PM2.5 by hour of day across the available history."
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    hourly = query_hourly_pattern()

    with col1:
        heatmap_data = hourly.pivot(
            index="location_name",
            columns="hour_of_day",
            values="avg_pm2_5",
        )

        heatmap = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=[f"{int(x):02d}:00" for x in heatmap_data.columns],
                y=heatmap_data.index,
                colorbar=dict(title="PM2.5"),
                hovertemplate=(
                    "City: %{y}<br>"
                    "Hour: %{x}<br>"
                    "Avg PM2.5: %{z:.1f} µg/m³"
                    "<extra></extra>"
                ),
            )
        )

        heatmap.update_layout(title="Hourly PM2.5 Pattern")
        st.plotly_chart(
            chart_layout(heatmap, 360),
            width="stretch",
        )

    with col2:
        hourly_line = px.line(
            hourly,
            x="hour_of_day",
            y="avg_pm2_5",
            color="location_name",
            markers=True,
            labels={
                "hour_of_day": "Hour of Day",
                "avg_pm2_5": "Average PM2.5 (µg/m³)",
                "location_name": "",
            },
            title="Average PM2.5 by Hour",
        )

        hourly_line.update_xaxes(dtick=2)

        st.plotly_chart(
            chart_layout(hourly_line, 360),
            width="stretch",
        )

    # ========================================================
    # ④ POLLUTANT PROFILE
    # ========================================================
    st.markdown(
        '<div class="section-title">④ POLLUTANT PROFILE</div>'
        '<div class="section-subtitle">'
        "What pollutants are present in the latest observation?"
        "</div>",
        unsafe_allow_html=True,
    )

    profile = query_pollutant_profile()

    profile_long = profile.melt(
        id_vars=["location_name"],
        value_vars=list(POLLUTANTS.values()),
        var_name="pollutant",
        value_name="concentration",
    )

    profile_long["pollutant"] = profile_long["pollutant"].map(
        DISPLAY_NAMES
    )

    # A heatmap is used because pollutant units and magnitudes differ.
    # Exact values remain available through hover.
    profile_pivot = profile_long.pivot(
        index="pollutant",
        columns="location_name",
        values="concentration",
    )

    profile_chart = go.Figure(
        data=go.Heatmap(
            z=profile_pivot.values,
            x=profile_pivot.columns,
            y=profile_pivot.index,
            colorbar=dict(title="Value"),
            hovertemplate=(
                "Pollutant: %{y}<br>"
                "City: %{x}<br>"
                "Value: %{z:.2f}"
                "<extra></extra>"
            ),
        )
    )

    profile_chart.update_layout(title="Current Pollutant Concentrations")

    st.plotly_chart(
        chart_layout(profile_chart, 420),
        width="stretch",
    )

    st.caption(
        "Pollutant concentrations use their source units. The heatmap is "
        "used because the pollutants have different scales; hover over a "
        "cell for the exact current value."
    )

    # ========================================================
    # ⑤ AIR QUALITY FORECAST
    # ========================================================
    st.markdown(
        '<div class="section-title">🔮 ⑤ AIR QUALITY FORECAST</div>'
        '<div class="section-subtitle">'
        "Historical PM2.5, future point predictions and uncertainty intervals."
        "</div>",
        unsafe_allow_html=True,
    )

    forecast_city = st.selectbox(
        "Forecast city",
        sorted(current["location_name"].unique()),
        key="forecast_city",
    )

    with st.spinner(f"Generating the {forecast_city} forecast..."):
        forecast_result = forecast_with_validation(
            forecast_city,
            horizon_hours=72,
        )

    historical = forecast_result["historical"]
    future = forecast_result["forecast"]

    forecast_fig = go.Figure()

    forecast_fig.add_trace(
        go.Scatter(
            x=historical["observed_at"],
            y=historical["pm2_5"],
            mode="lines",
            name="Historical PM2.5",
        )
    )

    forecast_fig.add_trace(
        go.Scatter(
            x=future["observed_at"],
            y=future["upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    forecast_fig.add_trace(
        go.Scatter(
            x=future["observed_at"],
            y=future["lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.16)",
            name="Prediction interval",
            hoverinfo="skip",
        )
    )

    forecast_fig.add_trace(
        go.Scatter(
            x=future["observed_at"],
            y=future["forecast"],
            mode="lines+markers",
            name=f"Forecast ({forecast_result['model']})",
        )
    )

    forecast_fig.add_vline(
        x=historical["observed_at"].max(),
        line_dash="dash",
        annotation_text="NOW",
        annotation_position="top",
    )

    forecast_fig.update_layout(
        title=f"{forecast_city} PM2.5 — 72-Hour Forecast",
        xaxis_title="Time",
        yaxis_title="PM2.5 (µg/m³)",
    )

    st.plotly_chart(
        chart_layout(forecast_fig, 480),
        width="stretch",
    )

    forecast_col1, forecast_col2, forecast_col3 = st.columns(3)

    with forecast_col1:
        st.metric(
            "Selected model",
            forecast_result["model"],
        )

    with forecast_col2:
        st.metric(
            "Validation MAE",
            (
                f"{forecast_result['sarimax_mae']:.2f}"
                if forecast_result["model"] == "SARIMAX"
                else f"{forecast_result['baseline_mae']:.2f}"
            ),
            help="Lower is better. Calculated on a hidden 24-hour validation window.",
        )

    with forecast_col3:
        st.metric(
            "Forecast horizon",
            "72 hours",
        )

    st.caption(
        "The forecast engine validates SARIMAX against a seasonal-naive "
        "24-hour baseline and uses the better model. The shaded interval "
        "represents forecast uncertainty; it is not a guarantee that actual "
        "future pollution will remain inside the band."
    )

    # ========================================================
    # ⑥ CURRENT CONDITIONS VS RECENT BASELINE
    # ========================================================
    st.markdown(
        '<div class="section-title">'
        "🚨 ⑥ CURRENT CONDITIONS VS RECENT BASELINE"
        "</div>"
        '<div class="section-subtitle">'
        "Is current PM2.5 unusual compared with the previous seven-day average?"
        "</div>",
        unsafe_allow_html=True,
    )

    baseline = query_baseline()

    baseline_columns = st.columns(3)

    for column, city in zip(
        baseline_columns,
        ["Port Harcourt", "Lagos", "Abuja"],
    ):
        row = baseline[baseline["location_name"] == city]

        with column:
            if row.empty:
                st.warning(f"No baseline data for {city}.")
                continue

            record = row.iloc[0]
            pct = record["percent_vs_baseline"]

            if pd.isna(pct):
                interpretation = "Insufficient baseline data"
            elif pct > 20:
                interpretation = "🔴 Unusually high"
            elif pct > 0:
                interpretation = "⚠️ Above recent average"
            elif pct >= -20:
                interpretation = "🟢 Within recent range"
            else:
                interpretation = "🟢 Below recent average"

            st.markdown(
                f"""
                <div class="status-card">
                    <div class="status-city">{city}</div>
                    <p><b>Current:</b> {record["current_pm2_5"]:.1f} µg/m³</p>
                    <p><b>7-day avg:</b> {record["seven_day_avg_pm2_5"]:.1f} µg/m³</p>
                    <p><b>Change:</b> {pct:+.1f}% vs baseline</p>
                    <p><b>{interpretation}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ========================================================
    # KEY INSIGHT
    # ========================================================
    highest = current.sort_values(
        "pm2_5", ascending=False
    ).iloc[0]

    highest_baseline = baseline[
        baseline["location_name"] == highest["location_name"]
    ]

    if not highest_baseline.empty:
        pct = float(
            highest_baseline.iloc[0]["percent_vs_baseline"]
        )
    else:
        pct = 0.0

    forecast_values = future["forecast"].to_numpy()
    recent_value = float(historical["pm2_5"].iloc[-1])
    future_mean = float(np.mean(forecast_values))

    if future_mean > recent_value * 1.10:
        direction = "deteriorating"
    elif future_mean < recent_value * 0.90:
        direction = "improving"
    else:
        direction = "relatively stable"

    st.markdown(
        '<div class="section-title">💡 KEY INSIGHT</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="insight">
            <b>{highest["location_name"]}</b> currently has the highest
            PM2.5 concentration among the three monitored cities at
            <b>{highest["pm2_5"]:.1f} µg/m³</b>.
            Its current level is <b>{pct:+.1f}%</b> versus its seven-day
            PM2.5 baseline. For the selected forecast city,
            the model projects conditions that are
            <b>{direction}</b> over the next 72 hours.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # DATA PIPELINE
    # ========================================================
    st.markdown(
        '<div class="section-title">DATA PIPELINE</div>',
        unsafe_allow_html=True,
    )

    last_log = query_last_ingestion()
    records = query_record_count()

    if not last_log.empty:
        log = last_log.iloc[0]
        api_status = (
            "🟢 API Connected"
            if log["status"] in {"SUCCESS", "PARTIAL"}
            else "🔴 API Error"
        )
        ingestion_status = (
            "🟢 Data Ingested"
            if log["rows_received"] > 0
            else "🟠 No New Row"
        )
        updated_status = (
            "🟢 DuckDB Updated"
            if log["rows_inserted"] > 0
            else "🟠 No New Timestamp"
        )
    else:
        api_status = "🟠 API Pending"
        ingestion_status = "🟠 Pending"
        updated_status = "🟠 Pending"

    st.markdown(
        f"""
        <div class="pipeline">
            <span>{api_status}</span> →
            <span>{ingestion_status}</span> →
            <span>{updated_status}</span> →
            <span>🟢 Dashboard Live</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    last_ingestion_text = (
        str(last_log.iloc[0]["completed_at_utc"])
        if not last_log.empty
        else "Not available"
    )

    pipeline_col1, pipeline_col2 = st.columns(2)

    with pipeline_col1:
        st.caption(
            f"Last ingestion: {last_ingestion_text} UTC"
        )

    with pipeline_col2:
        st.caption(
            f"Records stored: {records:,}"
        )

    # ========================================================
    # DATA-SOURCE DISCLOSURE
    # ========================================================
    st.markdown("---")
    st.markdown(
        """
        <div class="small-note">
        <b>Data source:</b> Open-Meteo Air Quality API using CAMS-based
        modelled air-quality data. Values represent model output rather than
        street-level sensor measurements. AQI labels are used for dashboard
        communication and should not be interpreted as causal evidence.
        </div>
        """,
        unsafe_allow_html=True,
    )


# Run the live dashboard.
live_dashboard()
