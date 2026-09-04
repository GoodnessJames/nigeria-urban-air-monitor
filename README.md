# Nigeria Urban Air Monitor

Real-Time Air Quality Intelligence dashboard for Lagos, Abuja and Port Harcourt.

## Capstone requirements

### 1. Automated Data Ingestion
`ingestion_client.py` fetches live data from the Open-Meteo Air Quality API and stores new observations in DuckDB. It supports a 60-second continuous background loop.

### 2. In-Memory Analytical Queries
`app.py` uses DuckDB SQL to calculate current pollution, seven-day baselines, hourly patterns, pollutant profiles and record counts.

### 3. Visual Uncertainty Forecasts
The dashboard compares a seasonal-naive baseline with SARIMAX using a hidden 24-hour validation window. It selects the lower-MAE model and displays a 72-hour point forecast with an uncertainty interval.

### 4. Live Public Deployment
The Streamlit app is designed for Streamlit Community Cloud. Push the repository to GitHub and deploy `app.py` from Community Cloud.

## Google Colab

Upload or clone this repository into Colab, then run:

```python
!pip install -r requirements.txt
```

Test ingestion once:

```python
!python ingestion_client.py --once
```

Run the Streamlit dashboard:

```python
!streamlit run app.py --server.headless true --server.port 8501 &
```

For a temporary Colab browser URL, use a tunnel such as:

```python
!npm install -g localtunnel
!lt --port 8501
```

## Important deployment note

GitHub is the source-code repository; it is not the Streamlit web host. The public Streamlit URL is provided by Streamlit Community Cloud.

Because Community Cloud instances have ephemeral local storage, DuckDB is suitable for this capstone's local demonstration, but it should not be treated as permanent production storage. A production system would use persistent managed storage.

## Data-source note

Open-Meteo's Air Quality API provides modelled air-quality data based on CAMS. These values are not equivalent to street-level sensor measurements.
