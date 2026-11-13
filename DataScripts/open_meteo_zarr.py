import datetime as dt
import os
from time import sleep

import openmeteo_requests
import requests_cache
from retry_requests import retry

import pandas as pd
import xarray as xr
import numpy as np
from itertools import product

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# API URL
FORECAST_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
REANALYSIS_URL = "https://archive-api.open-meteo.com/v1/archive"

# COORDS
# 55.65990105662984, 12.591038037916826 ITU
# 55.691993, 12.101548 DTU Risø
# LATITUDE = [55.69]
# LONGITUDE = [12.10]
# COORDS FAKE GRID ITU
# LATITUDE = [55.64, 55.65, 55.66, 55.67, 55.68] 
# LONGITUDE = [12.57, 12.58, 12.59, 12.60, 12.61]
# COORDS FAKE GRID DTU 
LATITUDE = [55.67, 55.68, 55.69, 55.70, 55.71] 
LONGITUDE = [12.08, 12.09, 12.10, 12.11, 12.12] 
COORDS = list(product(LATITUDE, LONGITUDE))

# CHANNELS
CHANNELS = [
    "temperature_2m",
    "shortwave_radiation",
    "direct_radiation",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "cloud_cover",
]

# MODEL
MODEL =  "ecmwf_ifs" #"dmi_harmonie_arome_europe"
FORECAST_DAYS = 1

# FILES
FORECAST_CSV_PATH = "ECMWF_forecast_DTU.csv"
REANALYSIS_CSV_PATH = "ECMWF_reanalysis_DTU.csv"
FORECAST_ZARR_PATH = "nwp_dtu.zarr"
REANALYSIS_ZARR_PATH = "Datasets/Reanalysis/reanalysis_dtu.zarr"

FORECAST = False
if FORECAST:
    CSV_PATH = FORECAST_CSV_PATH
    ZARR_PATH = FORECAST_ZARR_PATH
else:
    CSV_PATH = REANALYSIS_CSV_PATH
    ZARR_PATH = REANALYSIS_ZARR_PATH

START_DATE = dt.datetime(2025, 7, 1, 0, 0)
END_DATE = dt.datetime(2026, 8, 31, 23, 0)


def fetch_forecast_data(run, coords):
    """ 
    Fetches OpenMeteo single run API data for specified latitude, longitude and run time
    Returns a dataframe with the important fields from the response
    """
    params = {
        "latitude": [lat for lat, _ in coords],
        "longitude": [lon for _, lon in coords],
        "forecast_days": FORECAST_DAYS,
        "run": run,
        "hourly":  CHANNELS,
        "models": MODEL,
    }

    responses = openmeteo.weather_api(FORECAST_URL, params=params)

    frames = []
    for (lat, lon), response in zip(coords, responses):
        hourly = response.Hourly()
        values = [hourly.Variables(j).ValuesAsNumpy() for j in range(len(CHANNELS))]
        n_steps = len(values[0])

        df = pd.DataFrame({
            "init_time_utc": run,
            "latitude": lat,           
            "longitude": lon,
            "step": np.arange(n_steps),
        })

        for ch, v in zip(CHANNELS, values):
            df[ch] = v

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def fetch_reanalysis_data(start_date, coords):
    """ 
    Fetches OpenMeteo reanalysis API data for specified latitude, longitude and date
    Returns a dataframe with the important fields from the response
    """
    params = {
        "latitude": [lat for lat, _ in coords],
        "longitude": [lon for _, lon in coords],
        "start_date": start_date,
        "end_date": start_date + dt.timedelta(hours=23),
        "hourly": CHANNELS,
        "models": MODEL,
    }

    responses = openmeteo.weather_api(REANALYSIS_URL, params=params)

    frames = []
    for (lat, lon), response in zip(coords, responses):
        hourly = response.Hourly()
        values = [hourly.Variables(j).ValuesAsNumpy() for j in range(len(CHANNELS))]

        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ).tz_convert(None)

        df = pd.DataFrame({
            "init_time_utc":times.strftime("%Y-%m-%dT%H:%M"),
            "latitude": lat,           
            "longitude": lon,
            "step": 0,
        })

        for ch, v in zip(CHANNELS, values):
            df[ch] = v
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_existing_keys():
    """
    Returns a set of the saved responses indexes in the CSV file 
    """
    if not os.path.exists(CSV_PATH):
        return set()

    df = pd.read_csv(CSV_PATH, usecols=["init_time_utc", "latitude", "longitude"])

    return set(zip(df["init_time_utc"], df["latitude"], df["longitude"]))


def append_to_csv(df):
    write_header = not os.path.exists(CSV_PATH)
    df.to_csv(CSV_PATH, mode="a", header=write_header, index=False)


def collect_data(start_date, no_days):
    """
    Load existing values from CSV
    Requests the ones not present
    Returns amount of fetched and skipped requests
    """
    existing = load_existing_keys()
    fetched = skipped = 0

    for i in range(no_days):
        date = start_date + dt.timedelta(days=i)
        date_str = date.strftime("%Y-%m-%dT%H:%M")

        missing = [(lat, lon) for lat, lon in COORDS if (date_str, lat, lon) not in existing]
        skipped += len(COORDS) - len(missing)

        if missing:
            if FORECAST:
                r = fetch_forecast_data(date_str, missing)
            else:
                r = fetch_reanalysis_data(date.date(), missing)
            append_to_csv(r)
            fetched += len(missing)
            sleep(10) # adjust to avoid reaching minutely limit

    print(f"CSV update done: {fetched} locations fetched, {skipped} already saved")


def csv_to_zarr():
    """
    Reads csv file
    Generates final ZARR
    """
    df = pd.read_csv(CSV_PATH, parse_dates=["init_time_utc"])
    df = pre_process(df)

    ds_raw = (
        df.set_index(["init_time_utc", "step", "latitude", "longitude"])[CHANNELS]
        .to_xarray()
    )

    da = (
        ds_raw.to_array(dim="channel")
        .transpose("init_time_utc", "channel", "step", "latitude", "longitude")
        .astype(np.float32)
        .rename("nwp")
    )

    ds = da.to_dataset()
    ds = ds.chunk({"init_time_utc": 1, "channel": -1, "step": -1, "latitude": -1, "longitude": -1})
    ds.to_zarr(ZARR_PATH, mode="w", consolidated=True)
    print(ZARR_PATH)
    print(f"Wrote {ZARR_PATH}")
    print(ds)


def pre_process(df):
    """
    Removes empty values
    Filters analysis time
    """
    df["step"] = pd.to_timedelta(df["step"], unit="h")

    # Filter time to only time used for model
    start = df['init_time_utc'] >= START_DATE
    end = df['init_time_utc']  <= END_DATE
    df = df.loc[start & end]

    # Fill NWP empty values with most recent known value (up to 3 hours before or ahead)
    for c in CHANNELS:
        df[c] = df[c].ffill(limit=3)
    # Drop if longer than that
    df = df.dropna() 

    return df


def compute_normalisation_constants():
    """
    Computes normalisation constants form ZARR File
    """
    ds = xr.open_zarr(ZARR_PATH)
    da = ds["nwp"]

    reduce_dims = [d for d in da.dims if d != "channel"]

    means = da.mean(dim=reduce_dims, skipna=True).compute()
    stds = da.std(dim=reduce_dims, skipna=True).compute()

    print("normalisation_constants:")
    for channel in da["channel"].values:
        mean = float(means.sel(channel=channel).values)
        std = float(stds.sel(channel=channel).values)
        print(f"    {channel}:")
        print(f"        mean: {mean}")
        print(f"        std: {std}")


def main():
    start_date = dt.datetime(2026, 7, 1, 0, 0)
    no_days = 5

    collect_data(start_date, no_days)
    csv_to_zarr()
    compute_normalisation_constants()


if __name__ == "__main__":
    main()