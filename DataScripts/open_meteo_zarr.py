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
URL = "https://single-runs-api.open-meteo.com/v1/forecast"

# COORDS
# 55.65990105662984, 12.591038037916826 ITU
# 55.691993, 12.101548 DTU Risø
# LATITUDE = [55.69]
# LONGITUDE = [12.10]
# COORDS FAKE GRID ITU
# LATITUDE = [55.64, 55.65, 55.66, 55.67, 55.68] 
# LONGITUDE = [12.57, 12.58, 12.59, 12.60, 12.61]
# # COORDS FAKE GRID DTU 
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
CSV_PATH = "forecast_DTU_multiple_coords.csv"
ZARR_PATH = "forecast_nwp_dtu.zarr"
KEY_COLS = ["init_time_utc", "latitude", "longitude"]


def fetch_forecast_data(run, coords):
    """ 
    Fetches OpenMeteo API data for specified latitude, longitude and run time
    Returns a dataframe with the important fields from the response
    """
    weather_variables = [*CHANNELS,"is_day"]
    params = {
        "latitude": [lat for lat, _ in coords],
        "longitude": [lon for _, lon in coords],
        "forecast_days": FORECAST_DAYS,
        "run": run,
        "hourly": weather_variables,
        "models": MODEL,
    }

    responses = openmeteo.weather_api(URL, params=params)

    frames = []
    for (lat, lon), response in zip(coords, responses):
        hourly = response.Hourly()
        values = [hourly.Variables(j).ValuesAsNumpy() for j in range(len(weather_variables))]
        n_steps = len(values[0])

        df = pd.DataFrame({
            "init_time_utc": run,
            "latitude": lat,           
            "longitude": lon,
            "step": np.arange(n_steps),
        })

        for ch, v in zip(weather_variables, values):
            df[ch] = v

        frames.append(df)

    return pd.concat(frames, ignore_index=True)



def load_existing_keys():
    """
    Returns a set of the saved responses indexes in the CSV file 
    """
    if not os.path.exists(CSV_PATH):
        return set()

    df = pd.read_csv(CSV_PATH, usecols=KEY_COLS)

    return set(zip(df["init_time_utc"], df["latitude"], df["longitude"]))


def append_to_csv(df):
    write_header = not os.path.exists(CSV_PATH)
    df.to_csv(CSV_PATH, mode="a", header=write_header, index=False)


def collect_forecast_data(start_date, no_days):
    """
    Load existing values from CSV
    Requests the ones not present
    Returns amount of fetched and skipped requests
    """
    existing = load_existing_keys()
    fetched = skipped = 0

    for i in range(no_days):
        run_dt = start_date + dt.timedelta(days=i)
        run_str = run_dt.strftime('%Y-%m-%dT%H:%M')

        missing = [(lat, lon) for lat, lon in COORDS if (run_str, lat, lon) not in existing]
        skipped += len(COORDS) - len(missing)

        if not missing:
            continue

        r = fetch_forecast_data(run_str, missing)
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
    print(f"Wrote {ZARR_PATH}")
    print(ds)

def pre_process(df):
    """
    """
    df["step"] = pd.to_timedelta(df["step"], unit="h")

    for c in CHANNELS:
        df[c] = df[c].ffill().bfill()

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
    start_date = dt.datetime(2025, 8, 10, 0, 0)
    no_days = 365

    collect_forecast_data(start_date, no_days)
    csv_to_zarr()
    compute_normalisation_constants()


if __name__ == "__main__":
    main()