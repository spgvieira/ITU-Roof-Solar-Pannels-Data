import pandas as pd
import datetime as dt
import numpy as np
from pvlib import solarposition
from pyproj import Geod


def load_dataset(cfg):
    """
    loads both PV and NWP dataset
    merges the two datasets by time
    """
    # Load Datasets
    nwp = pd.read_csv(cfg.nwp_csv, parse_dates=["init_time_utc"])
    pv = pd.read_csv(cfg.pv_csv, parse_dates=["time"])

    # Pre-process data
    pv = pre_process_solar(cfg, pv)
    nwp = pre_process_nwp(cfg, nwp)

    # Merge dataset
    df = merge_datasets(cfg, pv, nwp)

    # df = encode_time(df)

    return df


def merge_datasets(cfg, pv_dataset, nwp_dataset):
    """
    Returns one dataset with both pv and nwp information
    """
    # Merge both datasets and remove not used columns
    df = pd.merge(pv_dataset, nwp_dataset)
    df = df.set_index("time").sort_index()

    return df


def pre_process_solar(cfg, df):
    """
    Removes night time values
    Removes outliers
    """  
    # Filter time
    df = filter_time(df, cfg.train_set.start, cfg.test_set.end)

    # Remove night time values
    solpos = solarposition.get_solarposition(pd.DatetimeIndex(df['time']), cfg.latitude, cfg.longitude)
    is_night = (solpos["apparent_elevation"] <= 0).to_numpy()
    df.loc[is_night, 'solar_power'] = np.nan

    # Drop hours where there is no solar power measurement
    df = df.dropna(subset=['solar_power'])

    # TODO: Outliers
    df.loc[df['solar_power'] < 0, 'solar_power'] = 0

    return df


def pre_process_nwp(cfg, df):
    """
    Calculates time of prediction
    Handles grid data
    Removes empty values
    """
    # Sum init_time to step to get actual time
    step = pd.to_timedelta(df["step"], unit="h")
    df['time'] = (df['init_time_utc'] + step).dt.floor("h")
    df = df.drop(columns = ['step', 'init_time_utc', 'is_day'])

    # Handle grid data
    if cfg.grid_data:
        df =  df.set_index(["time", "latitude", "longitude"]).sort_index()
        df = flatten_grid(cfg, df)
        df = df.reset_index(drop=False)
    else:
        latitude = df['latitude'] == cfg.latitude
        longitude = df['longitude'] == cfg.longitude
        df = df.loc[latitude & longitude]
        df = df.drop(columns = ['latitude', 'longitude'])

    # Filter time
    df = filter_time(df, cfg.train_set.start, cfg.test_set.end)

    # Fill NWP empty values with most recent known value (up to 3 hours before or ahead)
    for c in cfg.forecast_columns:
        df[c] = df[c].ffill(limit=3)
    # Drop if longer than that
    df = df.dropna()

    # TODO: Outliers

    return df


def filter_time(df, start_date, end_date):
    """
    Returns only dataset inside of time interval
    """
    if 'time' in df: 
        start = df['time'] >= start_date
        end = df['time'] <= end_date
    else: 
        start = df.index >= start_date
        end = df.index <= end_date
    df = df.loc[start & end]

    return df


def encode_time(df):
    """
    Encode hour information to find daily periodicity
    """
    hour = df.index.hour
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)

    return df


def save_dataset(df, csv_path):
    """
    Save dataset to csv
    """
    df.to_csv(csv_path, index = True)


def add_distance(df, latitude, longitude):
    """
    Transform different coords into a distance column
    """
    geod = Geod(ellps="WGS84")

    n = len(df)
    _, _, dist_m = geod.inv(
        df['longitude'].to_numpy(), df['latitude'].to_numpy(),   
        np.full(n, longitude), np.full(n, latitude),
    )

    df["distance_km"] = dist_m / 1000

    return df


def flatten_grid(cfg, df):
    """
    flattens nwp grid by adding a column for each variable for each coordinate pair
    updates the name of columns used to forecast
    """
    wide = df[cfg.forecast_columns].unstack(level=[l for l in df.index.names if l != "time"])
    wide.columns = ["_".join(map(str, c)) for c in wide.columns]
    cfg.forecast_columns = wide.columns
    return wide.sort_index()