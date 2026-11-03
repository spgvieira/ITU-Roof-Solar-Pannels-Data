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

    save_dataset(df, "double_check.csv")

    return df


def merge_datasets(cfg, pv_dataset, nwp_dataset):
    """
    Returns one dataset with both pv and nwp information
    """
    # Merge both datasets and remove not used columns
    if cfg.grid_data:
        df = pd.merge(pv_dataset, nwp_dataset, how='outer')
        df = add_distance(df, cfg.latitude, cfg.longitude)
        df = df.set_index(["time", "distance_km"]).sort_index()
    else:
        nwp_dataset = nwp_dataset.drop_duplicates(subset="time")
        df = pd.merge(pv_dataset, nwp_dataset)
        df = df.set_index("time").sort_index()

    df = df.drop(columns = ['step', 'init_time_utc', 'is_day', 'latitude', 'longitude'])

    return df


def pre_process_solar(cfg, df):
    """
    """  
    # Filter time
    df = filter_time(df, cfg.train_set.start, cfg.test_set.end)

    # Drop hours where there is no solar power measurement
    df = df.dropna(subset=['solar_power'])

    # Remove night time values
    solpos = solarposition.get_solarposition(pd.DatetimeIndex(df['time']), cfg.latitude, cfg.longitude)
    is_night = (solpos["apparent_elevation"] <= 0).to_numpy()
    df.loc[is_night, 'solar_power'] = 0

    # TODO: Outliers
    df.loc[df['solar_power'] < 0, 'solar_power'] = 0

    return df


def pre_process_nwp(cfg, df):
    """
    """
    # Sum init_time to step to get actual time
    step = pd.to_timedelta(df["step"], unit="h")
    df['time'] = (df['init_time_utc'] + step).dt.floor("h")

    # Filter time
    df = filter_time(df, cfg.train_set.start, cfg.test_set.end)

    # Fill NWP empty values with most recent known value
    for c in cfg.forecast_columns:
        df[c] = df[c].ffill().bfill()

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