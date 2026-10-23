import pandas as pd
import datetime as dt
import numpy as np
from pvlib import solarposition
import matplotlib.pyplot as plt
import seaborn as sb

# COORDS
LATITUDE = 55.66
LONGITUDE = 12.59

CHANNELS = [
    "temperature_2m",
    "shortwave_radiation",
    "direct_radiation",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "cloud_cover",
]

KEY_COLS = ["init_time_utc", "latitude", "longitude"]

def load_dataset(nwp_csv, pv_csv, start_date, end_date):
    """
    loads both PV and NWP dataset
    merges the two datasets by time
    """
    # Load Datasets
    nwp = pd.read_csv(nwp_csv, parse_dates=["init_time_utc"])
    pv = pd.read_csv(pv_csv, parse_dates=["time"])

    # # Focusing on only one pair of coordinates for now
    # longitude = nwp['longitude'] == LONGITUDE
    # latitude = nwp['latitude'] == LATITUDE
    # nwp = nwp.loc[latitude & longitude]

    # Merge dataset
    df = merge_datasets(pv,nwp)

    # Filter time interval
    df = filter_time(df, start_date, end_date)

    # Pre-process data
    df = pre_process(df)
    # df = encode_time(df)

    return df


def merge_datasets(pv_dataset, nwp_dataset):
    """
    Returns one dataset with both pv and nwp information
    """
    # Match times from both datasets
    step = pd.to_timedelta(nwp_dataset["step_hours"], unit="h")
    nwp_dataset['time'] = (nwp_dataset['init_time_utc'] + step).dt.floor("h")

    # Merge both datasets and remove not used columns
    df = pd.merge(pv_dataset, nwp_dataset, how="outer")
    df = df.drop(columns = ['latitude', 'longitude', 'step_hours', 'init_time_utc', 'is_day'])

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


def pre_process(df):
    """
    """
    # Enforce hourly periodicity 
    df = df.set_index("time").sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", name="time")
    df = df.reindex(full_idx) 

    plot_missing_values(df)

    # Fill empty  values
    for c in CHANNELS:
        df[c] = df[c].ffill().bfill()
    df['solar_power'] = df['solar_power'].fillna(0) # right now filled with zeros maybe better options?

    # TODO: Outliers

    # Remove night time
    solpos = solarposition.get_solarposition(full_idx, LATITUDE, LONGITUDE)
    is_day = (solpos["apparent_elevation"] == 0).to_numpy()
    df[is_day] = 0

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


def plot_missing_values(df):
    """
    Plot missing values
    """
    print(f"Summary missing data:", df.isna().sum())
    plot_df = df.isna().transpose()
    plot_df.columns = plot_df.columns.strftime("%Y-%m-%d")

    plt.figure(figsize=(30, 15))
    sb.heatmap(plot_df,
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})
    plt.savefig("missing_values.jpg")
    