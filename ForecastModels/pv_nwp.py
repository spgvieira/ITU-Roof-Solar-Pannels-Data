import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats

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

    # Focusing on only one pair of coordinates for now
    longitude = nwp['longitude'] == LONGITUDE
    latitude = nwp['latitude'] == LATITUDE
    nwp = nwp.loc[latitude & longitude]

    # Merge dataset
    df = merge_datasets(pv,nwp)

    # Filter time interval
    df = filter_time(df, start_date, end_date)

    # Pre-process data
    df = pre_process(df)
    df = encode_time(df)

    # Save dataset to csv
    save_dataset(df, 'double_check.csv')

    return df


def merge_datasets(pv_dataset, nwp_dataset):
    """
    Returns one dataset with both pv and nwp information
    """
    # Match times from both datasets
    step = pd.to_timedelta(nwp_dataset["step_hours"], unit="h")
    nwp_dataset['time'] = (nwp_dataset['init_time_utc'] + step).dt.floor("h")

    # Merge both datasets and remove not used columns
    df = pd.merge(pv_dataset, nwp_dataset)
    df = df.drop(columns = ['latitude', 'longitude', 'step_hours', 'init_time_utc'])

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

    print("Missing hours:", df["solar_power"].isna().sum())

    # Fill empty  values
    for c in CHANNELS:
        df[c] = df[c].ffill().bfill()

    df['solar_power'] = df['solar_power'].fillna(0)

    # TODO: Outliers

    # TODO: NightTime clean 

    return df


def encode_time(df):
    """
    Encode hour information to find daily periodicity
    """
    hour = df.index.hour
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)

    # TODO: Solar position instead?

    return df


def save_dataset(df, csv_path):
    """
    Save dataset to csv
    """
    df.to_csv(csv_path, index = True)


def plot_nwp_vs_pv_data(df, metric):
    """
    Plot the values for a NWP metric in certain interval of time
    """

    _, ax1 = plt.subplots(figsize=(20, 8))

    ax1.plot(df.index, df[metric], label=metric, color="green", linewidth = 1.5 )
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['solar_power'] , label="Solar Energy (Wh)", color="orange", linewidth = 1.5)

    plt.title("PV Power Generation Vs " + metric)
    plt.show()


def correlation_heatmap(df):
    """
    Plot correlation between different 
    """
    sb.heatmap(df.corr(), annot=True, fmt='.2f',  cmap="crest")
    plt.show()


def box_plot(df, metric):
    """
    """
    sb.boxplot(x=df[metric])
    plt.title('Boxplot ' + metric)
    plt.show()


def scatter_plot(df, metric):
    """
    """
    plt.scatter(df['solar_power'], df[metric])
    plt.xlabel('solar power')
    plt.ylabel(metric)
    plt.title('Scatter Plot of Solar Power vs ' + metric)
    plt.show()


def z_score(df, metric):
    """
    """
    z = np.abs(stats.zscore(df[metric]))

    threshold_z = 2

    outlier_indices = np.where(z > threshold_z)[0]
    # no_outliers = df.drop(outlier_indices)

    # print("Original DataFrame Shape:", df.shape)
    print(metric + "Indetified Outliers:", len(outlier_indices))