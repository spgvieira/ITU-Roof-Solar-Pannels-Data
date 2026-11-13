import pandas as pd
import numpy as np
from pvlib import solarposition
from pyproj import Geod
from tensorflow import keras

PAD = -999.0

def load_dataset(cfg):
    """
    Loads both PV and NWP dataset
    Pre-process the data
    Merges the two datasets by time
    """
    # Load Datasets
    nwp = pd.read_csv(cfg.nwp_csv, parse_dates=["init_time_utc"])
    pv = pd.read_csv(cfg.pv_csv, parse_dates=["time"])

    # Pre-process data
    pv = pre_process_solar(cfg, pv)
    nwp = pre_process_nwp(cfg, nwp, True)

    # Merge dataset
    pv_nwp = merge_datasets(pv, nwp)

    # If we want to train on reanalysis data instead of forecast data
    if not cfg.forecast:
        reanalysis = pd.read_csv(cfg.reanalysis_csv, parse_dates=["init_time_utc"])
        reanalysis = pre_process_nwp(cfg, reanalysis, False)
        pv_reanalysis = merge_datasets(pv, reanalysis)

        return pv_reanalysis, pv_nwp
    
    return pv_nwp


def merge_datasets(pv_dataset, nwp_dataset):
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
    df = filter_time(df, cfg.analysis_time.start, cfg.analysis_time.end)

    # Enforce 0 for nightime values
    solpos = solarposition.get_solarposition(pd.DatetimeIndex(df['time']), cfg.latitude, cfg.longitude)
    is_night = (solpos["apparent_elevation"] <= 0).to_numpy()
    df.loc[is_night, 'solar_power'] = 0

    # Drop hours where there is no solar power measurement
    df = df.dropna(subset=['solar_power'])

    # TODO: Outliers
    df.loc[df['solar_power'] < 0, 'solar_power'] = 0

    return df


def pre_process_nwp(cfg, df, forecast):
    """
    Calculates time of prediction
    Handles grid data
    Removes empty values
    """
    # Sum init_time to step to get actual time
    step = pd.to_timedelta(df["step"], unit="h")
    df['time'] = (df['init_time_utc'] + step).dt.floor("h")
    df = df.drop(columns = ['step', 'init_time_utc'])

    # Handle grid data
    if cfg.grid_data:
        df =  df.set_index(["time", "latitude", "longitude"]).sort_index()
        df = flatten_grid(cfg, df, forecast)
        df = df.reset_index(drop=False)
    else:
        latitude = df['latitude'] == cfg.latitude
        longitude = df['longitude'] == cfg.longitude
        df = df.loc[latitude & longitude]
        df = df.drop(columns = ['latitude', 'longitude'])

    # Filter time
    df = filter_time(df, cfg.analysis_time.start, cfg.analysis_time.end)

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


def flatten_grid(cfg, df, forecast):
    """
    Flattens nwp grid by adding a column for each variable for each coordinate pair
    """
    if forecast:
        wide = df[cfg.forecast_columns].unstack(level=[l for l in df.index.names if l != "time"])
        wide.columns = ["_".join(map(str, c)) for c in wide.columns]
        cfg.forecast_columns = wide.columns
    else:
        wide = df[cfg.reanalysis_columns].unstack(level=[l for l in df.index.names if l != "time"])
        wide.columns = ["_".join(map(str, c)) for c in wide.columns]
        cfg.reanalysis_columns = wide.columns
    return wide.sort_index()


def remove_night_time(cfg, df):
    """
    Removes nightime values
    """
    if 'time' in df:
        solpos = solarposition.get_solarposition(pd.DatetimeIndex(df['time']), cfg.latitude, cfg.longitude)
    else: 
        solpos = solarposition.get_solarposition(pd.DatetimeIndex(df.index), cfg.latitude, cfg.longitude)
    is_night = (solpos["apparent_elevation"] <= 0).to_numpy()
    df.loc[is_night, 'solar_power'] = np.nan
    df = df.dropna()

    return df


def window_data(cfg, history_set, forecast_set, y_set=None, return_times=False):
    """
    Organizes data into windows.
    """
    step = pd.Timedelta("1h")
    X_history, X_forecast, y, times = [], [], [], []

    for t in pd.date_range(history_set.index.min(), forecast_set.index.max(), freq=cfg.forecast_period * step):
        history_start, history_end = history_set.index.searchsorted(t), history_set.index.searchsorted(t + cfg.history_period * step)
        forecast_start, forecast_end = forecast_set.index.searchsorted(t + cfg.history_period * step), forecast_set.index.searchsorted(t + cfg.history_period * step + cfg.forecast_period * step)

        # Not enough history or end of time interval, then continue
        if (history_end - history_start) < cfg.min_hist or (forecast_end - forecast_start) == 0:
            continue

        X_history.append(history_set[history_start:history_end])
        X_forecast.append(forecast_set[forecast_start:forecast_end])
        times.append(forecast_set.index[forecast_start:forecast_end])

        # For testing set
        if y_set is not None:
            y.append(y_set[forecast_start:forecast_end])
        else:
            y.append(forecast_set.index[forecast_start:forecast_end])

    lengths = np.array([len(a) for a in X_forecast])
    weights = (np.arange(cfg.forecast_period)[None, :] < lengths[:, None]).astype("float32")

    # Pad missing values
    X_history = keras.utils.pad_sequences(X_history, maxlen=cfg.history_period, dtype="float32", padding="post", value=PAD)
    X_forecast = keras.utils.pad_sequences(X_forecast, maxlen=cfg.forecast_period, dtype="float32", padding="post", value=PAD)

    if y_set is not None:
        y = keras.utils.pad_sequences(y, maxlen=cfg.forecast_period, dtype="float32", padding="post", value=0.0)
        out = (X_history, X_forecast, y, weights)
    else:
        out = (X_history, X_forecast, y, lengths)

    return out + (times,) if return_times else out