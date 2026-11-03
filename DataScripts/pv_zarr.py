import numpy as np
import pandas as pd
import xarray as xr
from pvlib import solarposition

INPUT_CSV = "Datasets/PV/PV_DTU.csv"
OUTPUT_ZARR = "PV_dtu.zarr"

TIME_COLUMN = "time"
POWER_COLUMN = "solar_power" 

SITE_CAPACITY_MWP = 0.0075 
SITE_LATITUDE = 55.69    
SITE_LONGITUDE = 12.10   
SITE_LOCATION_ID = 1  

def load_csv():
    df = pd.read_csv(INPUT_CSV)

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], utc=True).dt.tz_localize(None)
    df["generation_kw"] = df[POWER_COLUMN]

    return df


def csv_to_zarr(df):
    time_utc = df[TIME_COLUMN].values
    n_times = len(time_utc)

    df = pre_process(df)
 
    generation_mw = (df["generation_kw"].values * 1e-3).astype(np.float32)
    gen = generation_mw.reshape(n_times, 1)
    cap = np.full((n_times, 1), SITE_CAPACITY_MWP, dtype=np.float32)
 
    ds = xr.Dataset(
        data_vars={
            "generation_mw": (("time_utc", "location_id"), gen),
            "capacity_mwp": (("time_utc", "location_id"), cap),
        },
        coords={
            "time_utc": time_utc.astype("datetime64[ns]"),
            "location_id": np.array([SITE_LOCATION_ID], dtype=np.int64),
            "latitude": ("location_id", np.array([SITE_LATITUDE], dtype=np.float64)),
            "longitude": ("location_id", np.array([SITE_LONGITUDE], dtype=np.float64)),
        },
    )

    ds.to_zarr(OUTPUT_ZARR, mode="w")
    print(ds)
    print(f"\nWrote {OUTPUT_ZARR} with {ds.sizes['time_utc']} timesteps.")


def pre_process(df):
    """
    """
    # Enforce hourly periodicity  
    df = df.set_index("time").sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", name="time")
    df = df.reindex(full_idx) 

    df['solar_power'] = df['solar_power'].fillna(0) # right now filled with zeros maybe better options?

    # Remove night time
    solpos = solarposition.get_solarposition(full_idx, SITE_LATITUDE, SITE_LONGITUDE)
    is_night = (solpos["apparent_elevation"] == 0).to_numpy()
    df[is_night] = 0

    return df

def main():
    df = load_csv()
    csv_to_zarr(df)


if __name__ == "__main__":
    main()