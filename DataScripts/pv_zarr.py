import numpy as np
import pandas as pd
import xarray as xr

INPUT_CSV = "PV.csv"
OUTPUT_ZARR = "PV.zarr"

TIME_COLUMN = "time"
POWER_COLUMN = "solar_power" 

SITE_CAPACITY_MWP = 0.0001   
SITE_LATITUDE = 55.6741      
SITE_LONGITUDE = 12.6000     
SITE_LOCATION_ID = 1  

def load_csv():
    df = pd.read_csv(INPUT_CSV)

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], utc=True).dt.tz_localize(None)
    df["generation_w"] = df[POWER_COLUMN]

    return df


def csv_to_zarr(df):
    time_utc = df[TIME_COLUMN].values
    n_times = len(time_utc)
 
    generation_mw = (df["generation_w"].values * 1e-6).astype(np.float32)
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


def main():
    df = load_csv()
    csv_to_zarr(df)


if __name__ == "__main__":
    main()