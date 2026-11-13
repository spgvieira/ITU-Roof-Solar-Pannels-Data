import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, ParameterSampler

from evaluate import *
from config import load_config
from rf import train_test_split, prep_input
from pv_nwp import *

PARAM_SPACE = {
    "n_estimators": [30, 50, 100, 150, 200],
    "max_features": [0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    "min_samples_leaf": [1, 2, 3, 5, 10],
    "max_depth": [None, 10, 20],
    "max_samples": [None, 0.5, 0.8],
}


def eval_mask(times, daytime_idx, weights):
    T = weights.shape[1]
    m = np.zeros((len(times), T), dtype=bool)
    for i, t in enumerate(times):
        m[i, :len(t)] = pd.DatetimeIndex(t).isin(daytime_idx)
    return m & (weights > 0)


def tune_rf(X, y, mask, n_estimators, n_iter, n_splits, seed):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    records = []
    for i, params in enumerate(ParameterSampler(PARAM_SPACE, n_iter=n_iter, random_state=seed)):
        folds = []
        for tr, va in tscv.split(X):
            m = RandomForestRegressor(n_estimators=n_estimators, random_state=seed,
                                      n_jobs=-1, **params)
            m.fit(X[tr], y[tr])
            tr_m = masked_metrics(y[tr], m.predict(X[tr]), mask[tr])
            va_m = masked_metrics(y[va], m.predict(X[va]), mask[va])
            folds.append({**{f"train_{k}": v for k, v in tr_m.items()},
                          **{f"val_{k}": v for k, v in va_m.items()}})
        df = pd.DataFrame(folds)
        rec = {"params": params, **df.mean().to_dict(), "val_MAE_std": df["val_MAE"].std()}
        records.append(rec)
        print(f"[{i + 1}/{n_iter}] val_MAE={rec['val_MAE']:.4f} "
              f"(+/- {rec['val_MAE_std']:.4f})  {params}")
    return sorted(records, key=lambda r: r["val_MAE"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Split train/test
    train_set, _ = train_test_split(cfg)
    X, Y, weights, times = prep_input(cfg, train_set)

    # Remove night time from validation
    daytime_idx = remove_night_time(cfg, train_set).index
    mask = eval_mask(times, daytime_idx, weights)

    # Search for best hyper parameter
    records = tune_rf(X, Y, mask, args.n_estimators, args.n_iter, args.n_splits, args.seed)

    # Save to csv
    table = pd.DataFrame([{**r, "params": str(r["params"])} for r in records])
    table.to_csv("rf_tuning_results.csv", index=False)
    print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()