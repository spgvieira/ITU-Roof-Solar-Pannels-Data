import argparse

from sklearn.ensemble import RandomForestRegressor

from evaluate import *
from pv_nwp import *
from config import * 


def train_test_split(cfg):
    """
    Loads data
    Splits it into train/test
    """
    # Load Datasets
    if cfg.forecast:
        dataset = load_dataset(cfg)
        test_dataset = dataset
    else:
        dataset, test_dataset = load_dataset(cfg)
    
    # Split Train/Validation/Test
    train_set = filter_time(dataset, cfg.rf.train_set.start, cfg.rf.train_set.end)
    test_set = filter_time(test_dataset, cfg.rf.test_set.start, cfg.rf.test_set.end)

    return train_set, test_set


def prep_input(cfg, df):
    """
    Make padded windows with history/forecast available data
    Flattens input
    """
    X_history, X_forecast, Y, weights, times = window_data(cfg, df[cfg.history_columns], df[cfg.forecast_columns], df[cfg.target_columns], return_times=True)

    if Y.ndim == 3:
        Y = Y.squeeze(-1)

    n = X_history.shape[0]
    X = np.concatenate([X_history.reshape(n, -1), X_forecast.reshape(n, -1)], axis=1)

    return X, Y, weights, times


def predict(model, X_test, T, lengths):
    """
    Make predictions
    """
    rf_pred = model.predict(X_test).reshape(-1, T)
    mask = lengths.reshape(-1) > 0
    rf_pred = rf_pred.reshape(-1)[mask]
    return rf_pred


def get_model(params):
    """
    Get model
    """
    return RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        max_features=params["max_features"],
        min_samples_leaf=params["min_samples_leaf"],
        max_samples=params["max_samples"],
        random_state=params["random_state"],
        n_jobs=-1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Create results dir
    setup_results(cfg, args.config, "rf")

    # Load Data
    train_set, test_set = train_test_split(cfg)

    # Organize data into available data windows 
    X_train, Y_train, weights_train, _ = prep_input(cfg, train_set)
    X_test, Y_test, weights_test, times_test = prep_input(cfg, test_set)

    T = Y_train.shape[1]

    # Get model
    model = get_model(vars(cfg.rf))
    full = weights_train.min(axis=1) == 1
    X_fit, Y_fit = X_train[full], Y_train[full]
    model.fit(X_fit, Y_fit)

    # Test model
    predicted = predict(model, X_test, T, weights_test)
    mask = weights_test.reshape(-1) > 0
    y_true = Y_test.reshape(-1)[mask]
   
    # Restore timestamps
    idx = np.concatenate([np.asarray(t) for t in times_test])
    y_test_series = pd.Series(y_true, index=idx).to_frame('solar_power')
    rf_pred_series = pd.Series(predicted, index=idx).to_frame('solar_power')

    # Store results
    y_test_series = remove_night_time(cfg, y_test_series)
    rf_pred_series = remove_night_time(cfg, rf_pred_series)
    store_results(cfg, y_test_series, rf_pred_series, model)


if __name__ == "__main__":
    main()