import argparse

from sklearn.ensemble import RandomForestRegressor

from evaluate import *
from pv_nwp import *
from config import * 

def main():
    # Load config file (either default or as argument)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Load datasets
    dataset = load_dataset(cfg)

    # Split Train/Test
    training_set = filter_time(dataset, cfg.train_set.start, cfg.train_set.end)
    test_set = filter_time(dataset, cfg.test_set.start, cfg.test_set.end)

    X_train, y_train = training_set[cfg.forecast_columns], training_set[cfg.target_columns]
    X_test, y_test = test_set[cfg.forecast_columns], test_set[cfg.target_columns]

    # Train model
    model = RandomForestRegressor(n_estimators=cfg.rf.n_estimators,
                                  max_depth=cfg.rf.max_depth, 
                                  random_state=cfg.rf.random_state)
    model.fit(X_train, y_train)

    # Predict
    rf_pred = pd.Series(model.predict(X_test), index=X_test.index)

    # Store results
    store_results(cfg, args.config, y_test, rf_pred, model)


if __name__ == "__main__":
    main()