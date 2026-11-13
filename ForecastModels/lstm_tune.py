import argparse
import pandas as pd
import keras_tuner as kt
from tensorflow import keras

from config import load_config
from lstm import * 


class LSTMHyperModel(kt.HyperModel):
    def __init__(self, cfg, scaler_history):
        super().__init__()
        self.cfg = cfg
        self.scaler_history = scaler_history

    def build(self, hp):
        params = dict(
            units=hp.Int("units", 32, 256, step=32),
            dropout=hp.Float("dropout", 0.0, 0.5, step=0.1),
            lr=hp.Float("lr", 1e-4, 1e-2, sampling="log"),
            huber_delta=hp.Float("huber_delta", 0.5, 2.0, step=0.5),
        )
        return get_model(self.cfg, self.scaler_history, params)

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice("batch_size", [32, 64, 128, 256], default=64)
        return model.fit(*args, batch_size=batch_size, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="best_hyperparameters.yaml")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Get train and validation sets
    train_set, val_set, _ = train_val_test_split(cfg)

    # Get scalers
    scaler_forecast, scaler_history = get_scalers(cfg, train_set)

    # Prepare input
    X_train, Y_train, weights_train = prep_input(cfg, scaler_history, scaler_forecast, train_set)
    X_val, Y_val, weights_val = prep_input(cfg, scaler_history, scaler_forecast, val_set)

    # Search for best hyper parameters
    tuner = kt.Hyperband(
        LSTMHyperModel(cfg, scaler_history),
        objective=kt.Objective("val_mae_orig", direction="min"),
        max_epochs=cfg.lstm.epochs,
        factor=3,
        directory="tuning",
        project_name="lstm_forecast",
        overwrite=True,
        seed=42,
    )

    tuner.search(
        X_train, Y_train,
        sample_weight=weights_train,
        validation_data=(X_val, Y_val, weights_val),
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_mae_orig", mode="min", patience=cfg.lstm.epochs, restore_best_weights=True)],
        verbose=1,
    )

    # Save in csv
    trials = tuner.oracle.get_best_trials(num_trials=args.top)
    rows = [{**t.hyperparameters.values, "val_mae_orig": t.score} for t in trials]
    table = pd.DataFrame(rows)
    table.to_csv("lstm_tuning_results.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()