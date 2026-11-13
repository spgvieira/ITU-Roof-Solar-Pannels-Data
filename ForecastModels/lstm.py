import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf

from evaluate import *
from pv_nwp import *
from config import * 


def train_val_test_split(cfg):
    """
    Loads data
    Splits it into train/val/test
    """
    # Load Datasets
    if cfg.forecast:
        dataset = load_dataset(cfg)
        test_dataset = dataset
    else:
        dataset, test_dataset = load_dataset(cfg)
    
    # Split Train/Validation/Test
    train_set = filter_time(dataset, cfg.lstm.train_set.start, cfg.lstm.train_set.end)
    val_set = filter_time(dataset, cfg.lstm.val_set.start, cfg.lstm.val_set.end)
    test_set = filter_time(test_dataset, cfg.lstm.test_set.start, cfg.lstm.test_set.end)

    return train_set, val_set, test_set


def get_scalers(cfg, training_set):
    """
    Get scalers from training set data only
    """
    scaler_forecast = StandardScaler().fit(training_set[cfg.forecast_columns])
    scaler_history = StandardScaler().fit(training_set[cfg.history_columns])

    return scaler_forecast, scaler_history


def prep_input(cfg, scaler_history, scaler_forecast, df, return_times=False):
    """
    Scale data
    Make padded windows with history/forecast available data
    """
    history = pd.DataFrame(scaler_history.transform(df[cfg.history_columns]),
                           index=df.index, columns=cfg.history_columns)
    forecast = pd.DataFrame(scaler_forecast.transform(df[cfg.forecast_columns]),
                            index=df.index, columns=cfg.forecast_columns)
    y =  pd.DataFrame(scaler_history.transform(df[cfg.target_columns]),
                           index=df.index, columns=cfg.target_columns)

    if return_times:
        X_history, X_forecast, Y_index, weights, times = window_data(cfg, history, forecast, return_times=return_times)
        return [X_history, X_forecast], Y_index, weights, times
    else:
        X_history, X_forecast, Y, weights = window_data(cfg, history, forecast, y)
        return [X_history, X_forecast], Y, weights


def make_metrics(scaler_history):
    """
    """
    std = float(scaler_history.scale_[0])

    def mae_orig(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred) * std, axis=-1)

    return [mae_orig]


def get_model(cfg, scaler_history, params):
    """
    Builds LSTM model architecture
    """
    history = keras.Input((None, len(cfg.target_columns)), name="history")
    forecast = keras.Input((None, len(cfg.forecast_columns)), name="forecast")

    h_in = keras.layers.Masking(mask_value=PAD)(history)
    f_in = keras.layers.Masking(mask_value=PAD)(forecast)

    _, h, c = keras.layers.LSTM(params['units'], return_state=True)(h_in)
    x = keras.layers.LSTM(params['units'], return_sequences=True)(f_in, initial_state=[h, c])
    x = keras.layers.Dropout(params['dropout'])(x)
    out = keras.layers.Dense(1)(x)

    model = keras.Model([history, forecast], out)
    model.compile(
        optimizer=keras.optimizers.Adam(params['learning_rate']), 
        loss=keras.losses.Huber(delta=params['delta']),
        weighted_metrics=make_metrics(scaler_history),
    )

    return model


def train_model(cfg, scaler_history, train, val, params):
    """
    Train model
    """
    X_train, Y_train, weights_train = train
    X_val, Y_val, weights_val = val

    model = get_model(cfg, scaler_history, params)
    callback = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=params['patience'], restore_best_weights=True),
        keras.callbacks.CSVLogger(cfg.run_dir / "training_log.csv"),
    ]

    curve = model.fit(
        X_train, Y_train,
        sample_weight=weights_train,
        validation_data=(X_val, Y_val, weights_val),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        callbacks=callback,
        verbose=1,
    )

    report_training(cfg, curve)

    return model


def predict(model, X_test, scaler_y, lengths):
    """
    Make predictions
    """
    pred_scaled = model.predict(X_test)[..., 0]
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
    pred = np.clip(pred, 0, None)
    return np.concatenate([p[:n] for p, n in zip(pred, lengths)])   


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Create results dir
    setup_results(cfg, args.config, "lstm")

    # Load Data
    train_set, val_set, test_set = train_val_test_split(cfg)

    # Get scalers
    scaler_forecast, scaler_history = get_scalers(cfg, train_set)

    # Prepare input
    X_train, Y_train, weights_train = prep_input(cfg, scaler_history, scaler_forecast, train_set)
    X_val, Y_val, weights_val = prep_input(cfg, scaler_history, scaler_forecast, val_set)
    X_test, Y_test, lengths_test, times_test = prep_input(cfg, scaler_history, scaler_forecast, test_set, return_times=True)

    # Train model
    train = X_train, Y_train, weights_train
    val = X_val, Y_val, weights_val
    model = train_model(cfg, scaler_history, train, val, vars(cfg.lstm))

    # Test model
    predicted = predict(model, X_test, scaler_history, lengths_test)
    idx = np.concatenate([np.asarray(t) for t in times_test])
    predicted = pd.Series(predicted, index=idx).to_frame('solar_power')

    # Store results
    forecast_index = Y_test[0].append(Y_test[1:])
    real = remove_night_time(cfg, test_set.loc[forecast_index])
    predicted = remove_night_time(cfg, predicted)
    store_results(cfg, real, predicted)


if __name__ == "__main__":
    main()