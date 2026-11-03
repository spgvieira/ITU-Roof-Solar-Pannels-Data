import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from evaluate import *
from pv_nwp import *
from config import * 

PAD = -999.0 
MIN_HIST = 10

def get_scalers(cfg, training_set):
    """
    Get scalers from training set data only
    """
    scaler_forecast = StandardScaler().fit(training_set[cfg.forecast_columns])
    scaler_history = StandardScaler().fit(training_set[cfg.target_columns])

    return scaler_forecast, scaler_history


def scale(cfg, scaler_history, scaler_forecast, df):
    """
    Scale data
    """
    history = pd.DataFrame(scaler_history.transform(df[cfg.target_columns]),
                           index=df.index, columns=cfg.target_columns)
    forecast = pd.DataFrame(scaler_forecast.transform(df[cfg.forecast_columns]),
                            index=df.index, columns=cfg.forecast_columns)
    y =  pd.DataFrame(scaler_history.transform(df[cfg.target_columns]),
                           index=df.index, columns=cfg.target_columns)
    return history, forecast, y


def window_data(cfg, history_set, forecast_set, y_set = None):
    """
    Organizes data into windows
    """
    step = pd.Timedelta("1h")
    X_past, X_future, y = [], [], []

    for t in pd.date_range(history_set.index.min(), forecast_set.index.max(), freq=cfg.lstm.forecast_period * step):
        p_start = history_set.index.searchsorted(t)
        p_end = history_set.index.searchsorted(t + cfg.lstm.history_period * step)
        f_start = forecast_set.index.searchsorted(t + cfg.lstm.history_period * step)
        f_end = forecast_set.index.searchsorted(t + cfg.lstm.history_period * step + cfg.lstm.forecast_period * step)

        if (p_end - p_start) < MIN_HIST or (f_end - f_start) == 0:
            continue

        X_past.append(history_set[p_start:p_end])
        X_future.append(forecast_set[f_start:f_end])
        if y_set is not None:
            y.append(y_set[f_start:f_end])
        else:
            y.append(forecast_set.index[f_start:f_end])

    lengths = np.array([len(a) for a in X_future])
    weights = (np.arange(cfg.lstm.forecast_period)[None, :] < lengths[:, None]).astype("float32") 

    X_past = keras.utils.pad_sequences(X_past, maxlen=cfg.lstm.history_period, dtype="float32", padding="post", value=PAD)
    X_future = keras.utils.pad_sequences(X_future, maxlen=cfg.lstm.forecast_period, dtype="float32", padding="post", value=PAD)

    if y_set is not None:
        y = keras.utils.pad_sequences(y, maxlen=cfg.lstm.forecast_period, dtype="float32", padding="post", value=0.0) 
        return X_past, X_future, y, weights
    else:
        return X_past, X_future, y, lengths


def get_model(cfg):
    """
    Builds LSTM model architecture
    """
    history = keras.Input((None, len(cfg.target_columns)), name="history")
    forecast = keras.Input((None, len(cfg.forecast_columns)), name="forecast")

    h_in = keras.layers.Masking(mask_value=PAD)(history)
    f_in = keras.layers.Masking(mask_value=PAD)(forecast)

    _, h, c = keras.layers.LSTM(64, return_state=True)(h_in)
    x = keras.layers.LSTM(64, return_sequences=True)(f_in, initial_state=[h, c])
    x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(1)(x)

    model = keras.Model([history, forecast], out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="huber")
    
    return model


def train_model(cfg, X_train, Y_train, weights_train, X_val, Y_val, weights_val ):
    """
    Train model
    """
    model = get_model(cfg)
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    model.fit(X_train, Y_train, sample_weight=weights_train,
            validation_data=(X_val, Y_val, weights_val), 
            epochs= cfg.lstm.epochs, 
            batch_size= cfg.lstm.batch_size,
            callbacks=[callback], verbose=1)
    
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

    # Load Datasets
    dataset = load_dataset(cfg)
    
    # Split Train/Validation/Test
    training_set = filter_time(dataset, cfg.train_set.start, cfg.train_set.end)
    val_set = filter_time(dataset, cfg.val_set.start, cfg.val_set.end)
    test_set = filter_time(dataset, cfg.test_set.start, cfg.test_set.end)

    # Get scalers
    scaler_forecast, scaler_history = get_scalers(cfg, training_set)

    # Scale sets
    train_history_scaled, train_forecast_scaled, train_y_scaled = scale(cfg, scaler_history, scaler_forecast, training_set)
    val_history_scaled, val_forecast_scaled, val_y_scaled = scale(cfg, scaler_history, scaler_forecast, val_set)
    test_history_scaled, test_forecast_scaled, _ = scale(cfg, scaler_history, scaler_forecast, test_set)

    # Organize data into available data windows 
    X_train_past, X_train_future, Y_train, weights_train = window_data(cfg, train_history_scaled, train_forecast_scaled, train_y_scaled)
    X_val_past, X_val_future, Y_val, weights_val = window_data(cfg, val_history_scaled, val_forecast_scaled, val_y_scaled)
    X_test_past, X_test_future, y_test, lengths_test  = window_data(cfg, test_history_scaled, test_forecast_scaled)

    # Train model
    model = train_model(cfg, 
                    [X_train_past, X_train_future], Y_train, weights_train, 
                    [X_val_past, X_val_future], Y_val, weights_val)

    # Test model
    predicted = predict(model, [X_test_past, X_test_future], scaler_history, lengths_test)

    # Store results
    forecast_index = y_test[0].append(y_test[1:])
    store_results(cfg, args.config, test_set.loc[forecast_index], predicted)


if __name__ == "__main__":
    main()