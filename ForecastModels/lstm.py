import datetime as dt
import numpy as np

from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from evaluate import *
from pv_nwp import *

# DATASET
PV_CSV = 'Datasets/PV.csv'
NWP_CSV = 'Datasets/ECMWF_forecast.csv'

# TRAIN/TEST SPLIT (intervals are inclusive [])
TRAIN_DATE_START = dt.datetime(2025, 10, 16, 0, 0)
TRAIN_DATE_END = dt.datetime(2026, 6, 30, 23, 0)
TEST_DATE_START = dt.datetime(2026, 7, 1, 0, 0)
TEST_DATE_END = dt.datetime(2026, 8, 31, 23, 0)

# MEMORY
HISTORY_PERIOD = 48 
FORECAST_PERIOD = 24

FORECAST_COLUMNS = [
    "temperature_2m",
    "shortwave_radiation",
    "direct_radiation",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "cloud_cover",
    # "is_day"
    "hour_sin",
    "hour_cos",
]
HISTORY_COLUMNS = FORECAST_COLUMNS + ['solar_power']


# Prepare Training data
def get_training_set(train_set_history, train_set_forecast, train_set_y):
    """
    Retrieves train set from original dataset
    Normalizes training set values
    Makes sure each training instance only has access to inputs inside memory window
    """
    X_past, X_future, y = [], [], []

    for i in range(HISTORY_PERIOD, len(train_set_history), FORECAST_PERIOD):  
        X_past.append(train_set_history[i - HISTORY_PERIOD: i])
        X_future.append(train_set_forecast[i: i + FORECAST_PERIOD])
        y.append(train_set_y[i: i + FORECAST_PERIOD])

    return np.stack(X_past), np.stack(X_future), np.stack(y)


# Prepare Test data
def get_test_set(test_set_history, test_set_forecast):
    """
    Retrieves test set from original dataset
    Makes sure each test instance only has access to inputs inside memory window
    Normalizes X_test values (same as done in training)
    """
    X_past, X_future = [], []

    for i in range(HISTORY_PERIOD, len(test_set_history), FORECAST_PERIOD):  
        X_past.append(test_set_history[i - HISTORY_PERIOD:i])
        X_future.append(test_set_forecast[i:i + FORECAST_PERIOD])

    return np.stack(X_past), np.stack(X_future)


# Build the model
def get_model():
    """
    Builds LSTM model architecture
    """
    history = keras.Input((HISTORY_PERIOD, len(HISTORY_COLUMNS)), name="history")
    forecast = keras.Input((FORECAST_PERIOD, len(FORECAST_COLUMNS)), name="forecast")

    _, h, c = keras.layers.LSTM(64, return_state=True)(history)
    x = keras.layers.LSTM(64, return_sequences=True)(forecast, initial_state=[h, c])
    x = keras.layers.Dropout(0.2)(x)
    # x = keras.layers.TimeDistributed(keras.layers.Dense(32, activation="relu"))(x)
    # out = keras.layers.TimeDistributed(keras.layers.Dense(1))(x)
    out = keras.layers.Dense(1)(x)
    out = keras.layers.Reshape((FORECAST_PERIOD,))(out)

    model = keras.Model([history, forecast], out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    
    return model


def main():

    # Load Datasets
    dataset = load_dataset(NWP_CSV, PV_CSV, TRAIN_DATE_START, TEST_DATE_END)
    
    # Split Train/Test
    training_set = filter_time(dataset, TRAIN_DATE_START, TRAIN_DATE_END)
    test_set = filter_time(dataset, TEST_DATE_START, TEST_DATE_END)

    # Scale features
    scaler_forecast = StandardScaler().fit(training_set[FORECAST_COLUMNS])
    scaler_history = StandardScaler().fit(training_set[HISTORY_COLUMNS])
    scaler_y = StandardScaler().fit(training_set[['solar_power']])

    training_history_set_scaled = scaler_history.transform(training_set[HISTORY_COLUMNS])
    training_forecast_set_scaled = scaler_forecast.transform(training_set[FORECAST_COLUMNS])
    training_set_y = scaler_y.transform(training_set[['solar_power']]).ravel()
    test_history_set_scaled = scaler_history.transform(test_set[HISTORY_COLUMNS])
    test_forecast_set_scaled = scaler_forecast.transform(test_set[FORECAST_COLUMNS])

    X_train_past, X_train_future, Y_train = get_training_set(training_history_set_scaled, training_forecast_set_scaled, training_set_y)
    X_test_past, X_test_future = get_test_set(test_history_set_scaled, test_forecast_set_scaled)

    # Train model
    model = get_model()
    # callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit([X_train_past, X_train_future], Y_train, epochs=50, batch_size=32, verbose=1)

    # Test model
    predicted_scaled = model.predict([X_test_past, X_test_future])
    predicted = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1)).reshape(predicted_scaled.shape)
    predicted = np.clip(predicted, 0, None)

    predicted = flatten_predictions(FORECAST_PERIOD, predicted)
    get_evaluation_metrics(test_set[HISTORY_PERIOD:]['solar_power'], predicted)
    plot_predictions(test_set[HISTORY_PERIOD:], predicted)


if __name__ == "__main__":
    main()