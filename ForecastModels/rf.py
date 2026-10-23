from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

from evaluate import *
from pv_nwp import *

# DATASET
PV_CSV = 'Datasets/PV_DTU.csv'
NWP_CSV = 'Datasets/forecast_DTU.csv'

# TRAIN/TEST SPLIT (intervals are inclusive [])
TRAIN_DATE_START = dt.datetime(2025, 10, 20, 0, 0)
TRAIN_DATE_END = dt.datetime(2026, 6, 30, 23, 0)
TEST_DATE_START = dt.datetime(2026, 7, 1, 0, 0)
TEST_DATE_END = dt.datetime(2026, 8, 31, 23, 0)


FORECAST_COLUMNS = [
    "temperature_2m",
    "shortwave_radiation",
    "direct_radiation",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "cloud_cover",
    # "is_day"
    # "hour_sin",
    # "hour_cos",
]

dataset = load_dataset(NWP_CSV, PV_CSV, TRAIN_DATE_START, TEST_DATE_END)
X = dataset[FORECAST_COLUMNS]
y = dataset['solar_power']

# Split Train/Test
X_train = filter_time(X, TRAIN_DATE_START, TRAIN_DATE_END)
y_train = filter_time(y, TRAIN_DATE_START, TRAIN_DATE_END)
X_test = filter_time(X, TEST_DATE_START, TEST_DATE_END)
y_test = filter_time(dataset, TEST_DATE_START, TEST_DATE_END)

model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=0)
model.fit(X_train, y_train)
rf_pred = pd.Series(model.predict(X_test), index=X_test.index)

get_evaluation_metrics(y_test['solar_power'], rf_pred)
plot_predictions(y_test, rf_pred)
 
# fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
# plot_tree(model.estimators_[0],
#                feature_names = FORECAST_COLUMNS, 
#                class_names=["solar_power"],
#                filled = True);
# plt.show()