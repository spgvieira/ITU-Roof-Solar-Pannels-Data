import numpy as np
from time import gmtime, strftime
import shutil
from pathlib import Path

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

plt.style.use('fivethirtyeight')

RESULTS_DIR = Path("Results")

def store_results(cfg, cfg_file, test_set, predicted, model=None):
    """
    """
    if model == None:
        run_dir = RESULTS_DIR.joinpath("lstm_run_" + strftime("%Y-%m-%d %H:%M", gmtime()))
    else:
        run_dir = RESULTS_DIR.joinpath("rf_run_" + strftime("%Y-%m-%d %H:%M", gmtime()))
    run_dir.mkdir(parents=True, exist_ok=True)
    # Add config file
    shutil.copy(cfg_file, run_dir / "config.yaml")

    # Add evaluation metrics
    get_evaluation_metrics(run_dir, test_set[cfg.target_columns], predicted)

    # Add plot of test set predictions
    plot_predictions(run_dir, test_set, predicted)

    # Add plot of the tree
    if model != None:
        plot_rf_tree(cfg, run_dir, model)


def plot_predictions(run_dir, real_values, predicted_values):
    """
    Plots predicted vs actual values
    """
    df = real_values.copy()
    df['predicted'] = predicted_values

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['solar_power'],
                             name='Actual', line=dict(width=1.5, color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df['predicted'],
                             name='Predicted', line=dict(width=1.5,  color='orange')))
    fig.update_layout(
            title="PV forecast vs actual",
            xaxis=dict(rangeslider=dict(visible=True)),  # drag-to-filter date slider
            yaxis_title="Solar power (kW)",
            hovermode="x unified",
        )
    fig.write_html(run_dir / "predictions.html")


def get_evaluation_metrics(run_dir, real_values, predicted_values):
    """
    Calculate and record evaluation metrics
    """
    f = open(run_dir / "evaluation_metrics.txt", "w")  

    mae = mean_absolute_error(real_values, predicted_values)
    f.write("Mean absolute error : " + str(mae) + "\n")

    rmse = np.sqrt(mean_squared_error(real_values, predicted_values))
    f.write("Root Mean Square Error (RMSE): " + str(rmse) + "\n")

    r2score = r2_score(real_values, predicted_values)
    f.write("R²: " + str(r2score))

    f.close()


def plot_rf_tree(cfg, run_dir, model, tree_idx=0, max_depth=3):
    """
    Plot RF tree
    """
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
    plot_tree(model.estimators_[0],
                   feature_names = cfg.forecast_columns, 
                   class_names=["solar_power"],
                   filled = True);
    plt.savefig(run_dir / "rf_tree.png", bbox_inches="tight")