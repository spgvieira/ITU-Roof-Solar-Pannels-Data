import numpy as np
from time import gmtime, strftime
import shutil
from pathlib import Path
import pandas as pd

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

plt.style.use('fivethirtyeight')

RESULTS_DIR = Path("Results")

def setup_results(cfg, cfg_file, model):
    """
    Create results dir for model run
    Copy config file used into it
    """
    # Create directory
    if not hasattr(cfg, "run_dir") :
        run_dir = RESULTS_DIR.joinpath(model + "_run_" + strftime("%Y-%m-%d %H:%M", gmtime()))
        cfg.run_dir = run_dir
    else:
        run_dir = RESULTS_DIR.joinpath(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Add config file
    shutil.copy(cfg_file, run_dir / "config.yaml")

    cfg.run_dir = run_dir

def store_results(cfg, test_set, predicted, model):
    """
    """
    # Add evaluation metrics
    get_evaluation_metrics(cfg.run_dir, test_set[cfg.target_columns], predicted)

    # Add plot of test set predictions
    plot_predictions(cfg.run_dir, test_set, predicted)

    # Add plot of the tree
    if model != None:
        plot_rf_tree(cfg, model)


def plot_predictions(run_dir, real_values, predicted_values):
    """
    Plots predicted vs actual values
    """
    df = real_values.copy()
    df['predicted'] = predicted_values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['solar_power'],
        name='Actual',
        mode='lines',
        line=dict(width=1.5, color='green'),
        marker=dict(size=5, color='green', symbol='circle'),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['predicted'],
        name='Predicted',
        mode='lines',
        line=dict(width=1.5, color='orange'),
    ))
    fig.update_layout(
        title="PV forecast vs actual",
        xaxis=dict(rangeslider=dict(visible=True)),
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


def plot_rf_tree(cfg, model, tree_idx=0, max_depth=3):
    """
    Plot one tree of the RF, truncated to max_depth levels
    """
    hist = [f"{col}_t-{cfg.history_period - t}h" for t in range(cfg.history_period) for col in cfg.target_columns]
    fut = [f"{col}_t+{t}h" for t in range(cfg.forecast_period) for col in cfg.forecast_columns]
    names = hist + fut

    fig, ax = plt.subplots(figsize=(20, 10), dpi=150)
    plot_tree(model.estimators_[tree_idx],
              feature_names=names,
              max_depth=max_depth,
              filled=True,
              rounded=True,
              fontsize=8,
              ax=ax)
    fig.savefig(cfg.run_dir / "rf_tree.png", bbox_inches="tight")
    plt.close(fig)


def report_training(cfg, history):
    df = pd.DataFrame(history.history)
    best = df["val_loss"].idxmin()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df[["loss", "val_loss"]].plot(ax=axes[0], title="Loss (Huber, scaled)", linewidth=1.0, fontsize=10)
    df[["mae_orig", "val_mae_orig"]].plot(ax=axes[1], title="MAE (original units)", linewidth=1.0, fontsize=10)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.axvline(best, ls="--", c="gray")
        ax.title.set_size(15)
        ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(cfg.run_dir / "training_curves.png", dpi=150)


def compute_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

def masked_metrics(y, pred, mask):
    """y, pred: (n_days, T); mask: (n_days, T) bool of hours to score."""
    return compute_metrics(y[mask], pred[mask])