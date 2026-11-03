import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats
import numpy as np

def plot_nwp_vs_pv_data(df, metric):
    """
    Plot the values for a NWP metric in certain interval of time
    """

    _, ax1 = plt.subplots(figsize=(20, 8))

    ax1.plot(df.index, df[metric], label=metric, color="green", linewidth = 1.5 )
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['solar_power'] , label="Solar Energy (Wh)", color="orange", linewidth = 1.5)

    plt.title("PV Power Generation Vs " + metric)
    plt.show()


def correlation_heatmap(df):
    """
    Plot correlation between different 
    """
    sb.heatmap(df.corr(), annot=True, fmt='.2f',  cmap="crest")
    plt.show()


def box_plot(df, metric):
    """
    """
    sb.boxplot(x=df[metric])
    plt.title('Boxplot ' + metric)
    plt.show()


def scatter_plot(df, metric):
    """
    """
    plt.scatter(df['solar_power'], df[metric])
    plt.xlabel('solar power')
    plt.ylabel(metric)
    plt.title('Scatter Plot of Solar Power vs ' + metric)
    plt.show()


def z_score(df, metric):
    """
    """
    z = np.abs(stats.zscore(df[metric]))

    threshold_z = 2

    outlier_indices = np.where(z > threshold_z)[0]
    # no_outliers = df.drop(outlier_indices)

    # print("Original DataFrame Shape:", df.shape)
    print(metric + "Indetified Outliers:", len(outlier_indices))


def plot_missing_values(df):
    """
    Plot missing values
    """
    print(f"Summary missing data:", df.isna().sum())
    plot_df = df.isna().transpose()
    plot_df.columns = plot_df.columns.strftime("%Y-%m-%d")

    plt.figure(figsize=(30, 15))
    sb.heatmap(plot_df,
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})
    plt.savefig("missing_values.jpg")
    
