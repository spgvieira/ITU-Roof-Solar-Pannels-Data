import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
plt.style.use('fivethirtyeight')


def flatten_predictions(horizon, predicted_values):
    """
    """
    predicted = []
    for period in predicted_values:
        for i in range(horizon):
            predicted.append(period[i])
            
    return predicted


def plot_predictions(real_values, predicted_values):
    """
    Plots predicted vs actual values
    """
    df = real_values.copy()
    df['predicted'] = predicted_values

    plt.plot(df.index, df['solar_power'],  label='actual', color="green",)
    plt.plot(df.index, df['predicted'],  label='predicted', color="orange")
    plt.title('Solar Power')
    plt.xlabel('Time')

    plt.xticks(rotation=45)

    plt.legend()
    plt.show()


def get_evaluation_metrics(real_values, predicted_values):
    """
    Calculate and print evaluation metrics
    """
    
    mae = mean_absolute_error(real_values, predicted_values)
    print("Mean absolute error : " + str(mae))

    rmse = np.sqrt(mean_squared_error(real_values, predicted_values))
    print("Root Mean Square Error (RMSE):", rmse)

    r2score = r2_score(real_values, predicted_values)
    print(f"R²: {r2score}")


