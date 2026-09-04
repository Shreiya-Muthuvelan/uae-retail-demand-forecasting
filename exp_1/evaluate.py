from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)


def evaluate_model(y_true, predictions):

    mae = mean_absolute_error(y_true, predictions)
    rmse = mean_squared_error(y_true, predictions) ** 0.5

    non_zero = y_true != 0

    mape = mean_absolute_percentage_error(
        y_true[non_zero],
        predictions[non_zero]
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }


def print_results(metrics):

    print("\n===== Model Results =====")
    print(f"MAE  : {metrics['MAE']:.2f}")
    print(f"RMSE : {metrics['RMSE']:.2f}")
    print(f"MAPE : {metrics['MAPE']:.4f}")