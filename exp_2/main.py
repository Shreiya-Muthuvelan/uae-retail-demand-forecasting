import pandas as pd
import re
import mlflow
import mlflow.sklearn
from sklearn.metrics import r2_score
from exp_1.data_preparation import (clean_dataset, merge_datasets,create_weekly_dataset)
from .features import (add_lag_rolling_features,add_calendar_features,remove_missing_lag_rows,add_ramadan_features)
from exp_1.train_test_split import (train_test_split_for_time_series)
from exp_1.model import create_xgboost_pipeline
from exp_1.evaluate import (evaluate_model,print_results)

TRAIN_PATH = "data/train.csv"
STORE_PATH = "data/store.csv"
EXPERIMENT_NAME = ("UAE_Retail_Demand_Forecasting")
RUN_NAME = "XGBoost_Hijri"


FEATURES = ["Store","OpenDays","Promo","SchoolHoliday","StoreType","Assortment","CompetitionDistance","Promo2","PromoInterval",
    "Year","Month","WeekOfYear","Quarter","ramadan_flag","ramadan_week","lag_1","lag_2","lag_4","lag_8","rolling_mean_4","rolling_mean_8"]

df_train = pd.read_csv(TRAIN_PATH)
df_store = pd.read_csv(STORE_PATH)
df_train, df_store = clean_dataset( df_train, df_store)
dataset = merge_datasets(df_train,df_store)

weekly_sales = create_weekly_dataset( dataset)
dataset = add_lag_rolling_features( weekly_sales)
dataset=add_ramadan_features(dataset)
dataset = remove_missing_lag_rows(dataset)
dataset = add_calendar_features( dataset)
print(
    dataset[
        dataset["ramadan_flag"] == 1
    ][
        ["Week", "ramadan_flag", "ramadan_week"]
    ]
    .drop_duplicates()
    .to_string(index=False)
)


train_set, test_set = (train_test_split_for_time_series(dataset))

X_train = train_set[FEATURES]
y_train = train_set["Sales"]
X_test = test_set[FEATURES]
y_test = test_set["Sales"]

pipeline = create_xgboost_pipeline()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)


with mlflow.start_run(run_name=RUN_NAME):
    # Train
    pipeline.fit(X_train, y_train)
    # Predict
    predictions = pipeline.predict(X_test)
    # Add this after your predictions, still inside the mlflow run
    import numpy as np

    test_with_preds = test_set.copy()
    test_with_preds['predicted'] = predictions
    test_with_preds['error'] = np.abs(test_with_preds['Sales'] - test_with_preds['predicted'])

    # Ramadan weeks performance
    ramadan_test = test_with_preds[test_with_preds['ramadan_flag'] == 1]
    non_ramadan_test = test_with_preds[test_with_preds['ramadan_flag'] == 0]

    ramadan_mae = ramadan_test['error'].mean()
    non_ramadan_mae = non_ramadan_test['error'].mean()

    print(f"\n--- RAMADAN vs NON-RAMADAN ERROR ---")
    print(f"MAE during Ramadan weeks:     {ramadan_mae:.2f}")
    print(f"MAE during non-Ramadan weeks: {non_ramadan_mae:.2f}")
    print(f"Ramadan error premium:        {((ramadan_mae/non_ramadan_mae)-1)*100:.1f}%")

    mlflow.log_metric("ramadan_mae", round(ramadan_mae, 2))
    mlflow.log_metric("non_ramadan_mae", round(non_ramadan_mae, 2))

        # Evaluate
    metrics = evaluate_model( y_test,predictions)
    print_results(metrics)
    r2 = r2_score(y_test, predictions)
    print(f"R²: {r2:.4f}")
    mlflow.log_metric("R2", round(r2, 4))

    mlflow.log_param( "model","XGBoost with Ramadan Features")
    mlflow.log_param("n_estimators",500)
    mlflow.log_param("learning_rate",0.05)
    mlflow.log_param("max_depth",6)
    mlflow.log_param("num_stores",dataset["Store"].nunique())
    mlflow.log_param( "feature_count",len(FEATURES))
    mlflow.log_param("forecast_horizon","1 week")
    mlflow.log_param("training_rows",len(train_set))
    mlflow.log_param("testing_rows",len(test_set))
    mlflow.log_param("train_start",str(train_set["Week"].min()))
    mlflow.log_param("train_end",str(train_set["Week"].max()))
    mlflow.log_param("test_start",str(test_set["Week"].min()))
    mlflow.log_param("test_end",str(test_set["Week"].max()))

    mlflow.log_metric("MAE",metrics["MAE"])
    mlflow.log_metric("RMSE",metrics["RMSE"])
    mlflow.log_metric( "MAPE",metrics["MAPE"])

    mlflow.sklearn.log_model(
        pipeline,
        name="xgboost_model",
        skops_trusted_types=[
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBRegressor",
        ],
    )