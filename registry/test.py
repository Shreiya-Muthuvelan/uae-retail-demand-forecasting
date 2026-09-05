import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import r2_score
from exp_1.data_preparation import (clean_dataset, merge_datasets,create_weekly_dataset)
from exp_3.features import (add_lag_rolling_features,add_calendar_features,remove_missing_lag_rows,add_ramadan_features,add_eid_features)
from exp_1.train_test_split import (train_test_split_for_time_series)
from exp_1.model import create_xgboost_pipeline
from exp_1.evaluate import (evaluate_model,print_results)

TRAIN_PATH = "data/train.csv"
STORE_PATH = "data/store.csv"
EXPERIMENT_NAME = ("UAE_Retail_Demand_Forecasting")
RUN_NAME = "XGBoost_Hijri+Eid"


FEATURES = ["Store","OpenDays","Promo","SchoolHoliday","StoreType","Assortment","CompetitionDistance","Promo2","PromoInterval",
    "Year","Month","WeekOfYear","Quarter","ramadan_flag","ramadan_week","eid_flag","days_to_eid","post_eid_dip","lag_1","lag_2","lag_4","lag_8","rolling_mean_4","rolling_mean_8"]

df_train = pd.read_csv(TRAIN_PATH)
df_store = pd.read_csv(STORE_PATH)
df_train, df_store = clean_dataset( df_train, df_store)
dataset = merge_datasets(df_train,df_store)

weekly_sales = create_weekly_dataset( dataset)
dataset = add_lag_rolling_features( weekly_sales)
dataset=add_ramadan_features(dataset)
dataset=add_eid_features(dataset)
dataset = remove_missing_lag_rows(dataset)
dataset = add_calendar_features( dataset)

print(
    dataset[
        (dataset["eid_flag"] == 1) |
        (dataset["post_eid_dip"] == 1)
    ][
        [
            "Week",
            "eid_flag",
            "days_to_eid",
            "post_eid_dip"
        ]
    ]
    .drop_duplicates()
    .to_string(index=False)
)

train_set, test_set = (train_test_split_for_time_series(dataset))

X_train = train_set[FEATURES]
y_train = train_set["Sales"]
X_test = test_set[FEATURES]
y_test = test_set["Sales"]

# Load production model without needing the run ID
import mlflow.pyfunc

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.pyfunc.load_model(
    model_uri="models:/UAE-Retail-Demand-XGBoost/Production"
)

# Run inference
sample = X_test.iloc[:5]
preds  = model.predict(sample)
print(preds)