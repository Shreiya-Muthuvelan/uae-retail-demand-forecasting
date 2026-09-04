def add_lag_rolling_features(dataset):
    
    """
    Add historical sales lag and rolling mean features.

    All rolling features use shift(1) to prevent
    the current week's sales from leaking into the features.
    """

    dataset = dataset.copy()
    dataset = dataset.sort_values(["Store", "Week"])
    grouped_sales = dataset.groupby("Store")["Sales"]

    # Lag features
    dataset["lag_1"] = grouped_sales.shift(1)
    dataset["lag_2"] = grouped_sales.shift(2)
    dataset["lag_4"] = grouped_sales.shift(4)
    dataset["lag_8"] = grouped_sales.shift(8)

    # Rolling means
    dataset["rolling_mean_4"] = (grouped_sales.transform( lambda x: x.shift(1).rolling(4).mean()))
    dataset["rolling_mean_8"] = (grouped_sales.transform(lambda x: x.shift(1) .rolling(8) .mean()))

    return dataset


def add_calendar_features(dataset):
    # Add standard calendar features
    dataset = dataset.copy()
    dataset["Year"] = dataset["Week"].dt.year
    dataset["Month"] = dataset["Week"].dt.month
    dataset["WeekOfYear"] = (dataset["Week"].dt.isocalendar().week.astype(int))
    dataset["Quarter"] = ( dataset["Week"].dt.quarter)
    return dataset


def remove_missing_lag_rows(dataset):
    # Remove rows without sufficient historical sales
    required_features = ["lag_1","lag_2","lag_4","lag_8","rolling_mean_4","rolling_mean_8"]
    return dataset.dropna( subset=required_features)