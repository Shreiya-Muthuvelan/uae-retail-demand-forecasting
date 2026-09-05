import pandas as pd
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

def add_ramadan_features(dataset):
    """
    Add Ramadan-specific features.

    ramadan_flag:
        1 if weekly period overlaps Ramadan
        0 otherwise

    ramadan_week:
        Sequential week number within Ramadan.
        0 for non-Ramadan weeks.
    """

    ramadan_dates = {
        2013: {
            "start": pd.Timestamp("2013-07-10"),
            "end": pd.Timestamp("2013-08-07")
        },
        2014: {
            "start": pd.Timestamp("2014-06-29"),
            "end": pd.Timestamp("2014-07-27")
        },
        2015: {
            "start": pd.Timestamp("2015-06-18"),
            "end": pd.Timestamp("2015-07-16")
        }
    }

    dataset["ramadan_flag"] = 0
    dataset["ramadan_week"] = 0

    week_start = dataset["Week"]
    week_end = week_start + pd.Timedelta(days=6)

    for dates in ramadan_dates.values():

        ramadan_start = dates["start"]
        ramadan_end = dates["end"]

        mask = (
            (week_start <= ramadan_end) &
            (week_end >= ramadan_start)
        )

        ramadan_weeks = sorted(
            dataset.loc[mask, "Week"].unique()
        )

        dataset.loc[mask, "ramadan_flag"] = 1

        week_numbers = {
            week: i + 1
            for i, week in enumerate(ramadan_weeks)
        }

        dataset.loc[mask, "ramadan_week"] = (
            dataset.loc[mask, "Week"].map(week_numbers)
        )

    return dataset
        
def add_eid_features(dataset):
    """
    Add Eid-al-Fitr specific features.

    eid_flag:
        1 if the weekly period contains Eid-al-Fitr
        0 otherwise

    days_to_eid:
        Number of days from the weekly start date to Eid-al-Fitr.
        Positive values = before Eid
        0 = Eid week
        Negative values = after Eid

    post_eid_dip:
        1 for the week immediately following Eid week
        0 otherwise
    """

    eid_dates = {
        2013: pd.Timestamp("2013-08-08"),
        2014: pd.Timestamp("2014-07-28"),
        2015: pd.Timestamp("2015-07-17")
    }

    dataset["eid_flag"] = 0
    dataset["days_to_eid"] = 0
    dataset["post_eid_dip"] = 0

    week_start = dataset["Week"]
    week_end = week_start + pd.Timedelta(days=6)

    for year, eid_date in eid_dates.items():
        # Only consider weeks belonging to this Eid year
        year_mask = week_start.dt.year == year
        # Identify the Eid week
        eid_week_mask = (year_mask &(week_start <= eid_date) & (week_end >= eid_date))
        dataset.loc[eid_week_mask, "eid_flag"] = 1

        # Calculate distance from weekly start to Eid
        dataset.loc[year_mask, "days_to_eid"] = (eid_date - week_start[year_mask]).dt.days

        # Identify the week immediately after Eid
        eid_week_start = ( eid_date- pd.to_timedelta(eid_date.weekday(), unit="D"))
        post_eid_mask = (week_start == eid_week_start + pd.Timedelta(days=7))
        dataset.loc[post_eid_mask, "post_eid_dip"] = 1

    return dataset