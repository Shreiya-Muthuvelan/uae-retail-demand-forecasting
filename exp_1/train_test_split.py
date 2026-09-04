def train_test_split_for_time_series(dataset):
    """
    Perform chronological train/test split.

    The earliest 80% of the timeline is used for training
    and the latest 20% for testing.
    """

    cutoff_date = dataset["Week"].quantile(0.8)

    train_set = (
        dataset[
            dataset["Week"] <= cutoff_date
        ]
        .copy()
    )

    test_set = (
        dataset[
            dataset["Week"] > cutoff_date
        ]
        .copy()
    )

    print("\n===== Time-Based Split =====")

    print("Cutoff:", cutoff_date)

    print(
        "Train:",
        train_set["Week"].min(),
        "→",
        train_set["Week"].max()
    )

    print(
        "Test:",
        test_set["Week"].min(),
        "→",
        test_set["Week"].max()
    )

    return train_set, test_set