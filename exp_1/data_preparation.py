import pandas as pd


def clean_dataset(df_train, df_store):
    df_train = df_train.copy()
    df_store = df_store.copy()

    # Convert Date to datetime
    df_train["Date"] = pd.to_datetime(df_train["Date"])

    # Handle missing values in store dataset
    df_store["CompetitionDistance"] = (df_store["CompetitionDistance"].fillna(df_store["CompetitionDistance"].median()))

    # Track missing competition opening information
    df_store["CompetitionOpenMissing"] = (df_store["CompetitionOpenSinceMonth"].isna().astype(int))
    df_store["PromoInterval"] = (df_store["PromoInterval"] .fillna("None"))
    df_store["Promo2SinceWeek"] = (df_store["Promo2SinceWeek"].fillna(0))
    df_store["Promo2SinceYear"] = ( df_store["Promo2SinceYear"].fillna(0))

    return df_train, df_store


def merge_datasets(df_train, df_store):
    dataset = pd.merge(df_train, df_store, on="Store", how="inner")
    return dataset


def create_weekly_dataset(dataset):
    # Aggregate daily observations into weekly store-level sales.

    dataset = dataset.copy()
    # Create week identifier
    dataset["Week"] = (dataset["Date"].dt.to_period("W").dt.to_timestamp())

    weekly_sales = (dataset.groupby(["Store", "Week"]).agg(
            Sales=("Sales", "sum"),
            OpenDays=("Open", "sum"),
            Promo=("Promo", "max"),
            SchoolHoliday=("SchoolHoliday", "max"),
            StoreType=("StoreType", "first"),
            Assortment=("Assortment", "first"),
            CompetitionDistance=("CompetitionDistance", "first"),
            Promo2=("Promo2", "first"),
            PromoInterval=("PromoInterval", "first")
        ).reset_index())

    return weekly_sales


def data_exploration(dataset):
    print( f"Date range: "f"{dataset['Date'].min()} → {dataset['Date'].max()}")
    print(f"Number of unique stores: "f"{dataset['Store'].nunique()}")