import pandas as pd

df_train = pd.read_csv("data/train.csv")
print("Printing details about train csv")
print(df_train.head(n=5))
print(df_train.info())
print("Printing null values inspection for train csv")
print(df_train.isnull().sum())


df_store = pd.read_csv("data/store.csv")
print("Printing details about store csv")
print(df_store.head(n=5))
print(df_store.info())
print("Printing null values inspection for store csv")
print(df_store.isnull().sum())

# change Date column in train to date time
