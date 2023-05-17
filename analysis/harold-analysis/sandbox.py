# pyright: basic
import pandas as pd


df = pd.read_csv("../data/master_misinfo.csv", parse_dates=["date_posted"])

print(df)
df["new"] = df["tweet"].str.replace(r"(^@[^ ]+ (@[^ ]+ )*)", "", regex=True)

print(df["tweet"])
print(df["new"])

df["new"].to_csv("test.csv")

# for row in df.iterrows():
#     r

print("-----------------------------")
print(df["date_posted"])
df["diff_election"] = (
    df["date_posted"].dt.month - 5 + (df["date_posted"].dt.year - 2022) * 12
)

print(df[["diff_election", "date_posted"]])
