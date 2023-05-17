# pyright: basic

"""
NOTE: DEPRECATAED, LEGACY CODE
Preprocesses data before analysis and data exploration.
"""

import pandas as pd
import numpy as np


df_clean_1 = pd.read_csv("../data/Dataset - Group 21 - Data.csv").dropna(
    subset="Tweet URL"
)
df_clean_2 = pd.read_csv("../data/Dataset - Group 21 - New Data.csv").dropna(
    subset="Tweet URL"
)


df_clean = (
    pd.concat([df_clean_1, df_clean_2])[
        ["Tweet URL", "Account type", "Tweet Type", "Content type"]
    ]
    .rename(
        columns={
            "Tweet URL": "tweet_url",
            "Account type": "account_type",
            "Tweet Type": "tweet_type",
            "Content type": "content_type",
        }
    )
    .drop_duplicates(subset="tweet_url")
)

df_final = (
    pd.read_csv("../data/leni_sentiments.csv", parse_dates=["joined", "date_posted"])
    .rename(columns={"Leni Sentiment": "leni_sentiment"})
    .drop(columns=["account_bio_rendered", "tweet_rendered"])
    .dropna(subset="tweet_url")
)

df_final["tweet"] = df_final["tweet"].str.replace(r"(^(@[^ ]+ )+)", "", regex=True)

df_final["marcos_sentiment"] = "Unlabeled"
df_final["incident"] = "Unlabeled"

# join df_final and df_clean
df_final = df_final.merge(df_clean, on="tweet_url", how="left")
df_final.to_csv("../data/final_misinfo.csv", index=False)
print(df_final)
print(df_final.columns)
