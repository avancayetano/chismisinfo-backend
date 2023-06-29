# pyright: basic

"""
This should be under preprocessing

output a csv file like the scraper
input are five sheets
"""
import sys
import pandas as pd
from scraper_engine import ScraperEngine
from typing import List


def make_factual_df():
    csv_file_paths = [
        "../data/five-factual-data/not_all_factual_baguio.csv",
        "../data/five-factual-data/factual_scandal.csv",
        "../data/five-factual-data/factual_quarantine.csv",
        "../data/five-factual-data/factual_ladder.csv",
        "../data/five-factual-data/factual_others.csv",
    ]
    csv_dfs = [pd.read_csv(path) for path in csv_file_paths]
    print("----csv transformed to dfs----")

    csv_tweets = [make_sheet_df(csv_df) for csv_df in csv_dfs]
    print("----csv transformed to tweets----")

    tweets = [
        item for sublist in csv_tweets for item in sublist
    ]  # double list comp 0o0
    tweets_df = pd.DataFrame(tweets)
    tweets_df.to_csv("../data/factual_dataset_testing.csv")


def make_sheet_df(csv_df: pd.DataFrame):
    scraper = ScraperEngine()
    tweets = []
    all_errors = []
    for column in csv_df.columns:
        col_errors = []
        keyword = " ".join(str(column).split("_")[:-1])
        # if keyword == "let me educate you igorot":
        #     continue
        for tweet_url in csv_df[str(column)]:
            if str(tweet_url).lower() == "nan":
                break
            tweet_id = str(tweet_url).split("/")[-1]
            # tweets.append(scraper.get_tweet_info(tweet_id, keyword))
            try:
                tweets.append(scraper.get_tweet_info(tweet_id, keyword))
            except KeyboardInterrupt:
                sys.exit()
            except Exception as e:
                print(e)
                all_errors.append(tweet_url)
                col_errors.append(tweet_url)
                continue
        print(f"{column} done! except {col_errors}")

    print(f"All the errors for this incident urls {all_errors}")
    return tweets


make_factual_df()
