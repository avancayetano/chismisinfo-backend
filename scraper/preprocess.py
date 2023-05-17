# pyright: basic

import pandas as pd

from typing import Set, List
from scraper_engine import ScraperEngine
from data_types import TweetData


class Preprocessor:
    def process_master_raw_scraped(self, path: str) -> pd.DataFrame:
        df_raw = pd.read_csv(path).drop_duplicates(subset=["tweet_url"])
        df_raw.to_csv("../data/master_scraped.csv", index=False)
        return df_raw

    def process_misinfo_raw(self, path: str) -> pd.DataFrame:
        df_misinfo = (
            pd.read_csv(path, sep="\t")
            .loc[:, ["Tweet URL", "Keywords"]]
            .drop_duplicates(subset=["Tweet URL"], ignore_index=True)
        )

        df_temp = df_misinfo["Tweet URL"].str.split("/", expand=True)
        df_misinfo["Username"] = df_temp[3]
        df_misinfo["Tweet ID"] = df_temp[5]
        df_misinfo.to_csv("../data/misinfo_all.csv", index=False)

        misinfo_tweets: List[TweetData] = []

        deleted_tweets: List[str] = []

        for idx, row in df_misinfo.iterrows():
            try:
                tweet = scraper.get_tweet_info(row["Tweet ID"], row["Keywords"])
                misinfo_tweets.append(tweet)
            except:
                print(f"Error @ URL: {row['Tweet URL']}")
                deleted_tweets.append(row["Tweet URL"])

        df_misinfo_tweets = pd.DataFrame(misinfo_tweets)

        df_deleted = pd.read_csv("../data/master_raw_scraped.csv")
        df_deleted = df_deleted[df_deleted["tweet_url"].isin(deleted_tweets)]
        print(">>> DELETED TWEETS: ")
        print(df_deleted)

        df_misinfo_tweets = pd.concat([df_misinfo_tweets, df_deleted]).drop_duplicates(
            subset=["tweet_url"]
        )

        df_misinfo_tweets.to_csv("../data/master_misinfo.csv", index=False)

        return df_misinfo_tweets

    def get_all_tweets(self, df_scraped: pd.DataFrame, df_misinfo: pd.DataFrame):
        df = pd.concat([df_scraped, df_misinfo]).drop_duplicates(subset=["tweet_url"])
        df.to_csv("../data/master_tweets.csv", index=False)
        return df


if __name__ == "__main__":
    scraper = ScraperEngine()
    preprocessor = Preprocessor()
    df_scraped = preprocessor.process_master_raw_scraped(
        "../data/master_raw_scraped.csv"
    )
    print(">>> ALL SCRAPED TWEETS")
    print(df_scraped)
    df_misinfo_tweets = preprocessor.process_misinfo_raw("../data/misinfo_all_raw.csv")
    print(">>> ALL MISINFO TWEETS")
    print(df_misinfo_tweets)

    df_all_tweets = preprocessor.get_all_tweets(df_scraped, df_misinfo_tweets)
    print(">>> ALL TWEETS")
    print(df_all_tweets)
