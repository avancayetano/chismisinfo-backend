# pyright: basic

import pandas as pd
from typing import List
from scraper_engine import ScraperEngine
from data_types import TweetData

if __name__ == "__main__":
    scraper = ScraperEngine()

    srs_users = pd.read_csv("../data/final_misinfo.csv")[
        "account_handle"
    ].drop_duplicates()

    users_tweets: List[TweetData] = []
    for idx, username in enumerate(srs_users):
        print(f">>> Searching tweets of {username} | [{idx}]")
        tweets = scraper.get_user_tweets(username)
        print(f">>> Found {len(tweets)} tweets!")
        print(f">>> Remaining users: {len(srs_users) - idx}")
        print("-----------------------------------------------")
        users_tweets.extend(tweets)

    df_user_tweets = (
        pd.DataFrame(users_tweets)
        .drop(
            columns=[
                "keywords",
                "account_name",
                "account_bio",
                "account_bio_rendered",
                "account_verified",
                "joined",
                "following",
                "followers",
                "location",
                "tweet_rendered",
            ]
        )
        .drop_duplicates(subset="tweet_id")
    )
    print(df_user_tweets)
    df_user_tweets.to_csv("../data/misinfo_users_tweets.csv", index=False)
