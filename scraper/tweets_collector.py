# pyright: basic

import pandas as pd
import numpy as np
from typing import List
from scraper_engine import ScraperEngine
from data_types import TweetData


class TweetsCollector:
    """
    Collects all tweet data.

    Available files:
    - master_raw_scraped
        - all scraped tweets, contains duplicates.
        - combination of *_raw_dataset's from avan, harold, and raph's datasets/
        - duplicates are tweets that were scraped using different keywords
        - useful
    - master_scraped
        - master_raw_scraped but without duplicates
        - only the first keyword is retained for duplicate rows
        - useless
    - master_tweets
        - all candidate misinfo + master_scraped, no duplicates
        - need to make a copy of this with duplicates (universal_tweets; see below)
        - useless (use universal_tweets instead) but useful in doing the above bullet
    - master_misinfo
        - all candidate misinfo tweets
        - the same with candidate_misinfo, but without some columns
        - useless (use candidate_misinfo instead)
    - candidate_misinfo_tweets
        - the same as master_misinfo, but complete columns
        - useful

    Output files:
    - universal_tweets
        - universal set of relevant tweets, contains duplicates
        - master_raw_scraped + master_tweets
        - duplicates are tweets that were scraped using different keywords
        - NOTE: these are tweets that were searched (manual or scraped) via keywords
            (and hence, these do not subset misinfo_users_tweets)
    - universal_tweets_unique
        - unique version of universal_tweets
        - some info may be lost (which tweets appear in multiple keywords)
    - labeled_candidate_misinfo
        - Complete candidate_misinfo but only some are labeled
        - Contains spam tweets
        - Contains an extra column: alt-text
    - misinfo_users_tweets (TODO)
        - all the profile tweets of users in misinfo_tweets

    Useful files for analysis (located in data/analysis-data)
    - universal_tweets
        - only used if we want to know which tweets appear in multiple keywords
    - universal_tweets_unique
        - we'll most probably be using this
    - misinfo_tweets_labeled (manual TODO)
        - unique
        - to be manually created from labeled_candidate_misinfo
        - subset of universal_tweets (and universal_tweets_unique)
        - NOTE: only contains manually-labeled columns and tweet_url and tweed_id
        - Get the other data columns by merging misinfo_tweets_labeled and universal_tweets_unique
    - misinfo_users_tweets (TODO)
        - unique
        - subset of misinfo_tweets
    """

    def __init__(self, scraper: ScraperEngine):
        self.scraper = scraper

    def main(self):
        self.collect_universal_tweets()
        self.collect_candidate_misinfo_tweets()

    def collect_candidate_misinfo_tweets(self):
        print(">>> COLLECTING CANDIDATE MISINFO TWEETS")

        # no account_type, content_type
        df_univ_tweets = pd.read_csv(
            "../data/analysis-data/universal_tweets_unique.csv"
        ).drop(columns=["account_bio_rendered", "tweet_rendered"])

        """
        Meaning of columns for context:
        Tweet: Tweet with alt-text
        """

        df_candidate_misinfo_tweets = (
            pd.read_csv("../data/candidate_misinfo_tweets.csv")
            .rename(columns={"Tweet URL": "tweet_url"})
            .dropna(subset="tweet_url")
            .drop_duplicates(subset=["tweet_url"])[
                ["tweet_url", "Tweet", "Account type", "Tweet Type", "Content type"]
            ]
        )  # 211 unique tweets, one entry has no keywords

        # complete and updated labels, leni_sentiment, account_type, content_type
        # just to be consistent, i'll use the tweet content of the scraper
        # i.e. the tweet column of df_univ_tweets
        df_partial_candidates_labeled = pd.read_csv(
            "../data/partial_candidates_labeled.csv"
        ).drop(columns=["tweet_with_alt_text", "tweet"])[
            [
                "tweet_url",
                "leni_sentiment",
                "marcos_sentiment",
                "incident",
                "account_type",
                "tweet_type",
                "content_type",
            ]
        ]  # 189 unique tweets, without News5

        # possibly outdated leni sentiment, but has delete annotations
        df_candidate_misinfo_remarks = pd.read_csv(
            "../data/candidate_misinfo_remarks.csv"
        )[
            ["tweet_url", "Leni Sentiment"]
        ]  # 210 rows not 211 for some reason

        df = (
            df_candidate_misinfo_tweets.merge(
                df_univ_tweets, on="tweet_url", how="left"
            )
            .merge(df_partial_candidates_labeled, on="tweet_url", how="left")
            .merge(df_candidate_misinfo_remarks, on="tweet_url", how="left")
        )

        df["account_type"] = np.where(
            df["account_type"].notna(), df["account_type"], df["Account type"]
        )

        df["content_type"] = np.where(
            df["content_type"].notna(), df["content_type"], df["Content type"]
        )

        df["leni_sentiment"] = np.where(
            df["leni_sentiment"].notna(), df["leni_sentiment"], df["Leni Sentiment"]
        )

        df["tweet_type"] = np.where(
            df["tweet_type"].notna(), df["tweet_type"], df["Tweet Type"]
        )
        # create new alt-text column
        df["alt-text"] = df["Tweet"].str.extract(r"({[\s\S]+})")

        df = df.drop(
            columns=[
                "Tweet",
                "Account type",
                "Content type",
                "Tweet Type",
                "Leni Sentiment",
            ]
        )
        df["marcos_sentiment"] = df["marcos_sentiment"].fillna("Unlabeled")
        df["incident"] = df["incident"].fillna("Unlabeled")

        print(">>> LABELED CANDIDATES")
        print(df)
        print(df.dtypes)
        df.to_csv("../data/labeled_candidate_misinfo.csv", index=False)

    def collect_universal_tweets(self):
        df_master_raw_scraped = pd.read_csv("../data/master_raw_scraped.csv")
        df_master_tweets = pd.read_csv("../data/master_tweets.csv")

        df_univ_tweets = df_master_tweets.rename(
            columns={"keywords": "keywords_l"}
        ).merge(
            df_master_raw_scraped[["tweet_url", "keywords"]].rename(
                columns={"keywords": "keywords_r"}
            ),
            on="tweet_url",
            how="left",
        )

        df_univ_tweets["keywords"] = np.where(
            df_univ_tweets["keywords_r"].notna(),
            df_univ_tweets["keywords_r"],
            df_univ_tweets["keywords_l"],
        )

        df_univ_tweets = df_univ_tweets.drop(columns=["keywords_l", "keywords_r"])
        print(">>> UNIVERSAL TWEETS WITH DUPLICATES")
        print(df_univ_tweets)
        df_univ_tweets.to_csv("../data/analysis-data/universal_tweets.csv", index=False)

        """
        Possible problems
        - Duplicate tweets may have different user data (following, followers, etc)
            - Simple solution: drop duplicates (essentially master_tweet)
        """

        df_univ_tweets_unique = df_univ_tweets.drop_duplicates(subset=["tweet_url"])
        print(">>> UNIVERSAL TWEETS WITHOUT DUPLICATES")
        print(df_univ_tweets_unique)
        df_univ_tweets_unique.to_csv(
            "../data/analysis-data/universal_tweets_unique.csv", index=False
        )

    def get_misinfo_users_tweets(self):
        srs_users = pd.read_csv("../data/misinfo_tweets.csv")[
            "account_handle"
        ].drop_duplicates()
        print(f">>> SCRAPING THE PROFILE TWEETS OF {len(srs_users)} USERS...")
        users_tweets: List[TweetData] = []
        for idx, username in enumerate(srs_users):
            print(f"[{idx}] Searching tweets of {username}")
            tweets = self.scraper.get_user_tweets(username)
            print(f"Found {len(tweets)} tweets!")
            print(f"Remaining users: {len(srs_users) - idx}\n")
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

        df_user_tweets.to_csv("../data/misinfo_users_tweets.csv", index=False)

        print(">>> SCRAPED MISINFO USERS TWEETS")
        print(df_user_tweets)


if __name__ == "__main__":
    scraper = ScraperEngine()
    tweets_collector = TweetsCollector(scraper)
    tweets_collector.main()
