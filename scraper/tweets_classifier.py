# pyright: basic

import pandas as pd


class TweetsClassifier:
    """
    Classifies each tweet in universal_tweets_unique as misinfo or not.
    - Positive dataset: misinfo or disinfo tweet (labeled as class=1)
    - Negative dataset: not misinfo and not disinfo (labeled as class=0)
    """

    def __init__(self) -> None:
        self.df_univ = pd.read_csv("../data/analysis-data/universal_tweets_unique.csv")
        self.df_positive = pd.read_csv(
            "../data/analysis-data/misinfo_tweets_labeled.csv"
        )["tweet_url"]

    def classify(self):
        pass

    def main(self):
        pass
