# pyright: basic

import pandas as pd
from typing import List


from preprocessor import Preprocessor
from nlp import NLP
from time_series import TimeSeriesAnalysis
from visualizer import Visualizer
from feature_analysis import FeatureAnalysis


class DataExplore:
    """
    Data exploration pipeline
    1. Preprocess
    2. Feature engineering
    3. Visualization
    """

    def __init__(self):
        """
        Initialized attributes:
        - df
        - feats: {date, num, single_cat, multi_cat, str}
        """

        df_misinfo_labeled = pd.read_csv(
            "../data/analysis-data/misinfo_tweets_labeled.csv",
        )

        date_feats = ["joined", "date_posted"]
        df_univ_tweets = pd.read_csv(
            "../data/analysis-data/universal_tweets_unique.csv",
            parse_dates=date_feats,
        )

        self.df = df_misinfo_labeled.merge(
            df_univ_tweets,
            on=["tweet_url", "tweet_id"],
            how="left",
            validate="one_to_one",
        )

        # read control tweets, add a new column: is_misinfo
        df_control_scraped = pd.read_csv(
            "../data/analysis-data/control_tweets_scraped.csv",
            parse_dates=date_feats,
        )
        self.df["is_misinfo"] = [1 for i in range(len(self.df.index))]
        df_control_scraped["is_misinfo"] = [
            0 for i in range(len(df_control_scraped.index))
        ]

        # escape the assertion handlers for now
        df_control_scraped["leni_sentiment"] = [
            "neutral" for i in range(len(self.df.index))
        ]
        df_control_scraped["marcos_sentiment"] = [
            "neutral" for i in range(len(self.df.index))
        ]
        df_control_scraped["incident"] = ["others" for i in range(len(self.df.index))]
        df_control_scraped["account_type"] = [
            "anonymous" for i in range(len(self.df.index))
        ]
        df_control_scraped["tweet_type"] = ["text" for i in range(len(self.df.index))]
        df_control_scraped["content_type"] = [
            "rational" for i in range(len(self.df.index))
        ]
        df_control_scraped["country"] = ["" for i in range(len(self.df.index))]
        df_control_scraped["alt-text"] = ["" for i in range(len(self.df.index))]

        # looks like concatenation was successful
        self.df = pd.concat([self.df, df_control_scraped], ignore_index=True)

        feats: List[str] = list(self.df.columns)
        num_feats = [
            "following",
            "followers",
            "likes",
            "replies",
            "retweets",
            "quote_tweets",
            "views",
            "has_leni_ref",
            "is_misinfo",
        ]

        single_cat_feats = [
            "leni_sentiment",
            "marcos_sentiment",
            "incident",
            "account_type",
            "country",
        ]
        multi_cat_feats = ["tweet_type", "content_type"]

        str_feats = list(
            filter(
                lambda f: f
                not in [*date_feats, *num_feats, *multi_cat_feats, *single_cat_feats],
                feats,
            )
        )

        self.feats = {
            "date": date_feats,
            "num": num_feats,
            "single_cat": single_cat_feats,
            "multi_cat": multi_cat_feats,
            "str": str_feats,
        }

        print(">>> INITIAL DATAFRAME")
        print(self.df)
        print(self.feats)

    def main(self):
        pass
        #print("-------- BEGIN: PREPROCESSING --------")
        # Preprocessing...
        #preprocessor = Preprocessor(self.df, feats=self.feats)
        #self.df = preprocessor.main()
        #print("-------- END: PREPROCESSING --------")

        #print("------------ BEGIN: NLP ------------")
        #nlp = NLP(self.df, self.feats)
        #self.df = nlp.main()
        #print("------------- END: NLP -------------")

        #print("-------------BEGIN: TIME SERIES ANALYSIS-------------")
        #time_series = TimeSeriesAnalysis(self.df, self.feats)
        #self.df = time_series.main()
        #print("-------------END: TIME SERIES ANALYSIS-------------")

        #print("-------------BEGIN: FEATURE ANALYSIS -------------")
        #feat_analysis = FeatureAnalysis(self.df, self.feats)
        #self.df = feat_analysis.main()
        #print("-------------END: FEATURE ANALYSIS-------------")

        print("-------------BEGIN: VISUALIZER-------------")
        visualizer = Visualizer(self.df, self.feats)
        visualizer.main()
        print("-------------END: VISUALIZER-------------")


if __name__ == "__main__":
    explore = DataExplore()
    explore.main()
