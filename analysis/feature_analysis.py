# pyright: basic

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from typing import Dict, List
import datetime


class FeatureAnalysis:
    """
    Feature Analysis checklist:
    - [X] Feature selection (statistical test, correlation, machine learning)
    - [ ] Dimensionality reduction (PCA, SVD)
    - [X] Feature generation/engineering (includes additional column/s during data collection)
    """

    def __init__(self, df: pd.DataFrame, feats: Dict[str, List[str]]):
        self.df = df
        self.feats = feats

    def feature_selection(self):
        """
        Features with zero variance are removed.
        """
        sel = VarianceThreshold().set_output(transform="pandas")
        selected_cols = sel.fit(self.df[self.feats["num"]]).get_feature_names_out()  # type: ignore
        removed_cols = list(
            filter(lambda col: col not in selected_cols, self.feats["num"])
        )
        print(f">>> Removed features: {removed_cols}")

        self.feats["num"] = list(selected_cols)

    def dimensionality_reduction(self):
        pass

    def feature_engineering(self):
        """
        Add two new features:
        - followers_bin = <10, <100, <1000, <10k, >=10k
        - engagement = likes + replies + retweets + quote_tweets
        - engagement_bin = <5, <25, <125, <625, <3125, >=3125
        - diff_joined_election = joined - election_date (in terms of days)
        - diff_date_posted_election = date_posted - election_date (in terms of days)
        """

        self.df["followers_bin"] = pd.cut(
            self.df["followers"].to_numpy(),
            bins=[-np.inf, 10, 100, 1000, 10000, np.inf],
            labels=[0, 1, 2, 3, 4],
        )

        self.df["engagement"] = (
            self.df["likes"]
            + self.df["replies"]
            + self.df["retweets"]
            + self.df["quote_tweets"]
        )

        self.df["engagement_bin"] = pd.cut(
            self.df["engagement"].to_numpy(),
            bins=[-np.inf, 5, 25, 625, 3125, np.inf],
            labels=[0, 1, 2, 3, 4],
        )

        print(self.df[["engagement", "engagement_bin"]])

        election_date = datetime.datetime(2022, 5, 9, tzinfo=datetime.timezone.utc)
        self.df["diff_joined_election"] = (self.df["joined"] - election_date).dt.days
        self.df["diff_date_posted_election"] = (
            self.df["date_posted"] - election_date
        ).dt.days

        self.feats["num"].extend(
            [
                "followers_bin" "engagement",
                "engagement_bin",
                "diff_joined_election",
                "diff_date_posted_election",
            ]
        )

        print(
            ">>> Added the following columns: followers_bin, engagement, engagement_bin, diff_joined_election, diff_date_posted_election"
        )

    def main(self):
        print("-------- BEGIN: FEATURE ENGINEERING --------")
        self.feature_selection()
        self.dimensionality_reduction()
        self.feature_engineering()
        print("-------- END: FEATURE ENGINEERING --------")
        return self.df
