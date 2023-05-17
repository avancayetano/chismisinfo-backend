# pyright: basic

import pandas as pd
import numpy as np
from typing import Dict, List


class TimeSeriesAnalysis:
    """
    Time Series Analysis checklist:
    - [X] Interpolation
    - [X] Binning

    New date columns:
    - joined_day
    - joined_week
    - joined_month
    - joined_year
    - date_posted_day
    - date_posted_week
    - date_posted_month
    - date_posted_year
    """

    def __init__(self, df: pd.DataFrame, feats: Dict[str, List[str]]):
        self.df = df
        self.feats = feats

    def find_outlier_dates(self):
        """
        Columns of interest:
        - joined_std
        - date_posted_std
        """
        df_joined_outliers = self.df[self.df["joined_std"].abs() > 3]
        print(f">>> Num of 'joined' outliers: {df_joined_outliers.shape[0]}")

        df_date_posted_outliers = self.df[self.df["date_posted_std"].abs() > 3]

        print(f">>> Num of 'date_posted' outliers: {df_date_posted_outliers.shape[0]}")
        print(df_date_posted_outliers[["date_posted"]])

    def interpolate(self):
        """There is nothing to interpolate."""
        print(
            "There is nothing to interpolate. There are no empty values in all of date columns."
        )

    def bin(self):
        """
        Bin into year, month, week, day
        - Year bins: 2016, 2017, etc (y)
        - Month bins: 2016-1, 2016-2, etc (y-m)
        - Week bins: 2016-1, 2016-2, etc (y-w)
        - Day bins: 2016-1-1, 2016-1-2, etc (y-m-d)
        """
        new_cols: List[str] = []
        for col in self.feats["date"]:
            col_year = f"{col}_year"
            col_month = f"{col}_month"
            col_week = f"{col}_week"
            col_day = f"{col}_day"

            self.df[col_year] = self.df[col].dt.year
            self.df[col_month] = (
                self.df[col].dt.year.astype("str")
                + "-"
                + self.df[col].dt.month.astype("str")
            )
            self.df[col_week] = (
                self.df[col].dt.year.astype("str")
                + "-"
                + self.df[col].dt.isocalendar().week.astype("str")
            )
            self.df[col_day] = (
                self.df[col].dt.year.astype("str")
                + "-"
                + self.df[col].dt.day.astype("str")
            )

            new_cols.extend([col_year, col_month, col_week, col_day])

        self.feats["cat"].extend(new_cols)
        self.df[new_cols] = self.df[new_cols].astype(pd.CategoricalDtype())
        print(">>> Binned dates")
        print(self.df[new_cols])

    def main(self):
        self.find_outlier_dates()
        self.interpolate()
        self.bin()

        return self.df
