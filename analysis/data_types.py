# pyright: basic

import pandas as pd

from dataclasses import dataclass
from typing import List


@dataclass
class DataFrameAnalysis:
    """
    Might need this later for easier managing of data.
    """

    data: pd.DataFrame
    feats: List[str]
    date_feats: List[str]
    num_feats: List[str]
    cat_feats: List[str]

    def set_df(self, path: str):
        self.data = pd.read_csv(path, parse_dates=self.date_feats)
