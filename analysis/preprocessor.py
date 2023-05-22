# pyright: basic
import pandas as pd
import numpy as np
from typing import List, Dict
from ast import literal_eval
from sklearn.preprocessing import OrdinalEncoder
from itertools import combinations

import copy
import re
import string


class Preprocessor:
    """
    Preprocessing checklist:
    - [X] Handling missing values/ensuring no missing values
    - [X] Handling outliers
    - [X] Ensuring formatting consistency (date, labels, etc.)
    - [X] Normalization/standardization/scaling
    - [X] Categorical data encoding

    Input
    - feats: {date, num, single_cat, multi_cat, str}
    - df: merge of df_misinfo_labeled and df_univ_tweets

    Output
    - df_preprocessed
    - self.feats (modifies this)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feats: Dict[str, List[str]],
    ):
        self.df = df
        self.feats = feats

    def set_dtypes(self):
        """Set dtypes of num and str columns"""
        self.df[self.feats["num"]] = self.df[self.feats["num"]].fillna(0.0)
        self.df[self.feats["num"]] = self.df[self.feats["num"]].astype(np.float64)
        self.df[self.feats["str"]] = self.df[self.feats["str"]].fillna("")
        self.df[self.feats["str"]] = self.df[self.feats["str"]].astype("string")

        print(self.df[self.feats["str"]].dtypes)

    def handle_missing_values(self):
        """
        Only the 'views' column has empty values. Empty 'views' values are set to 0.
        Empty cat & str feature values are filled with empty string.
        """
        self.df["views"] = 0

        self.df[self.feats["str"]] = self.df[self.feats["str"]].fillna("")
        self.df[self.feats["single_cat"]] = self.df[self.feats["single_cat"]].fillna("")
        self.df[self.feats["multi_cat"]] = self.df[self.feats["multi_cat"]].fillna("")

        print(">>> Handled missing values")

    def lower_case(self, cols: List[str]):
        """Lower case strings in cols (subset of str and cat features)"""
        for col in cols:
            self.df[col] = self.df[col].str.lower()

    def strip_str(self, cols: List[str]):
        """Remove leading and trailing whitespace"""
        for col in cols:
            self.df[col] = self.df[col].str.strip()

    def handle_outliers(self):
        print(">>> We decided not to remove the outliers.")
        print(">>> But here are some outlier numerical values")
        std_cols = list(
            filter(
                lambda col: col.endswith("_std")
                and col not in [f"{f}_std" for f in self.feats["date"]],
                self.feats["num"],
            )
        )
        for col in std_cols:
            mean = round(self.df[col.replace("_std", "")].mean(), 2)
            sd = round(self.df[col.replace("_std", "")].std(), 2)
            outliers = self.df[self.df[col].abs() > 3][
                col.replace("_std", "")
            ].to_list()
            num_outliers = len(outliers)
            print(
                f">>> COLUMN: '{col}' - {num_outliers} - {outliers} - mean: {mean} - sd: {sd}"
            )

    def assert_valid_single_val(self, col: str, categories: List[str]):
        """Ensure col has a valid value"""
        categories = list(map(lambda cat: f"({cat})", categories))
        assert self.df[col].str.match(rf"(^{'|'.join(categories)}$)").all()

    def assert_valid_multi_val(self, col: str, categories: List[str]):
        """Ensure col has a combination of valid values"""
        categories = list(map(lambda cat: f"({cat})", categories))
        possible_cats = r"|".join(categories)
        regex = rf"(^(({possible_cats}), ?)*({possible_cats})$)"
        assert self.df[col].str.match(regex).all()

    def ensure_formatting_consistency(self):
        """
        Ensures formatting consistency for the following manually labeled columns.
        - leni_sentiment
        - marcos_sentiment
        - incident
        - account_type
        - tweet_type
        - content_type
        - keywords
        - alt-text

        Also, cleans the above columns by lower-casing and stripping.

        No need to check for the formats of dates. They are guaranteed to be correct
        because their values were directly taken from the output of the scraper.
        """

        # these are the manually labeled columns (prone to human errors)
        labeled_columns = [
            "leni_sentiment",  # One of: Positive, Negative, or Neutral
            "marcos_sentiment",  # One of: Positive, Negative, or Neutral
            "incident",  # One of: Baguio, Ladder, Scandal, Quarantine, Others
            "account_type",  # One of: Identified, Anonymous, Media
            "tweet_type",  # Combination of: Text, Image, Video, URL, Reply
            "country",  # One of: Unspecified, any alphabetic (no number) string, empty string
            "content_type",  # Combination of: Rational, Emotional, Transactional
            "keywords",  # Any alphanumeric string, should not be empty. NOTE: not categorical data
            "alt-text",  # One of: string enclosed by {}, empty string. NOTE: not categorical data
        ]

        # lower case labeled_columns and
        # remove leading and trailing whitespace
        self.lower_case(labeled_columns)
        self.strip_str(labeled_columns)

        # assertions
        self.assert_valid_single_val(
            "leni_sentiment", ["positive", "negative", "neutral"]
        )
        self.assert_valid_single_val(
            "marcos_sentiment", ["positive", "negative", "neutral"]
        )
        self.assert_valid_single_val(
            "incident", ["baguio", "scandal", "quarantine", "ladder", "others"]
        )
        self.assert_valid_single_val(
            "account_type", ["identified", "anonymous", "media"]
        )
        self.assert_valid_multi_val(
            "tweet_type", ["text", "image", "video", "url", "reply"]
        )
        self.assert_valid_single_val("country", ["unspecified", "[a-zA-Z ]+", ""])
        self.assert_valid_multi_val(
            "content_type", ["rational", "emotional", "transactional"]
        )
        self.assert_valid_single_val("keywords", [".+"])
        self.assert_valid_single_val(
            "alt-text",
            [r"{[\s\S]*}", ""],
        )

        print(">>> Passed all labels format checks")

    def normalize(self, srs: pd.Series) -> pd.Series:
        """Normalize to range [0, 1]"""
        max_val = srs.max()
        min_val = srs.min()
        return (srs - min_val) / (max_val - min_val)

    def standardize(self, srs: pd.Series) -> pd.Series:
        """Get z-score"""
        mean = srs.mean()
        std = srs.std()
        return (srs - mean) / std

    def norm_std_ize(self):
        """
        Normalizes and standardizes num and date columns.

        Creates two new columns (*_norm and *_std) for each num and date columns.
        For date columns, creates another column corresponding to the unix/epoch time value.
        """
        new_num_cols: List[str] = []
        for col in self.feats["num"]:
            col_norm = f"{col}_norm"
            col_std = f"{col}_std"
            self.df[col_norm] = self.normalize(self.df[col])
            self.df[col_std] = self.standardize(self.df[col])

            new_num_cols.extend([col_norm, col_std])

        for col in self.feats["date"]:
            col_unix = f"{col}_unix"
            col_norm = f"{col}_norm"
            col_std = f"{col}_std"
            self.df[col_unix] = self.df[col].astype(np.int64) // 10**9
            self.df[col_norm] = self.normalize(self.df[col_unix])
            self.df[col_std] = self.standardize(self.df[col_unix])

            new_num_cols.extend([col_unix, col_norm, col_std])

        self.feats["num"].extend(new_num_cols)

        print(">>> Standardized and normalized numerical and date columns.")

    def encode_multi_cat(self, row: str) -> str:
        """
        Sample encoding (codes are sorted alphabetically)
        - Tweet type
            - Text => T
            - Text, Image, URL => ITU
            - Text, URL, Image, Reply => IRTU
            - Reply, Video, Text, URL, Image => IRTUV
            - Text, URL, Image, Reply, Video => IRTUV
        - Content type
            - Emotional => E
            - Rational, Emotional => ER
            - Transactional, Rational, Emotional => ERT
            - Rational, Emotional, Transactional => ERT
        """
        vals = sorted(set([val.strip() for val in row.split(",")]))
        code = "".join(map(lambda val: val[0].upper(), vals))
        return code

    def get_multi_categories(self, all_cats: str) -> List[str]:
        combs: List[str] = []
        for i in range(1, len(all_cats) + 1):
            combs.extend(["".join(comb) for comb in combinations(all_cats, i)])

        return combs

    def encode_cat_to_num_feats(self):
        """encode categorical features to numerical"""

        categories = {
            "leni_sentiment": ["negative", "neutral", "positive"],
            "marcos_sentiment": ["negative", "neutral", "positive"],
            "incident": ["baguio", "scandal", "quarantine", "ladder", "others"],
            "account_type": ["anonymous", "identified", "media"],
            "tweet_type": self.get_multi_categories("IRTUV"),
            "content_type": self.get_multi_categories("ERT"),
        }
        print(">>> Categorical to numerical mapping")
        print(categories)

        new_cols: List[str] = []
        for col in self.feats["cat"]:
            col_num = f"{col}_num"
            if col != "country":
                encoder = OrdinalEncoder(categories=[categories[col]]).set_output(
                    transform="pandas"
                )
                self.df[col_num] = encoder.fit_transform(  #  type: ignore
                    self.df[col].to_numpy().reshape(-1, 1)
                )

            else:
                self.df[col_num] = np.where(
                    self.df[col].isin(["", "unspecified"]), 0, 1
                )
            self.feats["num"].append(col_num)
            new_cols.append(col_num)

        self.df["leni_sentiment_num"] = self.df["leni_sentiment_num"] - 1
        self.df["marcos_sentiment_num"] = self.df["marcos_sentiment_num"] - 1

        print(self.df[new_cols])

    def encode_cat_feats(self):
        """
        Encode single_cat and multi_cat columns as categorical data.
        """

        # encode single_cat columns
        for f in self.feats["single_cat"]:
            self.df[f] = self.df[f].astype(pd.CategoricalDtype())

        # encode multi_cat columns
        for f in self.feats["multi_cat"]:
            self.df[f] = (
                self.df[f].astype(pd.StringDtype()).apply(self.encode_multi_cat)
            )
            self.df[f] = self.df[f].astype(pd.CategoricalDtype())

        # merge single_cat and multi_cat
        self.feats["cat"] = self.feats["single_cat"] + self.feats["multi_cat"]
        del self.feats["single_cat"]
        del self.feats["multi_cat"]

        # encode cat to num
        self.encode_cat_to_num_feats()

        print(">>> Encoded categorical values")

    def clean_tweet_content(self):
        """
        Remove the @username tokens at the start which are automatically added
        at the start of the tweet when replying to someone.

        Also, remove the links from the tweet.
        """

        # remove @username's at the start
        self.df["tweet"] = self.df["tweet"].str.replace(
            r"(^(@[^ ]+ )+)", "", regex=True
        )
        self.df["tweet_rendered"] = self.df["tweet_rendered"].str.replace(
            r"(^(@[^ ]+ )+)", "", regex=True
        )

        # this url regex was taken from https://regexr.com/39nr7
        url_regex = r"([(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*))"

        # remove links
        self.df["tweet"] = self.df["tweet"].str.replace(url_regex, "", regex=True)
        self.df["tweet_rendered"] = self.df["tweet_rendered"].str.replace(
            url_regex, "", regex=True
        )

        print(">>> Cleaned tweet content")

    def clean_tweet_emojis(self):
        # Handle Emojis [2]
        url_emoji = "https://drive.google.com/uc?id=1G1vIkkbqPBYPKHcQ8qy0G2zkoab2Qv4v"
        df_emoji = pd.read_pickle(url_emoji)
        df_emoji = {v: k for k, v in df_emoji.items()}

        def emoji_to_word(text):
            for emot in df_emoji:
                text = re.sub(
                    r"(" + emot + ")",
                    "_".join(df_emoji[emot].replace(",", "").replace(":", "").split()),
                    text,
                )
            return text

        # Handle Emoticons [2]
        url_emote = "https://drive.google.com/uc?id=1HDpafp97gCl9xZTQWMgP2kKK_NuhENlE"
        df_emote = pd.read_pickle(url_emote)

        def emote_to_word(text):
            for emot in df_emote:
                text = re.sub(
                    "(" + emot + ")",
                    "_".join(df_emote[emot].replace(",", "").split()),
                    text,
                )
                text = text.replace("<3", "heart")  # not included in emoticons database
                text = text.replace("🪄", "magicwand")
                text = text.replace("🤪", "zanyface")
                text = text.replace("🥳", "parytingface")
                text = text.replace("🤯", "explodinghead")
                text = text.replace("🤭", "facewithhandovermouth")
                text = text.replace("🤮", "facevommiting")
                text = text.replace("🥴", "woozyface")
                text = text.replace("🇵🇭", "philippineflag")
                text = text.replace(" — ", " ")
                text = text.replace(" … ", " ")
                text = text.replace("…", " ")
                text = text.replace(" « ", " ")
                text = text.replace(" “ ", "")
                text = text.replace(" “", " ")
                text = text.replace("“ ", " ")
                text = text.replace("“", "")

            return text

        texts = copy.deepcopy(list(self.df["tweet_rendered"]))

        texts = [emoji_to_word(t) for t in texts]
        texts = [emote_to_word(t) for t in texts]
        texts = [t.lower() for t in texts]

        # remove punctuation
        texts = [t.translate(str.maketrans("", "", string.punctuation)) for t in texts]

        self.df["tweet_lower"] = texts
        print(">>> Cleaned tweet emojis")
        print(self.df.columns)

    def main(self) -> pd.DataFrame:
        """Main driver function"""

        # Preprocessing checklist
        self.handle_missing_values()
        self.ensure_formatting_consistency()
        self.norm_std_ize()
        self.handle_outliers()  # intentional out of order
        self.encode_cat_feats()

        # extra preprocessing steps
        self.clean_tweet_content()
        # self.clean_tweet_emojis()
        self.set_dtypes()

        return self.df
