import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency

class ApplyChiSquare():
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def get_RCtable(self):
        row_labels = ['Disinformation', 'Factual']
        col_labels = ['negative', 'neutral', 'positive']

        df_misinfo = self.df[self.df['is_misinfo'] == 1]
        df_factual = self.df[self.df['is_misinfo'] == 0]
        self.RCtable = list(map(lambda df: [len(df[df['leni_sentiment'] == col].index) for col in col_labels], [df_misinfo, df_factual]))

        extended_RC = []

        col_major = [row_labels] + [[self.RCtable[0][i], self.RCtable[1][i]] for i in range(3)]
        fig = go.Figure(data=[go.Table(header=dict(values=['']+col_labels),
                 cells=dict(values=col_major))]
                     )
        # fig.show()
        ## I guess lets just draw the table, even the extended one.
        return self.RCtable

    def get_results(self):
        self.results = chi2_contingency(self.RCtable)
        return self.results

    def show_results(self):
        print(self.results)

    def main(self):
        self.get_RCtable()
        self.get_results()
        self.show_results()


if __name__ == '__main__':
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

    self_df = df_misinfo_labeled.merge(
        df_univ_tweets,
        on=["tweet_url", "tweet_id"],
        how="left",
        validate="one_to_one",
    )

    # read and merge factual tweets, add a new column: is_misinfo
    df_factual_labeled = pd.read_csv(
        "../data/analysis-data/factual_tweets_labeled.csv",
    )
    df_factual_scraped = pd.read_csv(
        "../data/analysis-data/factual_tweets_scraped.csv",
        parse_dates=date_feats,
    )
    df_factual_scraped = df_factual_labeled.merge(
        df_factual_scraped,
        on=["tweet_url"],
        how="left",
        validate="one_to_one",
    )


    self_df["is_misinfo"] = [1 for i in range(len(self_df.index))]
    df_factual_scraped["is_misinfo"] = [
        0 for i in range(len(df_factual_scraped.index))
    ]

    # escape the assertion handlers for now
    df_factual_scraped["incident"] = "others"
    df_factual_scraped["account_type"] = "anonymous"
    df_factual_scraped["tweet_type"] = "text"
    df_factual_scraped["content_type"] = "rational"
    df_factual_scraped["country"] = ""
    df_factual_scraped["alt-text"] = ""
    df_factual_scraped["robredo_sister"] = "AJT"

    # looks like concatenation was successful
    self_df = pd.concat([self_df, df_factual_scraped], ignore_index=True)

    # print(">>> INITIAL DATAFRAME")
    # print(self_df)

    chi2 = ApplyChiSquare(self_df)
    chi2.main()