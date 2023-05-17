# pyright: basic

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import re
from wordcloud import WordCloud


class Visualizer:
    """
    Visualizer for data exploration.
    Checklist:
    - [X] Scatterplots/histograms: features distribution
    - [X] Heat maps: features correlation
    - [X] Bar/swarm/violin plots: features comparison
    - [ ] Line graphs: time series

    Bonus:
    - [X] 3D/multidimensional/multivariate plots
        (e.g. Scatterplot matrix; see this blog for more examples)
    - [ ] Interactive plots (e.g. Plotly)

    NOTE: Each plot should have:
    - [X] A title/caption
    - [X] Axes labels with unit of measurement
    - [X] Legends (if using multiple colors/markers/sizes)
    """

    def __init__(self, df: pd.DataFrame, feats: Dict[str, List[str]]):
        self.df = df.copy()
        self.feats = feats.copy()
        sns.set_palette("deep")

    def plot_topics_kde(self):
        sns.set_palette("deep")
        df_less_row_1 = self.df.loc[2:202]
        # Plot Pclass feature
        g = sns.kdeplot(
            self.df.date_posted[self.df.incident_num == 0],
            fill=True,
            label="Baguio",
        )
        g = sns.kdeplot(
            self.df.loc[(self.df["incident_num"] == 1), "date_posted"],
            fill=True,
            label="Scandal",
        )
        g = sns.kdeplot(
            self.df.loc[(self.df["incident_num"] == 2), "date_posted"],
            fill=True,
            label="Quarantine",
        )
        g = sns.kdeplot(
            self.df.loc[(self.df["incident_num"] == 3), "date_posted"],
            fill=True,
            label="Ladder",
        )
        g = sns.kdeplot(
            df_less_row_1.loc[(self.df["incident_num"] == 4), "date_posted"],
            fill=True,
            label="Other",
        )

        plt.title("Distribution of Topic Tweets Across Time", fontsize=18, pad=20)

        # Show legends
        plt.legend()

        # Add plot labels for Pclass categories
        # labels = ["", "Middle", "Lower"]
        # plt.xticks(sorted(self.df.date_posted_day.unique()), labels)
        # plt.gcf().set_size_inches(12, 12)
        # plt.savefig("kde.png", dpi=100)
        plt.show()

    def plot_boxplot_join(self):
        df_without_account_dup = self.df.drop_duplicates(subset=["account_handle"])
        df_without_account_dup["days_after_election"] = df_without_account_dup[
            "diff_joined_election"
        ]
        sns.boxplot(
            x=df_without_account_dup["days_after_election"],
            orient="h",
        ).set(title="Distribution of Account Creation wrt Election Day")
        plt.savefig("boxplot_acc.png")
        plt.show()

    def plot_boxplot_date_posted(self):
        temp_df = self.df
        temp_df["days_after_election"] = temp_df["diff_date_posted_election"]
        sns.boxplot(
            x=temp_df["days_after_election"],
            orient="h",
        ).set(title="Distribution of Tweet Posting wrt Election Day")
        # plt.savefig("boxplot_post.png")
        plt.show()

    def num_feats_heatmap(self):
        """
        Heatmap of the correlation among numerical features.
        """
        heatmap_feats = [
            "following",
            "followers",
            "likes",
            "replies",
            "retweets",
            "quote_tweets",
            "joined_unix",
            "date_posted_unix",
            "leni_sentiment_num",
            "marcos_sentiment_num",
            "incident_num",
            "account_type_num",
            "country_num",
            "tweet_type_num",
            "content_type_num",
        ]
        ax = sns.heatmap(
            self.df[heatmap_feats].corr(), vmin=-1, vmax=1, cmap="PiYG", annot=True
        )
        ax.set_title("Correlation among the numerical features.")
        # plt.gcf().set_size_inches(14, 14)
        # plt.savefig("twocolor_heatmap.png", dpi=120)
        plt.show()
        print(">>> Plotted heatmap")

    def bargraph_tweets_per_topic(self):
        ax = sns.countplot(
            data=self.df,
            x="incident",
            order=["baguio", "scandal", "quarantine", "ladder", "others"],
        )

        ax.set_title("Number of tweets per incident.")
        # plt.savefig("topics_bar.png")
        plt.show()
        print(">>> Plotted bargraph")

    def bargraph_sentiments(self):
        ax = sns.countplot(
            data=self.df,
            x="leni_sentiment",
            order=["negative", "neutral", "positive"],
        )
        ax.set_title("Number of tweets per Leni sentiment.")
        plt.show()
        ax = sns.countplot(
            data=self.df,
            x="marcos_sentiment",
            order=["negative", "neutral", "positive"],
        )
        ax.set_title("Number of tweets per Marcos sentiment.")

        # plt.savefig("topics_bar.png")
        plt.show()
        print(">>> Plotted bargraph")

    def pairplot_leni_sentiment(self):
        # drop outlier
        df = self.df[self.df["date_posted_std"].abs() <= 3]

        pairplot_feats = [
            "followers_bin",
            "diff_joined_election",
            "diff_date_posted_election",
            "engagement_bin",
            "leni_sentiment_num",
        ]

        g = sns.pairplot(
            data=df[pairplot_feats].astype(np.int64),
            hue="leni_sentiment_num",
            hue_order=[-1, 0, 1],
            palette="deep",
        )
        g.fig.suptitle("Pairplot of some numerical features")
        plt.show()

    def pairplot_has_leni_ref(self):
        # drop outlier
        df = self.df[self.df["date_posted_std"].abs() <= 3]

        pairplot_feats = [
            "followers_bin",
            "diff_joined_election",
            "diff_date_posted_election",
            "engagement_bin",
            "has_leni_ref",
        ]

        g = sns.pairplot(
            data=df[pairplot_feats].astype(np.int64),
            hue="has_leni_ref",
            hue_order=[0, 1],
            palette="deep",
        )
        g.fig.suptitle("Pairplot of some numerical features")
        plt.show()

        print(">>> Plotted pairplots")

    def wordcloud(self):
        entities = {
            "Aika Robredo": [
                "aika",
                "aika diri",
                "aika robredo",
                "aika rob",
                "she admitted",
            ],
            "Bam Aquino": ["bembem"],
            "Bongbong Marcos": ["bbm", "bbmarcos", "marcos"],
            "Gwyneth Chua": ["chua"],
            "Jillian Robredo": [
                "jillian robredo",
                "mrs robredo daughter",
                "hindot ka",
                "jillian",
                "jillrobredo",
                "ma am jill",
            ],
            "Leni Robredo": [
                "kaylenipataytayo",
                "kaylenitalo",
                "leni lugaw",
                "leni robredog",
                "lutangina",
                "mrs robredo",
                "president leni",
                "president leni robredo",
                "vp leni",
                "vice president",
                "withdrawleni",
                "fake vp",
                "fake vp leni",
                "her mom",
                # "len 2x",
                "lenlen",
                "lenlenloser",
                "leni",
                "leni robredo",
                "lenirobredo",
                "lugaw",
                "lutang",
                "lutang ina",
                "lutang ina mo",
                "mama",
                "mama nyo",
                "mom",
                "mother",
                "nanay kong lutang",
                "nanay mong lumulutang",
                "philippines vice president",
                "robredog",
                "saint inamo",
                "sarili niyang ina",
            ],
            "Tricia Robredo": [
                "tricia",
                "tricia robredo",
                "trisha",
                "trisha robredo",
                "vice president daughter",
                "she went straight",
            ],
            "Thinking Pinoy": ["tp"],
            "BBM Supporters": ["bbm supporters", "maka bbm tao"],
            "Communists": ["cpp", "cpp ndf npa", "komunista"],
            "Filipino People": [
                "igorot sa baguio",
                "igorots",
                "igorot people",
                "igorot",
                "igorot friends",
                "igorot native",
                "ilocano",
                "kpatid na igorot",
                "locals",
                "taong bayan",
                "they are good",
                "they are respectful",
                "they value education",
            ],
            "Jillian's Baguio Group": [
                "grupo ni jillian",
                "her camp",
                "her crowd",
                "team nila jillian",
            ],
            "Kakampinks": [
                "baguio fenks",
                "dilapinks",
                "dilawkadiri",
                "dilawan",
                "fenks",
                "kakampikon",
                "kakampwet",
                "kakamdogs",
                "kakampink",
                "kakampinks",
                "kampo ni leni",
                "pink",
                "pinkilawan",
                "pinklawan",
                "supporters nyoga bastos",
            ],
            "Robredo Family": [
                "anak ni leni",
                "anak mo lenlen",
                "anak ni lenlen",
                "anak ni robredo",
                "daughter of robredo",
                "daughter of saint",
                "daughter of lugaw",
                "mga robredo",
                "mga anak niya",
                "robredo",
                "tatlong anak",
            ],
        }

        all_names = [name for entity in entities for name in entities[entity]]

        topics = ["baguio", "ladder", "scandal", "quarantine", "others", "all"]

        for topic in topics:
            self.plot_wordcloud(all_names, topic)

    def plot_wordcloud(self, all_names, topic: str):
        matches = {}
        if topic == "all":
            tweets_str = " ".join(self.df["tweet_rendered"].str.lower())
        else:
            tweets_str = " ".join(
                self.df[self.df["incident"] == topic]["tweet_rendered"].str.lower()
            )

        for name in all_names:
            matches[name] = len(re.findall(name, tweets_str))

        wordcloud = WordCloud(width=1000, height=600)
        wordcloud.generate_from_frequencies(frequencies=matches)
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud)
        plt.title(f"Wordcloud of names in {topic} incident/s.")
        plt.axis("off")

        plt.show()

    def main(self):
        # histograms

        # boxplots
        self.plot_boxplot_join()
        self.plot_boxplot_date_posted()

        # # heatmap
        self.num_feats_heatmap()

        # # bar graphs
        self.bargraph_tweets_per_topic()
        self.bargraph_sentiments()

        # # line graphs
        self.plot_topics_kde()

        # # pairplots
        self.pairplot_leni_sentiment()
        self.pairplot_has_leni_ref()

        # print(self.df.columns)

        # wordclouds
        self.wordcloud()
