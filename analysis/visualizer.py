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
    - [X] Line graphs: time series
        - [ ] time series of tweets

    Bonus:
    - [X] 3D/multidimensional/multivariate plots
        (e.g. Scatterplot matrix; see this blog for more examples)
    - [X] Interactive plots (e.g. Plotly)

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
        df_less_row_1 = self.df.loc[2:202]
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
        plt.yscale("log")
        plt.title(
            "Distribution of Disinformation Incident Tweets Across Time",
            fontsize=18,
            pad=20,
        )
        plt.legend()
        plt.gcf().set_size_inches(14, 12)
        # plt.savefig("kde.png")
        plt.show()

    def plot_boxplot_join(self):
        df_without_account_dup = self.df.drop_duplicates(subset=["account_handle"])
        df_without_account_dup["days_after_election"] = df_without_account_dup[
            "diff_joined_election"
        ]
        df_without_account_dup = df_without_account_dup[
            df_without_account_dup["is_misinfo"] == 1
        ]
        sns.boxplot(
            x=df_without_account_dup["days_after_election"],
            orient="h",
        ).set(title="Distribution of Account Creation wrt Election Day")
        # plt.savefig("boxplot_acc.png")
        plt.show()

    def plot_boxplot_date_posted(self):
        temp_df = self.df
        temp_df["days_after_election"] = temp_df["diff_date_posted_election"]
        temp_df = temp_df[temp_df["is_misinfo"] == 1]
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
        relevant_df = self.df[self.df["is_misinfo"] == 1]
        ax = sns.heatmap(
            relevant_df[heatmap_feats].corr(), vmin=-1, vmax=1, cmap="PiYG", annot=True
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
        relevant_df = self.df[self.df["is_misinfo"] == 1]
        ax = sns.countplot(
            data=relevant_df,
            x="leni_sentiment",
            order=["negative", "neutral", "positive"],
        )
        ax.set_title("Number of tweets per Leni sentiment.")
        plt.show()
        ax = sns.countplot(
            data=relevant_df,
            x="marcos_sentiment",
            order=["negative", "neutral", "positive"],
        )
        ax.set_title("Number of tweets per Marcos sentiment.")

        # plt.savefig("topics_bar.png")
        plt.show()
        print(">>> Plotted bargraph")

    def pairplot_leni_sentiment(self):
        # drop outlier
        relevant_df = self.df[self.df["is_misinfo"] == 1]
        df = relevant_df[relevant_df["date_posted_std"].abs() <= 3]

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
        relevant_df = self.df[self.df["is_misinfo"] == 1]
        df = relevant_df[relevant_df["date_posted_std"].abs() <= 3]

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
        relevant_df = self.df[self.df["is_misinfo"] == 1]
        matches = {}
        if topic == "all":
            tweets_str = " ".join(relevant_df["tweet_rendered"].str.lower())
        else:
            tweets_str = " ".join(
                relevant_df[relevant_df["incident"] == topic][
                    "tweet_rendered"
                ].str.lower()
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

    def cumulative_account_join(self, is_misinfo):
        misinfo_dict = {1: "misinfo", 0: "not_misintfo"}
        relevant_df = self.df[self.df["is_misinfo"] == is_misinfo]
        relevant_df = relevant_df.drop_duplicates(subset=["account_handle"])
        join_month_bins = list(relevant_df["joined_month"])
        join_month_bins.sort()

        first_month = join_month_bins[0]
        last_month = join_month_bins[-1]
        year1, month1 = first_month.split("-")
        yearn = last_month.split("-")[0]
        plotly_df_months = []
        plotly_df_counts = []
        for y in range(int(year1), int(yearn) + 1):
            for m in range(1, 13):
                month_str = f"{y}-{m}"
                count = (relevant_df["joined_month"] == month_str).sum()
                plotly_df_months.append(month_str)
                plotly_df_counts.append(count)
        plotly_df = pd.DataFrame()
        plotly_df["month"] = plotly_df_months
        plotly_df["count"] = plotly_df_counts
        plotly_df["cumulative"] = plotly_df["count"].cumsum()
        plotly_df.to_csv(f"../data/joined_month_{misinfo_dict[is_misinfo]}.csv")

    def compute_tweet_dist(self):
        relevant_df = self.df.loc[1:202]
        date_posted_bins = sorted(
            list(relevant_df["date_posted_day"]),
            key=lambda td: (int(td.split("-")[0]), int(td.split("-")[1])),
        )
        first_day = date_posted_bins[0]
        last_day = date_posted_bins[-1]
        year1 = first_day.split("-")[0]
        yearn = last_day.split("-")[0]
        plotly_df_days = []
        plotly_df_counts = []
        for y in range(int(year1), int(yearn) + 1):
            for d in range(1, 367):
                day_str = f"{y}-{d}"
                count = (relevant_df["date_posted_day"] == day_str).sum()
                plotly_df_days.append(day_str)
                plotly_df_counts.append(count)
        plotly_df = pd.DataFrame()
        plotly_df["day"] = plotly_df_days
        plotly_df["count"] = plotly_df_counts
        plotly_df["cumulative"] = plotly_df["count"].cumsum()
        plotly_df.to_csv(f"../data/tweet_dist_per_day.csv")

        a = plotly_df["day"]
        b = plotly_df["cumulative"]
    
        c = pd.DataFrame({"Day":a, "Count":b})

        g = sns.lineplot(data=c, x="Day", y="Count")
        plt.title("Cumulative Tweet Dist By Day")
        plt.xlabel("Day")

        for index, label in enumerate(g.get_xticklabels()):
            if index % 60 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)

        plt.xticks(rotation = 90, fontsize = 7)
        plt.ylabel("Number of Tweets")
        plt.show()

    def main(self):
        self.compute_tweet_dist()
        self.cumulative_account_join(0)
        self.cumulative_account_join(1)
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
