
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn3, venn3_circles


class AddtionalGraphs():
    '''
    What graphs
    two double bar graphs
    contingency table??? no already has bar graph...
    the double histo!
    smol kdes
    '''
    def __init__(self, df: pd.DataFrame):
        self.df = df
        sns.set_palette("deep")

    def sentiment_barplots(self):
        col_labels = ['Negative', 'Neutral', 'Positive']
        row_labels = ['Disinformation', 'Non-disinformation']
        leni_marcos = ['Leni', 'Marcos']
        df_misinfo = self.df[self.df['is_misinfo'] == 1]
        df_factual = self.df[self.df['is_misinfo'] == 0]
        leni_rc_table = list(map(lambda df: [len(df[df['leni_sentiment'] == col].index) for col in col_labels], [df_misinfo, df_factual]))
        marcos_rc_table = list(map(lambda df: [len(df[df['marcos_sentiment'] == col].index) for col in col_labels], [df_misinfo, df_factual]))
        # top_barplot = {'Leni':leni_rc_table[0], 'Marcos':marcos_rc_table[0]}
        # bot_barplot = {'Leni':leni_rc_table[1], 'Marcos':marcos_rc_table[1]}
        # top_and_bot = {'Disinformation': top_barplot, 'Factual': bot_barplot}

        fig, ax = plt.subplots(2, 1, figsize=(6,4), dpi=120)
        
        ax[0].set_title(f'{row_labels[0]} Tweets')
        ax[1].set_title(f'{row_labels[1]} Tweets')
        ax[0].legend(loc=1)
        ax[1].legend(loc=1)

        df_top = pd.DataFrame([leni_rc_table[0], marcos_rc_table[0]], columns=col_labels)
        df_top = df_top.stack().reset_index()
        df_top.columns = ['Presidentiable', 'Sentiment', 'Count']
        df_top['Presidentiable'] = df_top['Presidentiable'].map(arg=lambda x: leni_marcos[x])
        sns.barplot(y='Count', x='Presidentiable', palette = sns.color_palette("hls", 8), hue='Sentiment', data=df_top, ax=ax[0])

        df_bot = pd.DataFrame([leni_rc_table[1], marcos_rc_table[1]], columns=col_labels)
        df_bot = df_bot.stack().reset_index()
        df_bot.columns = ['Presidentiable', 'Sentiment', 'Count']
        df_bot['Presidentiable'] = df_bot['Presidentiable'].map(arg=lambda x: leni_marcos[x])
        sns.barplot(y='Count', x='Presidentiable', palette = sns.color_palette("hls", 8), hue='Sentiment', data=df_bot, ax=ax[1])

        fig.tight_layout()
        plt.show()

    def incidents_pie_chart(self):
        # declaring data
        relevant_df = self.df.loc[0:202]
        keys = ['Baguio', 'Scandal', 'Quarantine', 'Ladder', 'Others', 'others']
        data = [len(relevant_df[relevant_df['incident']==key].index) for key in keys]

        data = data[:-2] + [data[-1]+data[-2]]
        keys = keys[:-1]
        
        # define Seaborn color palette to use
        palette_color = sns.color_palette('deep')
        
        # plotting data on chart
        plt.pie(data, labels=keys, colors=palette_color, autopct='%.0f%%', textprops={'fontsize': 14})
        
        # displaying chart
        plt.title('Distribution of Disinformation Tweets Across Incidents')
        plt.show()

    def sisters_venn_diagram(self):
        relevant_df =self.df.loc[0:202]
        seven_values = ['A','J','AJ','T','AT','JT','AJT']
        sisters = ['Aika','Jillian','Tricia']
        counts = [len(relevant_df[relevant_df['robredo_sister']==seven].index) for seven in seven_values]

        out = venn3(subsets = tuple(counts), set_labels = tuple(sisters))
        venn3_circles(subsets=tuple(counts))
        for text in out.set_labels:
            text.set_fontsize(16)
        for text in out.subset_labels:
            text.set_fontsize(16)
        
        plt.title('Disinformation Tweets Targeting a Specific Robredo')
        plt.show()
    def plot_four_distributions(self):
        '''Baguio, Scandal, Quarantine, Ladder, and Overall'''
        import datetime as dt
        import time
        relevant_df = self.df.loc[2:202]

        fig, ax = plt.subplots(2, 2, figsize=(9,4), dpi=120)

        g = sns.kdeplot(
            relevant_df.date_posted[relevant_df.incident == 'Baguio'],
            fill=True,
            label="Baguio",
            ax=ax[0][0]
        )
        g = sns.kdeplot(
            relevant_df.loc[(relevant_df["incident"] == 'Scandal'), "date_posted"],
            fill=True,
            label="Scandal",
            color='orange',
            ax=ax[0][1]
        )
        g = sns.kdeplot(
            relevant_df.loc[(relevant_df["incident"] == 'Quarantine'), "date_posted"],
            fill=True,
            label="Quarantine",
            color='green',
            ax=ax[1][0]
        )
        g = sns.kdeplot(
            relevant_df.loc[(relevant_df["incident"] == 'Ladder'), "date_posted"],
            fill=True,
            label="Ladder",
            color='red',
            ax=ax[1][1]
        )

        incidents = ['Baguio', 'Scandal', 'Quarantine', 'Ladder']
        for i, axes in enumerate([axes for sublist in ax for axes in sublist]):
            anchor_dates = ['2022-01','2022-05','2022-09','2023-01']
            unix_dates = [tuple(item.split('-')+['01']) for item in anchor_dates]
            print(unix_dates)
            axes.set_xticks(anchor_dates)
            axes.set_xticklabels(anchor_dates)
            axes.set_yticks([])
            axes.set_ylabel(incidents[i])
            axes.set_xlabel('Date Posted')
        fig.tight_layout()
        plt.show()
        
    def tweet_distribution1(self):
        '''Baguio, Scandal, Quarantine, Ladder, and Overall'''
        relevant_df = self.df.loc[2:202]

        fig, ax = plt.subplots(2, 2, figsize=(15,5), dpi=120)
        
    
        plt.title("Incident Tweets Distribution", fontsize=18)

        g = sns.displot(
            relevant_df.date_posted[relevant_df.incident == 'Baguio'],
            fill=True,
            color='crimson',
            label="Baguio",
            ax = ax[0][0]
            )

        g = sns.displot(
            relevant_df.loc[(relevant_df["incident"] == 'Scandal'), "date_posted"],
            fill=True,
            color='limegreen',
            label="Scandal",
            kind="kde",
            ax = ax[0][1]
        )

        g = sns.displot(
            relevant_df.loc[(relevant_df["incident"] == 'Quarantine'), "date_posted"],
            color='blue',
            fill=True,
            label="Quarantine",
            kind="kde",
            ax = ax[1][0]
        )

        g = sns.displot(
            relevant_df.loc[(relevant_df["incident"] == 'Ladder'), "date_posted"],
            color='orange',
            fill=True,
            label="Ladder",
            ax = ax[1][1]
        )

    
        plt.tight_layout()
        plt.show()

    def tweet_distribution2(self):
        '''Baguio, Scandal, Quarantine, Ladder, and Overall'''
        relevant_df = self.df.loc[2:202]

        fig, ax = plt.subplots(1, 1, figsize=(15,5), dpi=120)

        g = sns.kdeplot(
            relevant_df.loc[(relevant_df["incident"] == 'Scandal'), "date_posted"],
            fill=True,
            label="Scandal",
        )

        plt.title("'Scandal' Incident Tweets Distribution", fontsize=18, pad=20)
        plt.show()

    def tweet_distribution3(self):
        '''Baguio, Scandal, Quarantine, Ladder, and Overall'''
        relevant_df = self.df.loc[2:202]

        fig, ax = plt.subplots(1, 1, figsize=(15,5), dpi=120)

        g = sns.kdeplot(
            relevant_df.loc[(relevant_df["incident"] == 'Quarantine'), "date_posted"],
            fill=True,
            label="Quarantine",
        )

        plt.title("'Quarantine' Incident Tweets Distribution", fontsize=18, pad=20)
        plt.show()

    def tweet_distribution4(self):
        '''Baguio, Scandal, Quarantine, Ladder, and Overall'''
        relevant_df = self.df.loc[2:202]

        fig, ax = plt.subplots(1, 1, figsize=(15,5), dpi=120)

        g = sns.kdeplot(
            relevant_df.loc[(relevant_df["incident"] == 'Ladder'), "date_posted"],
            fill=True,
            label="Ladder",
        )

        plt.title("'Ladder' Incident Tweets Distribution", fontsize=18, pad=20)
        plt.show()

    def tweet_distributions_total(self):
        '''Baguio, Scandal, Quarantine, Ladder, and Overall'''
        relevant_df = self.df.loc[2:202]

        g = sns.lineplot(
            data=relevant_df.date_posted,
        )

        plt.title("Cumulative Disinformation Tweets Over Time", fontsize=20, pad=20)
        plt.show()


    def account_creation_histograms(self):
        '''the long awaited and still waiting for'''
        df_misinfo = self.df[self.df['is_misinfo']==1]
        df_factual = self.df[self.df['is_misinfo']==0]
        # Plot distributions
        # g = sns.kdeplot(
        #     relevant_df.loc[(relevant_df["incident"] == 'Ladder'), "date_posted"],
        #     fill=True,
        #     label="Ladder",
        #     ax=ax[1][1]
        # )
        print(df_misinfo.columns)
        sns.histplot(df_misinfo['joined'].value_counts(), label='Disinfo Account', alpha=0.5)
        sns.histplot(df_factual['joined'].value_counts(), label='Factual Account', alpha=0.5)

        plt.xlabel('Months before election')
        plt.ylabel('Count')
        plt.title('Histograms of Twitter Join Date of "Disinformation" and "Factual" Accounts', fontsize=18, pad=20)
        plt.legend()
        plt.show()

    def diamond_clique(self):
        ''' Still not possible (need column incident_ref = BLOSQ, but only use four)
            Initially, connections (int) can be counted.
        '''

    def main(self):
        self.sentiment_barplots()
        #self.incidents_pie_chart()
        #self.tweet_distribution1()
        #self.tweet_distribution2()
        #self.tweet_distribution3()
        #self.tweet_distribution4()
        # self.tweet_distributions_total()
        #self.sisters_venn_diagram()
        #self.account_creation_histograms()
        #self.diamond_clique()

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

    more_graphs = AddtionalGraphs(self_df)
    more_graphs.main()