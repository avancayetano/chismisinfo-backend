import pandas as pd
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


df_misinfo = pd.read_csv("labeled_candidate_misinfo - filtered.csv")
df_misinfo["tweet"] = df_misinfo["tweet"].str.lower()


# all_names = pd.read_csv("all_names_filtered.csv")
# list_of_names = set(all_names["Phrase"].str.lower())

names_vect = CountVectorizer(lowercase=True, ngram_range=(1, 3))
names_dtm = names_vect.fit_transform(list(df_misinfo["tweet"]))
names_df = pd.DataFrame(
    data=names_dtm.toarray(), columns=names_vect.get_feature_names_out()
)

names_df["tweet_url"] = df_misinfo["tweet_url"]
names_df["tweet"] = df_misinfo["tweet"]
names_df = names_df[["tweet_url", "tweet"] + list(names_df)[:-2]]

names_df.to_csv("vectorized_labeled_misinfo.csv")
