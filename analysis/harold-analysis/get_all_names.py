"""
Output to a csv file all tokens.
Manually choose all tokens referring to any entities made of people
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

tweets_vect = CountVectorizer(lowercase=False, ngram_range=(1, 3))
df = pd.read_csv("../data/master_misinfo_raph_harold.csv")
df["tweet"] = df["tweet"].str.replace(r"(^@[^ ]+ (@[^ ]+ )*)", "", regex=True)
tweets_dtm = tweets_vect.fit_transform(df["tweet"].to_list())
all_names = pd.DataFrame()
all_names["is_name"] = tweets_vect.get_feature_names_out()

print(all_names)
all_names.to_csv("all_names.csv")
