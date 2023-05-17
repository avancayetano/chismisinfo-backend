"""
Should the larger datasets be used as dictionary for all the smaller datasets?
I believe we can utilize master_misinfo.csv dataset to answer our research questions
    Idea: Re research question, "Were the disinfo accounts new?"
    Instead of comparing to the regular growth of twitter,
    compare it to the dates in master_all_raw.csv instead...
    i.e. were disinfo accounts created as much as nondisinfo accounts?

    General idea: raw is the control, misinfo is the test subject
    
Remove words in the tweets which start with @
Only 6000 1-gram and 2-gram phrases!

"""


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

tweets_vect = CountVectorizer(
    lowercase=False
)  # CS 180 features to tune: stop_words, ngram_range, max_df, min_df

df = pd.read_csv("../data/master_misinfo.csv")
tweets = df["tweet"]

tweets_list = tweets.to_list()

tweets_vect.fit(tweets_list)
tweets_dtm = tweets_vect.transform(tweets_list)

tweets_vect_df = pd.DataFrame(
    data=tweets_dtm.toarray(), columns=tweets_vect.get_feature_names_out()
)

print("One gram bag of words of tweets\n")
print(tweets_vect_df)
tweets_vect_df.shape


tweets_vect2 = CountVectorizer(lowercase=False, ngram_range=(1, 3))
tweets_dtm2 = tweets_vect2.fit_transform(tweets_list)
tweets_vect2_df = pd.DataFrame(
    data=tweets_dtm2.toarray(), columns=tweets_vect2.get_feature_names_out()
)
print("\n\nOne gram and two gram of 'tweets'\n")
print(tweets_vect2_df)
tweets_vect2_df.shape
