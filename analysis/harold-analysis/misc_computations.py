# pyright: basic

"""
A function of visualizer.py
processes:
[ ] outdated getting all names (needs to be consistently outdated)
[ ] tweet median length
[ ] account_with_most_tweets
[ ] names_vectorizer.py
[ ] count_references.py


manual information via google sheet filtering:

count negative leni         50/203
count positive marcos       16/203

count the sentiments for chi square
negative leni and has reference         47
negative leni and no reference          1
nonnegative leni and has reference      50
nonnegative leni and no reference       105
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import copy
import string
import statistics

entities = {
    "Aika Robredo": ["aika", "aika diri", "aika robredo", "aika rob", "she admitted"],
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


def handle_symbols_and_lower(df: pd.DataFrame):
    df["new"] = df["tweet"].str.replace(r"(^@[^ ]+ (@[^ ]+ )*)", "", regex=True)

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

    texts = copy.deepcopy(list(df["new"]))

    texts = [emoji_to_word(t) for t in texts]
    texts = [emote_to_word(t) for t in texts]

    # convert to lowercase
    texts = [t.lower() for t in texts]

    # remove punctuation
    texts = [t.translate(str.maketrans("", "", string.punctuation)) for t in texts]

    return texts


def median_engagement(df_filtered):
    df_filtered["engagement"] = (
        df_filtered["likes"] + df_filtered["retweets"] + df_filtered["replies"]
    )
    print("Median engagement is the 50%:")
    print(df_filtered["engagement"].describe(), "\n")


def misc_data():
    # create misinfo df
    df_misinfo_labeled = pd.read_csv(
        "../../data/analysis-data/misinfo_tweets_labeled.csv",
    )
    date_feats = ["joined", "date_posted"]
    df_univ_tweets = pd.read_csv(
        "../../data/analysis-data/universal_tweets_unique.csv",
        parse_dates=date_feats,
    )
    df = df_misinfo_labeled.merge(
        df_univ_tweets,
        on=["tweet_url", "tweet_id"],
        how="left",
        validate="one_to_one",
    )

    # [all_names DataFrame] get all names (maintain the outdated-ness)
    print("-----getting all_names (outdated)----")
    tweets_vect = CountVectorizer(lowercase=False, ngram_range=(1, 3))
    outdated_df = pd.read_csv("../../data/master_misinfo.csv")
    outdated_df["tweet"] = outdated_df["tweet"].str.replace(
        r"(^@[^ ]+ (@[^ ]+ )*)", "", regex=True
    )
    tweets_dtm = tweets_vect.fit_transform(df["tweet"].to_list())
    all_names = pd.DataFrame()
    all_names["is_name"] = tweets_vect.get_feature_names_out()
    print(f"all_names.shape: {all_names.shape}", "\n")

    # [aliases_df DataFrame] make dataframe for 1-3 gram tokens
    print("-----getting count of every token----")
    df["tweet"] = df["tweet"].str.lower()
    tokens_vect = CountVectorizer(lowercase=True, ngram_range=(1, 3))
    tokens_dtm = tokens_vect.fit_transform(list(df["tweet"]))
    tokens_df = pd.DataFrame(
        data=tokens_dtm.toarray(), columns=tokens_vect.get_feature_names_out()
    )
    print(f"tokens_df.shape: {tokens_df.shape}", "\n")

    # count references!
    # possible after manual filtering of all_names
    print("-----getting count of every alias----")
    for entity in entities:
        total_refs = 0
        counts = []

        for phrase in entities[entity]:
            count = tokens_df[phrase].sum()
            counts.append(count)
        total_refs = sum(counts)
        print(f"Total ({entity}) = {total_refs}")
        print(entities[entity])
        print(counts)
    print("\n")

    # counting statistics
    print("-----counting a few more statistics-----")
    median_engagement(df)
    tweet_lengths = [len(tweet.split()) for tweet in handle_symbols_and_lower(df)]
    print(f"tweet_lengths={tweet_lengths}")
    print(f"Tweet median length: {statistics.median(tweet_lengths)} words", "\n")
    print("Account with most tweets:")
    print(df.account_handle.value_counts())


if __name__ == "__main__":
    misc_data()
