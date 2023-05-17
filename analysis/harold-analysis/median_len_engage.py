import pandas as pd
import re
import copy
import string
import statistics


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
    print("nlp cleaning done. Emojis, symbols, punctuations, lower are handled.")


def median_engagement(df_filtered):
    df_filtered["engagement"] = (
        df_filtered["likes"] + df_filtered["retweets"] + df_filtered["replies"]
    )

    print(df_filtered["engagement"].describe())


df_filtered = pd.read_csv("labeled_candidate_misinfo - filtered.csv")


median_engagement(df_filtered)
tweet_lengths = [len(tweet.split()) for tweet in handle_symbols_and_lower(df_filtered)]
print(tweet_lengths)
print(statistics.median(tweet_lengths))
