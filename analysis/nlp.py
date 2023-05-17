# pyright: basic

import pandas as pd
from typing import List

# import nltk
import re
import string
import copy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# from google_trans_new import google_translator as Translator
from googletrans import Translator

from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List, Dict

# Preprequisite data:
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


class NLP:
    """
    Natural Language Processing
    - [X] Tokenization and lower casing
        - [X] handle missing emojis
    - [X] Stop words removal
    - [X] Stemming and lemmatization
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feats: Dict[str, List[str]],
    ):
        self.df = df
        self.feats = feats
        self.texts_raw = self.df["tweet_rendered"]
        self.texts_lower = []
        self.texts_en = []
        self.texts_tok = []
        self.processed_texts = []

    def handle_symbols_and_lower(self):
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

        texts = copy.deepcopy(list(self.texts_raw))

        texts = [emoji_to_word(t) for t in texts]
        texts = [emote_to_word(t) for t in texts]

        # convert to lowercase
        texts = [t.lower() for t in texts]

        # remove punctuation
        texts = [t.translate(str.maketrans("", "", string.punctuation)) for t in texts]

        self.texts_lower = texts
        print("nlp cleaning done. Emojis, symbols, punctuations, lower are handled.")

    def translate(self):
        # Removing stopwords might be tedious for multilingual texts

        # CHEAP SOLUTION: translate texts to English (this is not 100% accurate)
        # from googletrans import Translator

        # translate to English
        translator = Translator()
        self.texts_en = [
            ""
            if len(text_lower) == 0
            else translator.translate(text_lower, src="tl", dest="en").text  # type: ignore
            for text_lower in self.texts_lower
        ]

        print("nlp translate done")

    def tokenize(self):
        for text in self.texts_en:
            # tokenize the text into words
            words = word_tokenize(text)

            # remove stopwords
            filtered_words = [
                word for word in words if word.lower() not in stopwords.words("english")
            ]

            # convert back into sentence
            filtered_sentence = " ".join(filtered_words)
            self.texts_tok.append(filtered_sentence)
        print("nlp tokenize done")

    def stem_lemmatize(self):
        # Initialize the stemmer and lemmatizer
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        texts_stem, texts_lem = [], []

        def stem_lem(text):
            words = text.split()

            # Stem each word
            stemmed_words = [stemmer.stem(word) for word in words]

            # Lemmatize each word
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

            # Return the stemmed and lemmatized words as a tuple
            texts_stem.append(stemmed_words)
            texts_lem.append(lemmatized_words)

            return (stemmed_words, lemmatized_words)

        # Process each text in the array
        self.processed_texts = [stem_lem(t) for t in self.texts_tok]
        print("nlp stem_lem done")

    def create_df(self) -> pd.DataFrame:
        self.df["tweet_lower"] = self.texts_lower
        self.df["tweet_gtranslated"] = self.texts_en
        self.df["tweet_stemmed"] = [
            stem_lem_tuple[0] for stem_lem_tuple in self.processed_texts
        ]
        self.df["tweet_lemmatized"] = [
            stem_lem_tuple[1] for stem_lem_tuple in self.processed_texts
        ]

        print(
            self.df[
                [
                    "tweet_stemmed",
                    "tweet_lemmatized",
                ]
            ]
        )
        self.feats["str"].extend(
            ["tweet_lower", "tweet_gtranslated", "tweet_stemmed", "tweet_lemmatized"]
        )

        return self.df

    def main(self) -> pd.DataFrame:
        self.handle_symbols_and_lower()
        self.translate()
        # self.texts_en = self.texts_lower
        self.tokenize()
        self.stem_lemmatize()
        return self.create_df()
