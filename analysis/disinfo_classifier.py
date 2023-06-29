# pyright: basic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import scipy as sp
import seaborn as sns
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from wordcloud import STOPWORDS, WordCloud
from xgboost import XGBClassifier, plot_importance


class DisinfoClassifier:
    def __init__(self, df):
        self.df = df
        self.df_relevant = self.df[
            ["tweet_lower", "leni_sentiment", "marcos_sentiment", "is_misinfo"]
        ][self.df["tweet_lower"] != ""]
        # target variable is is_misinfo

    def undersample(self):
        pass

    def hypertune(self):
        pass

    def perform_classification(self):
        # Prepare datasets

        # for XGBoost
        self.df_relevant["leni_sentiment"] = self.df_relevant["leni_sentiment"].replace(
            {"negative": 0, "neutral": 1, "positive": 2}
        )
        self.df_relevant["marcos_sentiment"] = self.df_relevant[
            "marcos_sentiment"
        ].replace({"negative": 0, "neutral": 1, "positive": 2})

        # features
        X = self.df_relevant[["tweet_lower", "leni_sentiment", "marcos_sentiment"]]

        # target
        y = self.df_relevant["is_misinfo"]

        skf = StratifiedKFold(n_splits=5)

        F = {}
        CM = np.zeros((2, 2))
        PREC = np.array([0.0, 0.0])
        RECALL = np.array([0.0, 0.0])
        F1_SCORE = np.array([0.0, 0.0])
        SUPPORT = np.array([0, 0])
        ACC = 0
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"\n --------------- ROUND: {i} -------------------\n")

            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]

            X_test = X.iloc[test_index]
            y_test = y.iloc[test_index]

            X_tweet_train, X_leni_train, X_marcos_train, y_train = (
                X_train["tweet_lower"],
                X_train["leni_sentiment"],
                X_train["marcos_sentiment"],
                y_train,
            )
            X_tweet_test, X_leni_test, X_marcos_test, y_test = (
                X_test["tweet_lower"],
                X_test["leni_sentiment"],
                X_test["marcos_sentiment"],
                y_test,
            )

            # Define NLP preprocessing steps
            def custom_tokenizer(text):
                # Correct typos
                # text = str(TextBlob(text).correct()) # This may take a lot of time
                # translate the whole tweet (keep prominent words, e.g. "lugaw", "lutang")
                tokens = word_tokenize(text)
                # create a custom dictionary for lemmatization
                # fil_lemma = {"lutang": "lumulutang-lutang"}
                stop_words = set(nltk_stopwords.words("english"))
                lemmatizer = WordNetLemmatizer()
                lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
                filtered_tokens = [
                    token
                    for token in lemmatized_tokens
                    if token.lower() not in stop_words
                ]
                filtered_tokens = [
                    token for token in tokens if token.lower() not in stop_words
                ]

                return filtered_tokens

            # Convert text into numerical features
            # vectorizer = CountVectorizer(tokenizer=custom_tokenizer) # Bag-of-words
            vectorizer = TfidfVectorizer(
                tokenizer=custom_tokenizer, ngram_range=(1, 1), token_pattern=None
            )

            X_tweet_train_vec = vectorizer.fit_transform(X_tweet_train)
            X_tweet_test_vec = vectorizer.transform(X_tweet_test)

            # print("+++++++++++++++++++++++++++++++++++++++++++++++")
            # svd = TruncatedSVD(n_components=100)

            X_train_dtm = np.asarray(X_tweet_train_vec.todense())
            # X_train_vec = np.c_[
            #     svd.fit_transform(X_train_dtm), X_leni_train, X_marcos_train
            # ]
            X_train_vec = np.c_[X_train_dtm, X_leni_train, X_marcos_train]
            X_test_dtm = np.asarray(X_tweet_test_vec.todense())
            # X_test_vec = np.c_[svd.transform(X_test_dtm), X_leni_test, X_marcos_test]
            X_test_vec = np.c_[X_test_dtm, X_leni_test, X_marcos_test]

            print(">>> DONE DIMENSIONALITY REDUCTION")
            tokens = vectorizer.get_feature_names_out()

            features = np.append(tokens, ["leni_sentiment", "marcos_sentiment"])
            print(features)
            print(f"SHAPE: {features.shape}")

            # models = {
            #     # "rf": RandomForestClassifier(random_state=12345, n_jobs=-1),
            #     "xgb": XGBClassifier(
            #         random_state=12345, n_jobs=-1, importance_type="weight"
            #     ),
            # }

            # params = {
            #     "rf": {
            #         "n_estimators": [3000],
            #         "criterion": ["entropy"],
            #         "max_features": ["log2"],
            #     },
            #     "xgb": {
            #         "objective": ["binary:logistic"],
            #         "n_estimators": [6000],
            #         "alpha": [3],
            #         "max_depth": [15],
            #         "subsample": [0.8],
            #         "learning_rate": [0.005],
            #     },
            # }

            xgb = XGBClassifier(
                random_state=12345,
                importance_type="gain",
                objective="binary:logistic",
                n_estimators=6000,
                alpha=3,
                max_depth=15,
                colsample_bytree=0.8,
                learning_rate=0.005,
            )

            print(f">>> BEGIN: HYPERPARAMETER TUNING ON {xgb.__class__.__name__}")

            xgb.fit(X_train_vec, y_train)

            print("Done training!")
            print("Feature importances")
            print(xgb.feature_importances_)

            for idx, f in enumerate(features):
                if f not in F:
                    F[f] = xgb.feature_importances_[idx] / 5
                else:
                    F[f] += xgb.feature_importances_[idx] / 5

            # print(f"SHAPE: {classifier.feature_importances_.shape}")
            # feat_importances = pd.Series(
            #     classifier.feature_importances_, index=features
            # ).sort_values(ascending=False)
            # print(feat_importances.head(20))

            # feat_importances = pd.Series(
            #     classifier.feature_importances_, index=features
            # ).sort_values(ascending=False)

            # print(classifier.feature_names_)
            # print(classifier.feature_importances_)
            # Predict the sentiment labels for the test set
            y_pred = xgb.predict(X_test_vec)

            # Print precision, recall, accuracy, fscore on test data
            print("Model Evaluation")
            print("Training samples:", len(y_train))
            print("Test samples:", len(y_test), "\n")

            print(classification_report(y_test, y_pred))

            prec, recall, f1_score, support = precision_recall_fscore_support(
                y_test,
                y_pred,
            )
            acc = accuracy_score(y_test, y_pred)

            PREC += prec
            RECALL += recall
            F1_SCORE += f1_score
            SUPPORT += support
            ACC += acc

            CM += confusion_matrix(y_test, y_pred)

        print("\nAVERAGE RESULTS OF 5-FOLD CV\n")
        PREC = PREC / 5
        RECALL = RECALL / 5
        F1_SCORE = F1_SCORE / 5
        SUPPORT = SUPPORT / 5
        ACC = ACC / 5

        sns.set(font_scale=2)

        print(f"PRECISION: {PREC}")
        print(f"RECALL: {RECALL}")
        print(f"F1_SCORE: {F1_SCORE}")
        print(f"SUPPORT: {SUPPORT}")
        print(f"ACCURACY: {ACC}")

        importance = pd.Series(F).sort_values(ascending=False)
        print(importance.head(20))

        plt.figure(figsize=(16, 8))
        CM = CM / 5  # get average
        print("CM")
        print(CM)
        sns.heatmap(
            CM,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            xticklabels=["Non-disinfo", "Disinfo"],
            yticklabels=["Non-disinfo", "Disinfo"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for the Disinfo Classifier")
        plt.show()

    def main(self):
        # self.visualize_model_data()
        self.perform_classification()
        pass
