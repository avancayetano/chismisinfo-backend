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
        # # Split by undersampling
        # class_counts = y.value_counts()
        # print(class_counts)
        # target_count = 30

        # train_df = pd.DataFrame()
        # test_df = pd.DataFrame()

        # for sentiment_class in class_counts.index:
        #     class_samples = df_senti[df_senti["leni_sentiment"] == sentiment_class]
        #     target_count = int(class_samples.shape[0] * 0.6)
        #     # target_count = 30
        #     # Resample each sentiment to have equal samples
        #     resampled_class = resample(
        #         class_samples, replace=False, n_samples=target_count, random_state=42
        #     )
        #     train_df = pd.concat([train_df, resampled_class])

        #     unchosen_class = class_samples.drop(resampled_class.index)
        #     # unchosen_class_17 = resample(
        #     #     unchosen_class, replace=False, n_samples=17, random_state=42
        #     # )
        #     # test_df = pd.concat([test_df, unchosen_class_17])
        #     test_df = pd.concat([test_df, unchosen_class])

        pass

    def hypertune(self):
        # clf = RandomForestClassifier(
        #     n_estimators=1000, criterion="entropy", random_state=12345
        # )
        # # clf = SVC(random_state=12345)

        # models = {
        #     "rf": RandomForestClassifier(random_state=12345, n_jobs=-1),
        #     "svc": SVC(random_state=12345, cache_size=1500),
        #     "gbt": GradientBoostingClassifier(random_state=12345),
        #     "mlp": MLPClassifier(random_state=12345, max_iter=10000),
        #     "dt": DecisionTreeClassifier(random_state=12345),
        # }
        # # first iteration
        # params = {
        #     "rf": {
        #         "n_estimators": [100, 1000, 3000],
        #         "criterion": ["entropy", "gini", "log_loss"],
        #         "max_features": [None, "sqrt", "log2"],
        #     },
        #     "svc": {
        #         "C": [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
        #         "gamma": [1000, 10, 1, 0.1, 0.01, 0.001, 0.0001],
        #         "kernel": ["linear", "poly", "rbf"],
        #     },
        #     "gbt": {
        #         "learning_rate": [0.1, 1, 10, 0.01],
        #         "n_estimators": [100, 1000, 3000],
        #         "max_features": [None, "sqrt", "log2"],
        #     },
        #     "mlp": {
        #         "hidden_layer_sizes": [(100,), (500,), (100, 100)],
        #         "activation": ["relu", "logistic"],
        #         "solver": ["lbfgs", "sgd", "adam"],
        #     },
        #     "dt": {"criterion": ["entropy", "gini", "log_loss"]},
        # }
        # # -------------------------------------------------------
        # # second iteration
        # models = {
        #     "rf": RandomForestClassifier(random_state=12345, n_jobs=-1),
        #     "svc": SVC(random_state=12345, cache_size=1500),
        #     "mlp": MLPClassifier(random_state=12345, max_iter=10000),
        #     "dt": DecisionTreeClassifier(random_state=12345),
        # }

        # params = {
        #     "rf": {
        #         "n_estimators": [1000],
        #         "criterion": ["entropy"],
        #         "max_features": ["log2"],
        #         "class_weight": ["balanced"],
        #     },
        #     "svc": {
        #         "C": [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
        #         "gamma": [1000, 10, 1, 0.1, 0.01, 0.001, 0.0001],
        #         "kernel": ["rbf", "sigmoid"],
        #     },
        #     "mlp": {
        #         "hidden_layer_sizes": [(100,), (500,), (100, 100)],
        #         "activation": ["relu", "logistic"],
        #         "solver": ["lbfgs", "sgd", "adam"],
        #     },
        #     "dt": {
        #         "criterion": ["entropy", "gini", "log_loss"],
        #         "splitter": ["best", "random"],
        #         "max_depth": [4, None],
        #         "max_features": ["sqrt", "log2"],
        #         "class_weight": [None, "balanced"],
        #     },
        # }

        # # third iteration (apparently tree-based models are the best)
        # models = {
        #     # "rf": RandomForestClassifier(random_state=12345, n_jobs=-1),
        #     "xgb": XGBClassifier(random_state=12345, n_jobs=-1),
        #     "mlp": MLPClassifier(random_state=12345, max_iter=10000),
        #     "dt": DecisionTreeClassifier(random_state=12345),
        #     # "svc": SVC(random_state=12345, cache_size=1500),
        # }

        # params = {
        #     "rf": {
        #         "n_estimators": [100, 1000],
        #         "criterion": ["entropy"],
        #         "max_features": ["log2", "sqrt"],
        #         "class_weight": ["balanced", "balanced_subsample"],
        #     },
        #     "xgb": {
        #         "objective": ["multi:softmax"],
        #         "n_estimators": [100, 3000],
        #         "alpha": [1, 10],
        #         "max_depth": [6],
        #         "subsample": [0.8, 1.0],
        #         "learning_rate": [0.01],
        #     },
        #     "svc": {
        #         "C": [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
        #         "gamma": [1000, 10, 1, 0.1, 0.01, 0.001, 0.0001],
        #         "kernel": ["rbf", "sigmoid"],
        #     },
        #     "mlp": {
        #         "hidden_layer_sizes": [(100,), (500,), (100, 100)],
        #         "activation": ["relu", "logistic"],
        #         "solver": ["lbfgs", "sgd", "adam"],
        #     },
        #     "dt": {
        #         "criterion": ["entropy", "gini", "log_loss"],
        #         "splitter": ["best", "random"],
        #         "max_depth": [4, None],
        #         "max_features": ["sqrt", "log2"],
        #         "class_weight": [None, "balanced"],
        #     },
        # }
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

        # Split training and test with stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=1234567
        )

        self.undersample()

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
                token for token in lemmatized_tokens if token.lower() not in stop_words
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

        models = {
            # "rf": RandomForestClassifier(random_state=12345, n_jobs=-1),
            "xgb": XGBClassifier(
                random_state=12345, n_jobs=-1, importance_type="weight"
            ),
        }

        params = {
            "rf": {
                "n_estimators": [100, 1000],
                "criterion": ["entropy"],
                "max_features": ["log2", "sqrt"],
                "class_weight": ["balanced", "balanced_subsample"],
            },
            # "xgb": {
            #     "objective": ["binary:logistic"],
            #     "n_estimators": [100, 3000],
            #     "alpha": [1, 10],
            #     "max_depth": [6],
            #     "subsample": [0.8, 1.0],
            #     "learning_rate": [0.01],
            # },
            "xgb": {
                "objective": ["binary:logistic"],
                "n_estimators": [100],
                "alpha": [1],
                "max_depth": [6],
                "subsample": [0.8],
                "learning_rate": [0.01],
            },
            "svc": {
                "C": [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
                "gamma": [1000, 10, 1, 0.1, 0.01, 0.001, 0.0001],
                "kernel": ["rbf", "sigmoid"],
            },
            "mlp": {
                "hidden_layer_sizes": [(100,), (500,), (100, 100)],
                "activation": ["relu", "logistic"],
                "solver": ["lbfgs", "sgd", "adam"],
            },
            "dt": {
                "criterion": ["entropy", "gini", "log_loss"],
                "splitter": ["best", "random"],
                "max_depth": [4, None],
                "max_features": ["sqrt", "log2"],
                "class_weight": [None, "balanced"],
            },
        }

        print(f">>> BEGIN: HYPERPARAMETER TUNING ON {list(models.keys())}")
        for model in models:
            print()
            print(f">>> TUNING: {models[model].__class__.__name__}")
            clf = models[model]
            param_grid = params[model]

            grid = GridSearchCV(
                clf, param_grid, refit=True, n_jobs=-1, scoring="accuracy"
            )
            grid.fit(X_train_vec, y_train)
            print(f"best params: {grid.best_params_}")
            classifier = grid.best_estimator_

            print("Done training!")
            print("Feature importances")
            print(classifier.feature_importances_)
            print(f"SHAPE: {classifier.feature_importances_.shape}")
            feat_importances = pd.Series(
                classifier.feature_importances_, index=features
            ).sort_values(ascending=False)
            print(feat_importances.head(20))

            # feat_importances = pd.Series(
            #     classifier.feature_importances_, index=features
            # ).sort_values(ascending=False)

            # print(classifier.feature_names_)
            # print(classifier.feature_importances_)
            # Predict the sentiment labels for the test set
            y_pred = classifier.predict(X_test_vec)
            print("\nPERFORMING PERMUTATION IMPORTANCE\n")
            # r = permutation_importance(
            #     classifier,
            #     X_test_vec,
            #     y_test,
            #     n_repeats=30,
            #     random_state=12345,
            #     n_jobs=-1,
            #     scoring="f1",
            # )
            # perm_importances = pd.Series(
            #     r.importances_mean, index=features
            # ).sort_values(ascending=False)
            # print("PERMUTATION IMPORTANCES")
            # print(perm_importances.head(40))

            # Print precision, recall, accuracy, fscore on test data
            print("Model Evaluation")
            print("Training samples:", len(y_train))
            print("Test samples:", len(y_test), "\n")

            print(classification_report(y_test, y_pred))
            # print(f"Balanced accuracy score: {balanced_accuracy_score(y_test, y_pred)}")

            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(16, 8))

            sns.heatmap(cm, annot=True, cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"{model}")
        plt.show()

    def main(self):
        # self.visualize_model_data()
        self.perform_classification()
        pass
