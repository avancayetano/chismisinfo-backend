# pyright: basic
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
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
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from wordcloud import STOPWORDS, WordCloud
from xgboost import XGBClassifier


class SentimentAnalysis:
    def __init__(self, df):
        self.df = df
        self.df_senti = self.df
        self.df_senti = self.df_senti[["tweet_lower", "leni_sentiment"]]

    def visualize_model_data(self):
        df_senti = self.df_senti
        # df_senti = df_senti.rename(columns={'clean_text': 'tweet'})
        # df_senti = df_senti.rename(columns={'category': 'sentiment'})

        sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        sentiment_colors = ["#c0392b", "#2c3e50", "#16a085"]
        df_senti["sentiment_label"] = df_senti["leni_sentiment"]

        # Count the occurrences of each sentiment
        sentiment_counts = df_senti["sentiment_label"].value_counts()
        sentiment_counts = sentiment_counts.reindex(["negative", "neutral", "positive"])

        # Plot the distribution of sentiments
        fig = px.bar(
            sentiment_counts,
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            labels={"x": "Sentiment", "y": "Count"},
            title="Twitter Sentiments Distribution",
        )

        fig.update_traces(marker_color=sentiment_colors)
        fig.show()

        # Initialize NLP components
        stop_words = set(nltk_stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        sentiment_groups = df_senti.groupby("leni_sentiment")
        sentiment_colors = ["#c0392b", "#2c3e50", "#16a085"]
        stopwords = set(STOPWORDS)
        # stop_words = set(stopwords.words('english'))

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))

        # Generate word cloud for each sentiment
        for ax, (sentiment, group), color in zip(
            axes, sentiment_groups, sentiment_colors
        ):
            text = group["tweet_lower"].str.cat(sep=" ")

            # Preprocess text
            tokens = word_tokenize(text)
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
            filtered_tokens = [
                token for token in lemmatized_tokens if token.lower() not in stop_words
            ]

            wordcloud = WordCloud(
                background_color=color,
                color_func=lambda *args, **kwargs: "white",
                prefer_horizontal=1.0,
                collocations=False,
            ).generate(" ".join(filtered_tokens))

            ax.imshow(wordcloud, interpolation="bilinear")
            ax.set_title(f"{sentiment}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def perform_classification(self):
        df_senti = self.df_senti

        # Prepare datasets
        df_senti = df_senti.dropna()
        X = df_senti["tweet_lower"]
        df_senti["leni_sentiment"] = df_senti["leni_sentiment"].replace(
            {"negative": 0, "neutral": 1, "positive": 2}
        )
        y = df_senti["leni_sentiment"]

        # Split training and test with stratify
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Split by undersampling
        class_counts = df_senti["leni_sentiment"].value_counts()
        print(class_counts)
        target_count = 30

        train_df = pd.DataFrame()
        test_df = pd.DataFrame()

        for sentiment_class in class_counts.index:
            class_samples = df_senti[df_senti["leni_sentiment"] == sentiment_class]
            target_count = int(class_samples.shape[0] * 0.6)
            target_count = 30
            # Resample each sentiment to have equal samples
            resampled_class = resample(
                class_samples, replace=False, n_samples=target_count, random_state=42
            )
            train_df = pd.concat([train_df, resampled_class])

            unchosen_class = class_samples.drop(resampled_class.index)
            # unchosen_class_17 = resample(
            #     unchosen_class, replace=False, n_samples=17, random_state=42
            # )
            # test_df = pd.concat([test_df, unchosen_class_17])
            test_df = pd.concat([test_df, unchosen_class])

        X_train, y_train = train_df["tweet_lower"], train_df["leni_sentiment"]
        X_test, y_test = test_df["tweet_lower"], test_df["leni_sentiment"]

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
            return filtered_tokens

        # Convert text into numerical features
        # vectorizer = CountVectorizer(tokenizer=custom_tokenizer) # Bag-of-words
        vectorizer = TfidfVectorizer(
            tokenizer=custom_tokenizer, ngram_range=(1, 1), token_pattern=None
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # param_grid = {
        #     "C": [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
        #     "gamma": [1000, 10, 1, 0.1, 0.01, 0.001, 0.0001],
        #     "kernel": ["linear", "poly", "rbf"],
        # }
        # clf = Pipeline([("scaler", StandardScaler(with_mean=False)), ("svc", SVC())])

        clf = RandomForestClassifier(
            n_estimators=1000, criterion="entropy", random_state=12345
        )
        # clf = SVC(random_state=12345)

        models = {
            "rf": RandomForestClassifier(random_state=12345, n_jobs=-1),
            "svc": SVC(random_state=12345, cache_size=1500),
            "gbt": GradientBoostingClassifier(random_state=12345),
            "mlp": MLPClassifier(random_state=12345, max_iter=10000),
            "dt": DecisionTreeClassifier(random_state=12345),
        }
        # first iteration
        params = {
            "rf": {
                "n_estimators": [100, 1000, 3000],
                "criterion": ["entropy", "gini", "log_loss"],
                "max_features": [None, "sqrt", "log2"],
            },
            "svc": {
                "C": [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
                "gamma": [1000, 10, 1, 0.1, 0.01, 0.001, 0.0001],
                "kernel": ["linear", "poly", "rbf"],
            },
            "gbt": {
                "learning_rate": [0.1, 1, 10, 0.01],
                "n_estimators": [100, 1000, 3000],
                "max_features": [None, "sqrt", "log2"],
            },
            "mlp": {
                "hidden_layer_sizes": [(100,), (500,), (100, 100)],
                "activation": ["relu", "logistic"],
                "solver": ["lbfgs", "sgd", "adam"],
            },
            "dt": {"criterion": ["entropy", "gini", "log_loss"]},
        }
        # -------------------------------------------------------
        # second iteration
        models = {
            "rf": RandomForestClassifier(random_state=12345, n_jobs=-1),
            "svc": SVC(random_state=12345, cache_size=1500),
            "mlp": MLPClassifier(random_state=12345, max_iter=10000),
            "dt": DecisionTreeClassifier(random_state=12345),
        }

        params = {
            "rf": {
                "n_estimators": [1000],
                "criterion": ["entropy"],
                "max_features": ["log2"],
                "class_weight": ["balanced"],
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

        # third iteration (apparently tree-based models are the best)
        models = {
            # "rf": RandomForestClassifier(random_state=12345, n_jobs=-1),
            "xgb": XGBClassifier(random_state=12345, n_jobs=-1),
            "mlp": MLPClassifier(random_state=12345, max_iter=10000),
            "dt": DecisionTreeClassifier(random_state=12345),
            # "svc": SVC(random_state=12345, cache_size=1500),
        }

        params = {
            "rf": {
                "n_estimators": [100, 1000],
                "criterion": ["entropy"],
                "max_features": ["log2", "sqrt"],
                "class_weight": ["balanced", "balanced_subsample"],
            },
            "xgb": {
                "objective": ["multi:softmax"],
                "n_estimators": [100, 3000],
                "alpha": [1, 10],
                "max_depth": [6],
                "subsample": [0.8, 1.0],
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

        # print("+++++++++++++++++++++++++++++++++++++++++++++++")
        # svd = TruncatedSVD(n_components=100)
        # import numpy as np

        # X_train_dtm = np.asarray(X_train_vec.todense())
        # X_train_vec = svd.fit_transform(X_train_dtm)
        # X_test_dtm = np.asarray(X_test_vec.todense())
        # X_test_vec = svd.transform(X_test_dtm)

        print(">>> DONE DIMENSIONALITY REDUCTION")

        print(f">>> BEGIN: HYPERPARAMETER TUNING ON {models.keys()}")
        for model in models:
            print()
            print(f">>> TUNING: {models[model].__class__.__name__}")
            clf = models[model]
            param_grid = params[model]

            grid = GridSearchCV(
                clf, param_grid, refit=True, n_jobs=-1, scoring="f1_macro"
            )
            grid.fit(X_train_vec, y_train)
            print(f"best params: {grid.best_params_}")
            classifier = grid.best_estimator_
            # classifier.fit(X_train_vec, y_train)

            print("Done training!")
            # # Create a pipeline with StandardScaler and SVR
            # pipeline = Pipeline([('scaler', StandardScaler()),
            #                      ('svc', SVC(kernel='rbf'))])

            # # Define the parameter grid for hyperparameter optimization
            # param_grid = {'svc__C': [10000, 1000000, 100],
            #               'svc__gamma': [10, 1000, 100]}

            # # Perform grid search with cross-validation on train data
            # grid_search = GridSearchCV(pipeline, param_grid)
            # grid_search.fit(x_train, y_train)

            # # Predict using best model on test data
            # best_svr = grid_search.best_estimator_
            # y_svr_pred_test = best_svr.predict(x_test)

            # Predict the sentiment labels for the test set
            y_pred = classifier.predict(X_test_vec)

            # Print precision, recall, accuracy, fscore on test data
            print("Model Evaluation")
            print("Training samples:", len(y_train))
            print("Test samples:", len(y_test), "\n")

            labels = ["negative", "neutral", "positive"]
            print("------------------------------------")

            print(classification_report(y_test, y_pred))
            print(f"Balanced accuracy score: {balanced_accuracy_score(y_test, y_pred)}")

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
