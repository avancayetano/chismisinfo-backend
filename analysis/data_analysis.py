import nltk
import pandas as pd
from apply_chi_square import ApplyChiSquare
from data_explore import DataExplore
from disinfo_classifier import DisinfoClassifier
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from poster_graphs import AddtionalGraphs
from sentiment_analysis import SentimentAnalysis
from wordcloud import STOPWORDS, WordCloud


class DataAnalysis(DataExplore):
    """
    [X] Get df from DataExplore
    [ ] Statistical Treatment
    [ ] Visualize data for modeling
    [ ] Perform classification and test
    """

    def __init__(self):
        super().__init__()
        super().main()
        print("--------BEGIN: ANALYSIS--------")
        print(self.df.columns)
        # self.df = self.df
        # self.df_senti = self.df[["tweet_lower", "leni_sentiment"]]

    # def visualize_model_data(self):
    #     df_senti = self.df_senti
    #     # df_senti = df_senti.rename(columns={'clean_text': 'tweet'})
    #     # df_senti = df_senti.rename(columns={'category': 'sentiment'})

    #     import matplotlib.pyplot as plt
    #     import plotly.express as px

    #     sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    #     sentiment_colors = ['#c0392b', '#2c3e50', '#16a085']
    #     df_senti['sentiment_label'] = df_senti['leni_sentiment']

    #     # Count the occurrences of each sentiment
    #     sentiment_counts = df_senti['sentiment_label'].value_counts()
    #     sentiment_counts = sentiment_counts.reindex(['negative', 'neutral', 'positive'])

    #     # Plot the distribution of sentiments
    #     fig = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values,
    #                 labels={'x': 'Sentiment', 'y': 'Count'},
    #                 title='Twitter Sentiments Distribution')

    #     fig.update_traces(marker_color=sentiment_colors)
    #     fig.show()

    #     # Initialize NLP components
    #     stop_words = set(nltk_stopwords.words('english'))
    #     lemmatizer = WordNetLemmatizer()

    #     sentiment_groups = df_senti.groupby('leni_sentiment')
    #     sentiment_colors = ['#c0392b', '#2c3e50', '#16a085']
    #     stopwords = set(STOPWORDS)
    #     # stop_words = set(stopwords.words('english'))

    #     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))

    #     # Generate word cloud for each sentiment
    #     for ax, (sentiment, group), color in zip(axes, sentiment_groups, sentiment_colors):
    #         text = group['tweet_lower'].str.cat(sep=' ')

    #         # Preprocess text
    #         tokens = word_tokenize(text)
    #         lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    #         filtered_tokens = [token for token in lemmatized_tokens if token.lower() not in stop_words]

    #         wordcloud = WordCloud(background_color=color,
    #                         color_func=lambda *args, **kwargs: 'white',
    #                         prefer_horizontal=1.0,
    #                         collocations=False).generate(' '.join(filtered_tokens))

    #         ax.imshow(wordcloud, interpolation='bilinear')
    #         ax.set_title(f'{sentiment}')
    #         ax.axis('off')

    #     plt.tight_layout()
    #     plt.show()

    # def perform_classification(self):
    #     df_senti = self.df_senti
    #     from sklearn.utils import resample
    #     from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    #     from sklearn.model_selection import train_test_split
    #     from sklearn.naive_bayes import MultinomialNB
    #     from sklearn.metrics import classification_report, confusion_matrix
    #     import seaborn as sns
    #     import matplotlib.pyplot as plt

    #     # Prepare datasets
    #     df_senti = df_senti.dropna()
    #     X = df_senti['tweet_lower']
    #     y = df_senti['leni_sentiment']

    #     # Split training and test with stratify
    #     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    #     # Split by undersampling
    #     class_counts = df_senti['leni_sentiment'].value_counts()
    #     target_count = int(class_counts.min() * 0.9)

    #     train_df = pd.DataFrame()
    #     test_df = pd.DataFrame()

    #     for sentiment_class in class_counts.index:
    #         class_samples = df_senti[df_senti['leni_sentiment'] == sentiment_class]

    #         # Resample each sentiment to have equal samples
    #         resampled_class = resample(class_samples, replace=False, n_samples=target_count, random_state=42)
    #         train_df = pd.concat([train_df, resampled_class])

    #         unchosen_class = class_samples.drop(resampled_class.index)
    #         test_df = pd.concat([test_df, unchosen_class])

    #     X_train, y_train = train_df['tweet_lower'], train_df['leni_sentiment']
    #     X_test, y_test = test_df['tweet_lower'], test_df['leni_sentiment']

    #     # Define NLP preprocessing steps
    #     def custom_tokenizer(text):
    #         # Correct typos
    #         # text = str(TextBlob(text).correct()) # This may take a lot of time
    #         # translate the whole tweet (keep prominent words, e.g. "lugaw", "lutang")
    #         tokens = word_tokenize(text)
    #         # create a custom dictionary for lemmatization
    #         # fil_lemma = {"lutang": "lumulutang-lutang"}
    #         stop_words = set(nltk_stopwords.words('english'))
    #         lemmatizer = WordNetLemmatizer()
    #         lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    #         filtered_tokens = [token for token in lemmatized_tokens if token.lower() not in stop_words]
    #         return tokens

    #     # Convert text into numerical features
    #     # vectorizer = CountVectorizer(tokenizer=custom_tokenizer) # Bag-of-words
    #     vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, ngram_range=(1,1), token_pattern=None)
    #     X_train_vec = vectorizer.fit_transform(X_train)
    #     X_test_vec = vectorizer.transform(X_test)

    #     # Train a Naive Bayes classifier
    #     classifier = MultinomialNB() #SVC(gamma='auto'))
    #     classifier.fit(X_train_vec, y_train)

    #     # # Create a pipeline with StandardScaler and SVR
    #     # pipeline = Pipeline([('scaler', StandardScaler()),
    #     #                      ('svc', SVC(kernel='rbf'))])

    #     # # Define the parameter grid for hyperparameter optimization
    #     # param_grid = {'svc__C': [10000, 1000000, 100],
    #     #               'svc__gamma': [10, 1000, 100]}

    #     # # Perform grid search with cross-validation on train data
    #     # grid_search = GridSearchCV(pipeline, param_grid)
    #     # grid_search.fit(x_train, y_train)

    #     # # Predict using best model on test data
    #     # best_svr = grid_search.best_estimator_
    #     # y_svr_pred_test = best_svr.predict(x_test)

    #     # Predict the sentiment labels for the test set
    #     y_pred = classifier.predict(X_test_vec)

    #     # Print precision, recall, accuracy, fscore on test data
    #     print('Model Evaluation')
    #     print('Training samples:', len(y_train))
    #     print('Test samples:', len(y_test), '\n')
    #     print(classification_report(y_test, y_pred))

    def main(self):
        # do_chi_square = ApplyChiSquare(self.df)
        # do_chi_square.main()
        disinfo_classifier = DisinfoClassifier(self.df)
        disinfo_classifier.main()
        # sentiment_analyzer = SentimentAnalysis(self.df)
        # sentiment_analyzer.main()
        # more_graphs = AddtionalGraphs(self.df)
        # more_graphs.main()


if __name__ == "__main__":
    data_analysis = DataAnalysis()
    data_analysis.main()
