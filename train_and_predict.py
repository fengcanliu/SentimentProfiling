from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np
import spacy.tokenizer as spacy_tokenizer
import pandas as pd
# from spacytextblob.spacytextblob import SpacyTextBlob
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from spacy.lang.en import English
import string
import pickle

#Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [text for text in X]
    def fit(self, X, y, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}


def extra_data(path):
    df = pd.read_csv(path)
    return df


def convert_label(df):
    conditions = [
        (df['sentiment'] == 'Object'),
        (df['sentiment'] == 'Support'),
        (df['sentiment'] == 'Neutral')
    ]

    # create a list of the values we want to assign for each condition
    values = [-1, 1, 0]

    # create a new column and use np.select to assign values to it using our lists as arguments
    df['sentiment'] = np.select(conditions, values)
    return df


def my_tokenizer(sentence):
    punctuations = string.punctuation
    parser = English()
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in ENGLISH_STOP_WORDS and word not in punctuations ]
    return mytokens


def train_and_predict(labeled_data):
    # Vectorization
    nlp = English()
    vectorizer = CountVectorizer(tokenizer=spacy_tokenizer.Tokenizer(nlp.vocab), ngram_range=(1, 1))
    classifier = LinearSVC(random_state=3, tol=1e-5)

    # Using Tfidf
    tfvectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer.Tokenizer(nlp.vocab))

    # Features and Labels
    X = labeled_data['Comment Question'].fillna("empty")
    ylabels = labeled_data['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=3)

    # Create the  pipeline to clean, tokenize, vectorize, and classify using"Count Vectorizor"
    print("Initialise pipeline")
    pipe_countvect = Pipeline([("cleaner", predictors()),
                               ('vectorizer', vectorizer),
                               ('classifier', classifier)])
    # Fit our data
    print("Start fitting data")
    pipe_countvect.fit(X_train, y_train)

    # save module
    print("Saving model")
    filename = 'linear_svm.sav'
    pickle.dump(pipe_countvect, open(filename, 'wb'))

    # Accuracy
    print("Accuracy: ", pipe_countvect.score(X_test, y_test))


if __name__ == '__main__':

    labeled_data = convert_label(extra_data("data/Hackathon Data - Keystone.csv"))
    train_and_predict(labeled_data)
