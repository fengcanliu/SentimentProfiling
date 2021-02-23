# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import spacy
import os
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter
import numpy as np
from spacy.attrs import ENT_IOB, ENT_TYPE
import spacy.tokenizer as spacy_tokenizer
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from spacy.lang.en import English
import string

#Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [text for text in X]
    def fit(self, X, y, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    return text.strip().lower()

def get_sentiment(text):
    nlp = spacy.load("en_core_web_sm")
    spacy_text_blob = SpacyTextBlob()
    nlp.add_pipe(spacy_text_blob)
    print(text[4])
    doc = nlp(text[4])
    print(doc._.sentiment.polarity)
    print(doc._.sentiment.assessments)


def preprocessing(path):
    text = open(path).read()
    txt = " ".join(
        [i for i in re.sub(r'[^a-z0-9A-Z\s]', "", text).split() if i not in ENGLISH_STOP_WORDS])
    return txt

def extraData(path):
    df = pd.read_csv(path)
    return df

# WIP
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

    # display updated DataFrame
    print(df['sentiment'])


    return df

def my_tokenizer(sentence):
    punctuations = string.punctuation
    parser = English()
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    return mytokens


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = extraData("data/Hackathon Data - Keystone.csv")
    # print(data)
    # get_sentiment(data)
    labeled_data = convert_label(data)

    # Vectorization
    nlp = English()
    vectorizer = CountVectorizer(tokenizer=spacy_tokenizer.Tokenizer(nlp.vocab), ngram_range=(1, 1))
    classifier = LinearSVC()

    # Using Tfidf
    tfvectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer.Tokenizer(nlp.vocab))

    # Features and Labels
    X = labeled_data['Comment Question'].fillna("empty")
    ylabels = labeled_data['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)

    # Create the  pipeline to clean, tokenize, vectorize, and classify using"Count Vectorizor"
    pipe_countvect = Pipeline([("cleaner", predictors()),
                               ('vectorizer', vectorizer),
                               ('classifier', classifier)])
    # Fit our data
    pipe_countvect.fit(X_train, y_train)
    # Predicting with a test dataset
    sample_prediction = pipe_countvect.predict(X_test)

    for (sample, pred) in zip(X_test, sample_prediction):
        print(sample, "Prediction=>", pred)

    # Accuracy
    print("Accuracy: ", pipe_countvect.score(X_test, y_test))