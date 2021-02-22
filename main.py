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
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob


def get_label(text, label):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for e in doc.ents:
        if e.label_ == label:
            print("find label " + label + " in text " + e.text)
        else:
            print("find label " + e.label_ + " in text " + e.text)
    labels = [x.label_ for x in doc.ents]
    print(Counter(labels))
    items = [x.text for x in doc.ents]
    print(Counter(items).most_common(3))


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = extraData("data/Hackathon Data - Keystone.csv")
    # print(data)
    # get_sentiment(data)
    convert_label(data)
