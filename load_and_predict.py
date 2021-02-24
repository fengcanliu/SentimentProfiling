import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


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


def extra_data(path):
    df = pd.read_csv(path)
    return df


def load_model_and_predict(filename, test_text):

    loaded_model = pickle.load(open(filename, 'rb'))

    result = loaded_model.predict(test_text)
    print(result)


if __name__ == '__main__':

    labeled_data = convert_label(extra_data("data/Hackathon Data - Keystone.csv"))
    load_model_and_predict('linear_svm.sav', 'I am not happy')