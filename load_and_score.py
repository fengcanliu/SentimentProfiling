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


def load_model_and_score(filename, data_path, test_size):

    loaded_model = pickle.load(open(filename, 'rb'))

    labeled_data = convert_label(extra_data(data_path))

    # Features and Labels
    X = labeled_data['Comment Question'].fillna("empty")
    ylabels = labeled_data['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=test_size, random_state=3)
    result = loaded_model.score(X_test, y_test)
    print(result)


if __name__ == '__main__':

    labeled_data = convert_label(extra_data("data/Hackathon Data - Keystone.csv"))
    load_model_and_score('linear_svm.sav', 'data/Hackathon Data - Keystone.csv', 0.2)