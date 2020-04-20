import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

titanic_train = 'https://storage.googleapis.com/tf-datasets/titanic/train.csv'
titanic_eval = 'https://storage.googleapis.com/tf-datasets/titanic/eval.csv'


def analyse_input(training_input: str = titanic_train):

    training_df = pd.read_csv(training_input)
    print(f"Training Set:")
    print(training_df.head())
    print()
    print(f"Statstics on Training Set:")
    print(training_df.describe())

    for cols in training_df.columns:
        if training_df[cols].dtypes == object:
            figo = plt.figure()
            training_df[cols].value_counts().plot(kind='barh')
        else:
            fign = plt.figure()
            training_df[cols].hist()
            plt.ylabel(cols)

    return training_df


def preprocess_dataset(output_field: str, training_input: str = titanic_train):

    training_df = pd.read_csv(training_input)

    output_training = training_df.pop(output_field)

    CATEGORICAL_COLUMNS = []
    NUMERIC_COLUMNS = []
    for cols in training_df.columns:
        if training_df[cols].dtypes == object:
            CATEGORICAL_COLUMNS.append(cols)
        else:
            NUMERIC_COLUMNS.append(cols)

    # Creating a list of features used in the dataset.
    # tf.feature_column. create an object that the model can
    # use to map string values to integers, avoiding to manually
    # having to encode the dataframes.

    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
        # gets a list of all unique values from given feature column
        vocabulary = training_df[feature_name].unique()
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
            feature_name, vocabulary))

    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(
            feature_name, dtype=tf.float32))

    print(feature_columns)

    return training_df, output_training, feature_columns
