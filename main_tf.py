import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

titanic_train = 'https://storage.googleapis.com/tf-datasets/titanic/train.csv'
titanic_eval = 'https://storage.googleapis.com/tf-datasets/titanic/eval.csv'


def analyse_input(input_file: str = titanic_train):
    """
    Parameters
    ----------
    input_file: str

    """

    input_df = pd.read_csv(input_file)
    print(f"Training Set:")
    print(input_df.head())
    print()
    print(f"Statstics on Training Set:")
    print(input_df.describe())

    for cols in input_df.columns:
        if input_df[cols].dtypes == object:
            figo = plt.figure()
            input_df[cols].value_counts().plot(kind='barh')
        else:
            fign = plt.figure()
            input_df[cols].hist()
            plt.ylabel(cols)

    return input_df


def preprocess_dataset(output_field: str, input_file: str = titanic_train):
    """
    Parameters
    ----------
    output_field: str
    input_file: str


    Returns
    -------
    input_df
    output_training
    feature_columns
    """

    input_df = pd.read_csv(input_file)

    output_training = input_df.pop(output_field)

    CATEGORICAL_COLUMNS = []
    NUMERIC_COLUMNS = []
    for cols in input_df.columns:
        if input_df[cols].dtypes == object:
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
        vocabulary = input_df[feature_name].unique()
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
            feature_name, vocabulary))

    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(
            feature_name, dtype=tf.float32))

    print(feature_columns)

    return input_df, output_training, feature_columns


def make_input_function(data_df: pd.core.frame.DataFrame, label_df: pd.core.series.Series, num_epochs: int = 10, shuffle: bool = True, batch_size: int = 32):
    """
    Parameters
    ----------
    data_df: pd.core.frame.DataFrame
    label_df: pd.core.series.Series
    num_epochs: int
    shuffle: bool
    batch_size: int

    Returns
    -------
    input_function: function
        returns a function object for use.

    """

    def input_function():  # inner function, this will be returned
        # create tf.data.Dataset object with data and its label
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        # split dataset into batches of 32 and repeat process for number of epochs
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds  # return a batch of the dataset

    return input_function  # return a function object for use


def train_linear_estimator(output_field: str = 'survived', training_file: str = titanic_train, num_epochs: int = 10, shuffle: bool = True, batch_size: int = 32):
    """
    Parameters
    ----------
    output_field: str
    training_file: str
    num_epochs: int 
    shuffle: bool
    batch_size: int

    Returns
    -------
    trained_linear_estimator

    """

    # preprocess the dataset
    training_df, output_training, feature_columns = preprocess_dataset(
        output_field, training_file)

    # make the training function
    training_function = make_input_function(
        training_df, output_training, num_epochs, shuffle, batch_size)

    # linear estimator
    linear_estimator = tf.estimator.LinearClassifier(
        feature_columns=feature_columns)

    trained_linear_estimator = linear_estimator.train(training_function)

    return trained_linear_estimator
