import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

titanic_train = 'https://storage.googleapis.com/tf-datasets/titanic/train.csv'
titanic_eval = 'https://storage.googleapis.com/tf-datasets/titanic/eval.csv'

'''dftrain = pd.read_csv() # training data
dfeval = pd.read_csv() # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')'''


def analyse_input(training_input: str = titanic_train):

    training_df = pd.read_csv(training_input)
    print(f"Training Set:")
    print(training_df.head())
    print()
    print(f"Statstics on Training Set:")
    print(training_df.describe())

    #for cols in training_df.columns:
        #training_df[cols].hist()

    return training_df


def run_ml_with_tf(output_field: str, training_input: str = titanic_train, evaluation_input: str = titanic_eval):

    training_df = pd.read_csv(training_input)
    evaluation_df = pd.read_csv(evaluation_input)

    output_training = training_df.pop(output_field)
    output_evaluation = evaluation_df.pop(output_field)

    return
