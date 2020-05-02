from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp  # We are using a different module from tensorflow this time

# linear regression
titanic_train = 'https://storage.googleapis.com/tf-datasets/titanic/train.csv'
titanic_eval = 'https://storage.googleapis.com/tf-datasets/titanic/eval.csv'
# classification
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
iris_train = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
iris_test = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
iris_train_file = "iris_training.csv"
iris_test_file = "iris_test.csv"


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

    '''
    history = model.fit(
        normed_train_data, train_labels, 
        epochs=EPOCHS, validation_split = 0.2, verbose=0,
        callbacks=[tfdocs.modeling.EpochDots()])
    '''

    return trained_linear_estimator


def evaluate_linear_estimator(trained_linear_estimator, output_field: str = 'survived', evaluation_file: str = titanic_eval):
    """
    Parameters
    ----------
    trained_linear_estimator
    output_field: str
    evaluation_file: str

    Returns
    -------
    results

    """

    evaluation_df = pd.read_csv(evaluation_file)  # testing data
    evaluation_output = evaluation_df.pop(output_field)
    eval_input_fn = make_input_function(
        evaluation_df, evaluation_output, num_epochs=1, shuffle=False)
    # get model metrics/stats by testing on tetsing data
    # results is of type 'dict'
    results = trained_linear_estimator.evaluate(eval_input_fn)
    # pred_dicts is a list
    pred_dicts = list(trained_linear_estimator.predict(eval_input_fn))

    probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

    probs.plot(kind='hist', bins=20, title='predicted probabilities')

    prediction_df = pd.DataFrame.from_dict(pred_dicts)
    '''

    # probs = prediction_df['probabilities'][1]

    # probs.plot(kind='hist', bins=20, title='predicted probabilities')

    for cols in prediction_df.columns:
        if prediction_df[cols].dtypes == object:
            figo = plt.figure()
            prediction_df[cols].value_counts().plot(kind='barh')
        else:
            fign = plt.figure()
            prediction_df[cols].hist()
            plt.ylabel(cols)
    '''

    return results, prediction_df


def analyse_input_classification(input_data: str = iris_train, input_file: str = iris_train_file, column_names: str = CSV_COLUMN_NAMES):

    train_path = tf.keras.utils.get_file(input_file, input_data)

    train_df = pd.read_csv(train_path, names=column_names, header=0)

    print(train_df.head())

    return train_df


def preprocess_dataset_classification(output_field: str, input_data: str = iris_train, input_file: str = iris_train_file, column_names: str = CSV_COLUMN_NAMES):

    train_df = analyse_input_classification(
        input_data, input_file, column_names)

    train_df_y = train_df.pop(output_field)

    print(train_df.shape)

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_df.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    print(my_feature_columns)

    return train_df, train_df_y, my_feature_columns


def input_classification_fn(features, labels, training: bool = True, batch_size: int = 256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)


def train_classifier(output_field: str, input_data: str = iris_train, input_file: str = iris_train_file, column_names: str = CSV_COLUMN_NAMES, hidden_units = [30, 10], steps: int = 5000):
    
    # preprocessing
    train_df, train_df_y, my_feature_columns = preprocess_dataset_classification(output_field, input_data, input_file, column_names)

    n_classes = len(train_df_y.unique())
    # Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
    classifier = tf.estimator.DNNClassifier(
        feature_columns = my_feature_columns,
        # hidden_units = [30, 10] => Two hidden layers of 30 and 10 nodes respectively.
        hidden_units = hidden_units,
        # The model must choose between 3 classes.
        n_classes = n_classes)

    # training
    classifier.train(
        input_fn=lambda: input_classification_fn(train_df, train_df_y, training=True),
        steps=steps)
    # We include a lambda to avoid creating an inner function previously
    # lambda is an anonymous fucntion which can be defne din one line

    return classifier


def evaluate_classifier(classifier, output_field: str, input_data: str = iris_test, input_file: str = iris_test_file, column_names: str = CSV_COLUMN_NAMES, hidden_units = [30, 10], steps: int = 5000):

    # preprocessing
    evaluate_df, evaluate_df_y, my_feature_columns = preprocess_dataset_classification(output_field, input_data, input_file, column_names)

    # evaluate
    evaluation_result = classifier.evaluate(
        input_fn=lambda: input_classification_fn(evaluate_df, evaluate_df_y, training=False)
        )

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**evaluation_result))

    return


def input_prediction_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


def classifier_predictions(classifier, output_field: str, column_names: list = CSV_COLUMN_NAMES):

    try:
        column_names.remove(output_field)
    except:
        column_names
    features = column_names
    predict = {}

    print(features)

    print("Please type numeric values as prompted.")
    for feature in features:
        valid = True
        while valid: 
            val = input(feature + ": ")
            if not val.isdigit(): valid = False

        predict[feature] = [float(val)]

    predictions = classifier.predict(input_fn=lambda: input_prediction_fn(predict))
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{}" ({:.1f}%)'.format(
            SPECIES[class_id], 100 * probability))

    return


def hidden_markov_model(initial_probs = [0.2, 0.8], transition_probs = [0.5, 0.5], mean_p = [0., 15.], std_p = [5., 10.], future_states_number: int = 7):

    print(f"number of states: {len(initial_probs)}")
    # normalize a list
    initial_probs = [x / sum(initial_probs) for x in initial_probs]
    transition_probs = [x / sum(transition_probs) for x in transition_probs]
    # making a shortcut for later on
    tfd = tfp.distributions  
    # Refer to point 2 above
    initial_distribution = tfd.Categorical(probs=initial_probs) 
    # refer to points 3 and 4 above 
    transition_distribution = tfd.Categorical(probs=[transition_probs,
                                                 initial_probs]) 
    # refer to point 5 above
    # the loc argument represents the mean and the scale is the standard devitation
    observation_distribution = tfd.Normal(loc=mean_p, scale=std_p)  

    # define the model
    model = tfd.HiddenMarkovModel(
        initial_distribution=initial_distribution,
        transition_distribution=transition_distribution,
        observation_distribution=observation_distribution,
        num_steps=future_states_number)

    mean = model.mean()

    # due to the way TensorFlow works on a lower level we need to evaluate part of the graph
    # from within a session to see the value of this tensor

    # in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
    with tf.compat.v1.Session() as sess:
        print(mean.numpy())

    return