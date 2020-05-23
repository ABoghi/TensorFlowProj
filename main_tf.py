from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
# We are using a different module from tensorflow this time
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow_datasets as tfds

from keras.datasets import imdb
from keras.preprocessing import sequence
import keras

import gym   # all you have to do to import and use open ai gym!
import time

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

shakespeare = tf.keras.utils.get_file(
    'shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# from google.colab import files
# path_to_file = list(files.upload().keys())[0]


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


def train_classifier(output_field: str, input_data: str = iris_train, input_file: str = iris_train_file, column_names: str = CSV_COLUMN_NAMES, hidden_units=[30, 10], steps: int = 5000):

    # preprocessing
    train_df, train_df_y, my_feature_columns = preprocess_dataset_classification(
        output_field, input_data, input_file, column_names)

    n_classes = len(train_df_y.unique())
    # Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # hidden_units = [30, 10] => Two hidden layers of 30 and 10 nodes respectively.
        hidden_units=hidden_units,
        # The model must choose between 3 classes.
        n_classes=n_classes)

    # training
    classifier.train(
        input_fn=lambda: input_classification_fn(
            train_df, train_df_y, training=True),
        steps=steps)
    # We include a lambda to avoid creating an inner function previously
    # lambda is an anonymous fucntion which can be defne din one line

    return classifier


def evaluate_classifier(classifier, output_field: str, input_data: str = iris_test, input_file: str = iris_test_file, column_names: str = CSV_COLUMN_NAMES, hidden_units=[30, 10], steps: int = 5000):

    # preprocessing
    evaluate_df, evaluate_df_y, my_feature_columns = preprocess_dataset_classification(
        output_field, input_data, input_file, column_names)

    # evaluate
    evaluation_result = classifier.evaluate(
        input_fn=lambda: input_classification_fn(
            evaluate_df, evaluate_df_y, training=False)
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
            if not val.isdigit():
                valid = False

        predict[feature] = [float(val)]

    predictions = classifier.predict(
        input_fn=lambda: input_prediction_fn(predict))
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{}" ({:.1f}%)'.format(
            SPECIES[class_id], 100 * probability))

    return


def hidden_markov_model(initial_probs=[0.2, 0.8], transition_probs=[0.5, 0.5], mean_p=[0., 15.], std_p=[5., 10.], future_states_number: int = 7):

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
        print(f"mean = {mean.numpy()}")

    fig = plt.figure()
    plt.plot(mean.numpy(), 'o')
    plt.ylabel('Prediction')
    plt.xlabel('future steps')

    return


def neural_network_images(n_hid: int = 128, activ_hid: str = 'relu', activ_out: str = 'softmax'):

    fashion_mnist = keras.datasets.fashion_mnist  # load dataset

    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()  # split into tetsing and training

    (n_im, n_x, n_y) = train_images.shape
    print(train_images.shape)

    print(train_images[0, 23, 23])  # let's have a look at one pixel

    # let's have a look at the first 10 training labels
    print(train_labels[:10])

    print(type(train_images))

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    plt.figure()
    plt.imshow(train_images[1])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    # normalize data
    max_train = np.amax(train_images)
    train_images = train_images / max_train

    # build the model. Sequential neural network
    # architecture of the NN
    model = keras.Sequential([
        # input layer (1) flattens into n_x * n_y
        keras.layers.Flatten(input_shape=(n_x, n_y)),
        # hidden layer (2) a bit smaller than the input layer
        keras.layers.Dense(n_hid, activation=activ_hid),
        # output layer (3)
        keras.layers.Dense(len(class_names), activation=activ_out)
    ])

    # next we need to pick the optimizer, loss and the metric
    model.compile(
        optimizer='adam',  # gradient descent
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    # hyperparameter tuning

    # train the model
    # we pass the data, labels and epochs and watch the magic!
    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)

    print('Test accuracy:', test_acc)

    predictions = model.predict(test_images)

    return model


def convolutional_neural_network_images(IMG_INDEX: int = 7, sample_size: int = 3, pooling_size: int = 2, filter_number: int = 32, activation: int = 'relu'):

    #  LOAD AND SPLIT DATASET
    (train_images, train_labels), (test_images,
                                   test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    max_train = np.amax(train_images)
    max_test = np.amax(test_images)
    train_images, test_images = train_images / max_train, test_images / max_test

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    len_classes = len(class_names)

    # Let's look at a one image
    IMG_INDEX = IMG_INDEX  # change this to look at other images

    plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
    plt.show()

    # convolutional neural network architecture
    # pooling layer reduces the dimensionality but it is not needed
    model = models.Sequential()
    model.add(layers.Conv2D(filter_number, (sample_size, sample_size),
                            activation=activation, input_shape=(filter_number, filter_number, sample_size)))
    model.add(layers.MaxPooling2D((pooling_size, pooling_size)))
    model.add(layers.Conv2D(2*filter_number,
                            (sample_size, sample_size), activation=activation))
    model.add(layers.MaxPooling2D((pooling_size, pooling_size)))
    model.add(layers.Conv2D(2*filter_number,
                            (sample_size, sample_size), activation=activation))
    # adding a dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(2*filter_number, activation=activation))
    model.add(layers.Dense(len_classes))
    # summarize so far
    model.summary()  # let's have a look at our model so far

    return model, train_images, train_labels, test_images, test_labels


def train_convolutional_neural_network(model, train_images, train_labels, test_images, test_labels, optimizer: str = 'adam', from_logits: bool = True, metrics_str: str = 'accuracy', epochs: int = 4):

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=from_logits),
                  metrics=[metrics_str])

    history = model.fit(train_images, train_labels, epochs=epochs,
                        validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(test_acc)

    return model, history


def data_augmentation(train_images, image_index: int = 20,
                      rotation_range: int = 40,
                      width_shift_range: float = 0.2,
                      height_shift_range: float = 0.2,
                      shear_range: float = 0.2,
                      zoom_range: float = 0.2,
                      horizontal_flip: bool = True,
                      fill_mode: str = 'nearest'
                      ):

    # creates a data generator object that transforms images
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode)

    # pick an image to transform
    test_img = train_images[image_index]
    img = image.img_to_array(test_img)  # convert image to numpy arry
    img = img.reshape((1,) + img.shape)  # reshape image

    i = 0
    # this loops runs forever until we break, saving images to current directory with specified prefix
    for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):
        plt.figure(i)
        plot = plt.imshow(image.img_to_array(batch[0]))
        i += 1
        if i > 4:  # show 4 images
            break

    plt.show()

    return


def use_pretrained_dataset(dataset_name: str = 'cats_vs_dogs', with_info: bool = True, as_supervised: bool = True,
                           split=['train[:80%]',
                                  'train[80%:90%]', 'train[90%:]'],
                           BATCH_SIZE: int = 32, SHUFFLE_BUFFER_SIZE: int = 1000):

    tfds.disable_progress_bar()

    # split the data manually into 80% training, 10% testing, 10% validation
    (raw_train, raw_validation, raw_test), metadata = tfds.load(
        dataset_name,
        split=['train', 'train', 'train'],
        #split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=with_info,
        as_supervised=as_supervised,
    )

    # creates a function object that we can use to get labels
    get_label_name = metadata.features['label'].int2str

    # display 2 images from the dataset
    for image, label in raw_train.take(5):
        plt.figure()
        plt.imshow(image)
        plt.title(get_label_name(label))

    train = raw_train.map(format_example)
    validation = raw_validation.map(format_example)
    test = raw_test.map(format_example)

    for image, label in train.take(2):
        plt.figure()
        plt.imshow(image)
        plt.title(get_label_name(label))

    train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    validation_batches = validation.batch(BATCH_SIZE)
    test_batches = test.batch(BATCH_SIZE)

    for img, label in raw_train.take(2):
        print("Original shape:", img.shape)

    for img, label in train.take(2):
        print("New shape:", img.shape)

    return train_batches, validation_batches


# All images will be resized to 160x160
def format_example(image, label, IMG_SIZE: int = 160):
    """
    returns an image that is reshaped to IMG_SIZE
    """
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


def pick_pretrained(train_batches, trainable: bool = False, IMG_SIZE: int = 160, include_top: bool = False, weights: str = 'imagenet'):

    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,  # don't include the classifier
        weights='imagenet'  # specific save of the weight
    )

    base_model.summary()

    for image, _ in train_batches.take(1):
        pass

    feature_batch = base_model(image)
    print(feature_batch.shape)

    base_model.trainable = trainable

    base_model.summary()

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    prediction_layer = keras.layers.Dense(1)

    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    model.summary()

    return model


def train_pretrained(model, train_batches, validation_batches, base_learning_rate: float = 0.0001, initial_epochs: int = 3, validation_steps: int = 20):

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  # because we are using two classes
                  metrics=['accuracy'])

    # We can evaluate the model right now to see how it does before training
    # it on our new images
    loss0, accuracy0 = model.evaluate(
        validation_batches, steps=validation_steps)

    print(f"loss0 = {loss0}, accuracy0 = {accuracy0}")

    # Now we can train it on our images
    history = model.fit(train_batches,
                        epochs=initial_epochs,
                        validation_data=validation_batches)

    acc = history.history['accuracy']
    print(acc)

    # we can save the model and reload it at anytime in the future
    model.save("dogs_vs_cats.h5")  # .h5 format to save model in keras
    new_model = tf.keras.models.load_model('dogs_vs_cats.h5')

    return new_model


vocab = {}  # maps word to integer representing it
word_encoding = 1


def bag_of_words(text):
    global word_encoding

    # create a list of all of the words in the text, well assume there is no grammar in our text for this example
    words = text.lower().split(" ")
    bag = {}  # stores all of the encodings and their frequency

    for word in words:
        if word in vocab:
            encoding = vocab[word]  # get encoding from vocab
        else:
            vocab[word] = word_encoding
            encoding = word_encoding
            word_encoding += 1

    if encoding in bag:
        bag[encoding] += 1
    else:
        bag[encoding] = 1

    return bag


def test_bag_of_words(text: str = "this is a test to see if this test will work is is test a a"):
    bag = bag_of_words(text)
    print(bag)
    print(vocab)


def movie_review(VOCAB_SIZE: int = 88584):

    (train_data, train_labels), (test_data,
                                 test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

    return train_data, train_labels, test_data, test_labels


def pre_process_movie_review(train_data: np.ndarray, test_data: np.ndarray, MAXLEN: int = 260):

    size = []
    for i in range(len(train_data)):
        m = len(train_data[i])
        size.append(m)

    maxlen = max(size)
    print(f"the actual maximum length is: {maxlen}")
    print(f"the used maximum lenght is: {MAXLEN}")
    # padding
    train_data = sequence.pad_sequences(train_data, MAXLEN)
    test_data = sequence.pad_sequences(test_data, MAXLEN)

    return train_data, test_data


def movie_RNN_model(VOCAB_SIZE: int = 88584):

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 32),
        tf.keras.layers.LSTM(32),
        # either positive or negative review
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    return model


def train_movie_RNN(model, train_data: np.ndarray, train_labels: np.ndarray, epochs: int = 10, validation_split: float = 0.2, loss: str = "binary_crossentropy", optimizer: str = "rmsprop", metrics=['acc']):

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    history = model.fit(train_data, train_labels,
                        epochs=epochs, validation_split=validation_split)

    return model, history


def evaluate_movie_RNN(model, test_data: np.ndarray, test_labels: np.ndarray):

    results = model.evaluate(test_data, test_labels)

    return results


word_index = imdb.get_word_index()


def encode_text(text, MAXLEN: int = 260):
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], MAXLEN)[0]


def evaluate_code_text(text: str = "that movie was just amazing, so amazing", MAXLEN: int = 260):
    encoded = encode_text(text, MAXLEN)
    return encoded


# while were at it lets make a decode function
reverse_word_index = {value: key for (key, value) in word_index.items()}


def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + " "

    return text[:-1]


def predict_en_de(text, model, MAXLEN: int = 260):
    encoded_text = encode_text(text)
    pred = np.zeros((1, MAXLEN))
    pred[0] = encoded_text
    result = model.predict(pred)
    if result[0] > 0.5:
        print(f"The review is positive ({100*float(result[0])}% accuracy)")
    else:
        print(f"The review is negative ({100-100*float(result[0])}% accuracy)")


def play_generator(path_to_file=shakespeare):

    # Read, then decode for py2 compat.
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    # length of text is the number of characters in it
    print('Length of text: {} characters'.format(len(text)))
    # Take a look at the first 250 characters in text
    print(text[:250])

    vocab = sorted(set(text))

    text_as_int = text_to_int(text, vocab)

    # lets look at how part of our text is encoded
    print("Text:", text[:13])
    print("Encoded:", text_to_int(text[:13], vocab))

    print(int_to_text(text_as_int[:13], vocab))

    seq_length = 100  # length of sequence for a training example
    examples_per_epoch = len(text)//(seq_length+1)

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    # we use map to apply the above function to every entry
    dataset = sequences.map(split_input_target)

    for x, y in dataset.take(2):
        print("\n\nEXAMPLE\n")
        print("INPUT")
        print(int_to_text(x, vocab))
        print("\nOUTPUT")
        print(int_to_text(y, vocab))

    return dataset, vocab


def text_to_int(text, vocab):
    # Creating a mapping from unique characters to indices
    char2idx = {u: i for i, u in enumerate(vocab)}
    return np.array([char2idx[c] for c in text])


def int_to_text(ints, vocab):

    idx2char = np.array(vocab)

    try:
        ints = ints.numpy()
    except:
        pass

    return ''.join(idx2char[ints])


def split_input_target(chunk):  # for the example: hello
    input_text = chunk[:-1]  # hell
    target_text = chunk[1:]  # ello
    return input_text, target_text  # hell, ello


def training_batches_play_generator(dataset, vocab, BATCH_SIZE: int = 64, EMBEDDING_DIM: int = 256, RNN_UNITS: int = 1024, BUFFER_SIZE: int = 10000):

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).

    VOCAB_SIZE = len(vocab)  # vocab is number of unique characters
    data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
    model.summary()

    return model, data


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),  # None because we don't know how long is each sequence
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,  # we want the output at each time step
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        # final nodes contain the same number of caracter, is a predictive layer
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def test_play_model(model, data, vocab):

    for input_example_batch, target_example_batch in data.take(1):
        # ask our model for a prediction on our first batch of training data (64 entries)
        example_batch_predictions = model(input_example_batch)
        # print out the output shape
        print(example_batch_predictions.shape,
              "# (batch_size, sequence_length, vocab_size)")

    # we can see that the predicition is an array of 64 arrays, one for each entry in the batch
    print(len(example_batch_predictions))
    print(example_batch_predictions)

    # lets examine one prediction
    pred = example_batch_predictions[0]
    print(len(pred))
    print(pred)
    # notice this is a 2d array of length 100, where each interior array is the prediction for the next character at each time step

    # and finally well look at a prediction at the first timestep
    time_pred = pred[0]
    print(len(time_pred))
    print(time_pred)
    # and of course its 65 values representing the probabillity of each character occuring next

    # If we want to determine the predicted character we need to sample the output distribution (pick a value based on probabillity)
    sampled_indices = tf.random.categorical(pred, num_samples=1)

    # now we can reshape that array and convert all the integers to numbers to see the actual characters
    sampled_indices = np.reshape(sampled_indices, (1, -1))[0]
    predicted_chars = int_to_text(sampled_indices, vocab)

    predicted_chars  # and this is what the model predicted for training sequence 1

    return


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def pre_train_model(model):

    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    return model, checkpoint_callback


def train_play(model, data, epochs: int = 50):

    model, checkpoint_callback = pre_train_model(model)
    history = model.fit(data, epochs=epochs, callbacks=[checkpoint_callback])

    return model, history


def loading_play(model, vocab, EMBEDDING_DIM: int = 256, RNN_UNITS: int = 1024):

    VOCAB_SIZE = len(vocab)  # vocab is number of unique characters

    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    try:
        checkpoint_num = 10
        model.load_weights(tf.train.load_checkpoint(
            "./training_checkpoints/ckpt_" + str(checkpoint_num)))
        model.build(tf.TensorShape([1, None]))
    except:
        print(f"No checkpoint = {checkpoint_num}")

    return


def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 800

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension

        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


def q_learning_example(make_str: str = 'FrozenLake-v0'):

    env = gym.make(make_str)  # we are going to use the FrozenLake enviornment
    print(env.observation_space.n)   # get number of states
    print(env.action_space.n)   # get number of actions

    env.reset()  # reset enviornment to default state

    action = env.action_space.sample()  # get a random action

    # take action, notice it returns information about the action
    new_state, reward, done, info = env.step(action)

    print(f"new_state = {new_state}")
    print(f"reward = {reward}")
    print(f"done = {done}")
    print(f"info = {info}")

    env.render()   # render the GUI for the enviornment


def q_learning_setup(make_str: str = 'FrozenLake-v0',
                     EPISODES: int = 2000,  # how many times to run the enviornment from the beginning
                     MAX_STEPS: int = 100,  # max number of steps allowed for each run of enviornment
                     LEARNING_RATE: float = 0.81,  # learning rate
                     GAMMA: float = 0.96,
                     RENDER: bool = False,  # if you want to see training set to true
                     epsilon: float = 0.9):  # start with a 90% chance of picking a random action

    env = gym.make(make_str)
    STATES = env.observation_space.n
    ACTIONS = env.action_space.n
    Q = np.zeros((STATES, ACTIONS))  # create a matrix with all 0 values

    '''
    # code to pick action
    if np.random.uniform(0, 1) < epsilon:  # we will check if a randomly selected value is less than epsilon.
        action = env.action_space.sample()  # take random action
    else:
        action = np.argmax(Q[state, :])  # use Q table to pick best action based on current values
    '''

    rewards = []
    for episode in range(EPISODES):

        state = env.reset()
        for _ in range(MAX_STEPS):
            if RENDER:
                env.render()

            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, done, _ = env.step(action)

            Q[state, action] = Q[state, action] + LEARNING_RATE * \
                (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

            state = next_state

            if done:
                rewards.append(reward)
                epsilon -= 0.001
                break  # reached goal

    print(Q)
    print(f"Average reward: {sum(rewards)/len(rewards)}:")
    # and now we can see our Q values!

    def get_average(values):
        return sum(values)/len(values)

    avg_rewards = []
    for i in range(0, len(rewards), 100):
        avg_rewards.append(get_average(rewards[i:i+100])) 

    plt.plot(avg_rewards)
    plt.ylabel('average reward')
    plt.xlabel('episodes (100\'s)')
    plt.show()

    return Q