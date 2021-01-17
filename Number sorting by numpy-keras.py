from keras.models import Sequential
from keras.layers.core import Activation, RepeatVector, Dropout, Dense
from keras.layers import recurrent
import numpy as np

# global parameters.
batch_size = 32
numbers_in_input_set = 10
maximum_number_in_set = 100


def encoding_loop(inputed_vector, matrix):
    """
    This Function is used for converting input to matrix by changing the corresponding element of matrix to 1 from 0
    """
    for i, j in enumerate(inputed_vector):
        for k, l in enumerate(j):
            matrix[i, k, l] = 1
    return matrix


def matrix_zero_maker(no_matrix, rows=numbers_in_input_set, columns=maximum_number_in_set):
    """
    This Function is used for making matrix of zeros for base encoding process with numpy
    """
    matrix_zeros = np.zeros((no_matrix, rows, columns), dtype=np.float32)
    return matrix_zeros


def hot_encoder(input_set):
    """
    This function is hot encoder that convert set to matrix so it can be used in machine learning
    """
    input_set_matrix = matrix_zero_maker(len(input_set))
    encoding_loop(input_set, input_set_matrix)
    return input_set_matrix


def batch_generator(batches_size):
    """
    This Function is used for generating and encoding infinite training batches
    """
    input_set_encoded = matrix_zero_maker(batches_size)
    sorted_input_set_encoded = matrix_zero_maker(batches_size)

    while True:
        input_set_vector = np.random.randint(maximum_number_in_set, size=(batches_size, numbers_in_input_set))

        sorted_input_set_vector = np.sort(input_set_vector, axis=1)

        encoding_loop(input_set_vector, input_set_encoded)

        encoding_loop(sorted_input_set_vector, sorted_input_set_encoded)

        yield input_set_encoded, sorted_input_set_encoded
        input_set_encoded.fill(0.0)
        sorted_input_set_encoded.fill(0.0)


# Model

model = Sequential()

# Layers

model.add(recurrent.LSTM(units=100, input_shape=(numbers_in_input_set, maximum_number_in_set)))
model.add(Dropout(0.25))
model.add(RepeatVector(numbers_in_input_set))
model.add(recurrent.LSTM(100, return_sequences=True))
model.add(Dense(maximum_number_in_set))
model.add(Dropout(0.5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Training the model
for iteration, (input_train, input_train_sorted) in enumerate(batch_generator(batch_size)):
    loss, acc = model.train_on_batch(input_train, input_train_sorted)

    if iteration % 500 == 0:
        test_data = np.random.randint(maximum_number_in_set, size=(1, numbers_in_input_set))
        print('Loss: ', loss, 'Accuracy: ', acc)
        encoded_test_data = hot_encoder(test_data)
        print('Input data for test is : ', test_data)
        predicted_test_data = model.predict(encoded_test_data, batch_size=1)
        print("Sorted data by numpy is : ", np.sort(test_data))
        print("Sorted data by my model is :", np.argmax(predicted_test_data, axis=2), "\n")
