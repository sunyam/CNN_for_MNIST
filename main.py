import keras

# Loading MNIST dataset which is in-built in Keras

from keras.datasets import mnist

print "\n\nLoading Data...."

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print "\nData has been loaded.\n"

#print x_train.shape
#print y_train.shape
#print x_test.shape
#print y_test.shape

# Resize the data to [numberOfSamples][channels][width][height]
training_samples = x_train.shape[0]
test_samples = x_test.shape[0]

x_train = x_train.reshape(training_samples, 1, 28, 28).astype('float32')
x_test = x_test.reshape(test_samples, 1, 28, 28).astype('float32')

# channels=1 (grayscale images)
# Normalise it to range 0-1
x_train /= 255
x_test /= 255

#print x_train.shape
#print x_test.shape

# Convert Y to one-hot vectors
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
numberOfClasses = y_test.shape[1]

# DEFINE our network architecture
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

def create_network():
    network = Sequential()
    network.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
    network.add(MaxPooling2D(pool_size=(2, 2)))
    network.add(Dropout(0.2))
    network.add(Flatten())
    network.add(Dense(128, activation='relu'))
    network.add(Dense(numberOfClasses, activation='softmax'))

    network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return network

# Creating network
network = create_network()

print "\n\nNetwork has been created.."

# Training our network
network.fit(x_train, y_train, nb_epoch=10, batch_size=200, verbose=2)

print "\n\nNetwork has been trained.."

# Evaluating the network results
score = network.evaluate(x_test, y_test, verbose=0)
print "\n\nAccuracy achieved: ", score