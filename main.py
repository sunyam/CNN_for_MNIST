import keras

# Loading MNIST dataset which is in-built in Keras

from keras.datasets import mnist

print "Loading Data...."

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print "Data has been loaded."

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
    







