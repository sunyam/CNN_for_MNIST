import keras

# Loading MNIST dataset which is in-built in Keras

from keras.datasets import mnist

print "Loading Data...."

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print "Data has been loaded: \n\n"

print x_train.shape
print y_train.shape
print x_test.shape
print y_test.shape
