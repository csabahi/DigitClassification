import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import time


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print (len(x_train))

# print(len(x_test))

# print(x_train[0].shape)

print(y_train[:5])

x_train_flattened = x_train.reshape(len(x_train),28*28)
x_test_flattened = x_test.reshape(len(x_test),28*28)
print(x_train_flattened.shape)