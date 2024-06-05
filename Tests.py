from Layer import *
from Activation import *
import numpy as np
import random as rand
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()

trainImages, testImages = trainImages / 255.0, testImages / 255.0

relu = ReLU()
softmax = Softmax()

layer1 = Layer(4, 4, relu)
layer2 = Layer(4, 3, softmax)
list = []

for i in range(4):
    integer = rand.random()
    list.append(integer)

input = np.array(list)

layer1.forward(input)
print(layer1.output)
print(layer1.output.shape)

layer2.forward(layer1.output)
print(layer2.output)