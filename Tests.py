from Network import *
import numpy as np
from MNISTDataLoader import *
from matplotlib import pyplot as plt

loader = MnistDataloader("MNISTData/train/images", "MNISTData/train/labels", "MNISTData/test/images", "MNISTData/test/labels")
(x_train, y_train),(x_test, y_test) = loader.load_data()

x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0], -1)
y_train = np.array(y_train)
x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = np.array(y_test)

x_train, x_test = x_train / 255.0, x_test / 255.0

model = Network(28*28)

split = 1000
trainBatches = np.split(x_train, split)
targetBatches = np.split(y_train, split)

for i in range(split):
    model.fit(trainBatches[i], targetBatches[i], iterations=100, alpha=0.01)

testData = x_test
testTargets = y_test
accuracy = model.test(testData, testTargets)
print("Final accuracy: ", accuracy)
