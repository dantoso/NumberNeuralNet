from Network import *
import numpy as np
import random as rand
import numpy as np
from MNISTDataLoader import *

loader = MnistDataloader("MNISTData/train/images", "MNISTData/train/labels", "MNISTData/test/images", "MNISTData/test/labels")
(x_train, y_train),(x_test, y_test) = loader.load_data()

x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0], -1)
y_train = np.array(y_train)
x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = np.array(y_test)

x_train, x_test = x_train / 255.0, x_test / 255.0

trainData = x_train[:256]
targets = y_train[:256]

model = Network(28*28)
model.fit(trainData, targets, alpha=0.1)