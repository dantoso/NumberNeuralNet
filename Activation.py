import numpy as np

class Activation:
    def calculate(inputs):
        pass
    def deriv(x):
        pass

class ReLU(Activation):
    def calculate(self, inputs):
        return np.maximum(0, inputs)
    def deriv(x):
        return x > 0

class Softmax(Activation):
    def calculate(self, inputs):
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        normValues = expValues / np.sum(expValues)
        return normValues
    def deriv(x):
        return 1