import numpy as np

class Activation:
    def calculate(inputs):
        pass

class ReLU(Activation):
    def calculate(self, inputs):
        return np.maximum(0, inputs)

class Softmax(Activation):
    def calculate(self, inputs):
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        normValues = expValues / np.sum(expValues)
        return normValues