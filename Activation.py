import numpy as np

class Activation:
    def calculate(inputs):
        pass
    def deriv(x):
        pass

class ReLU(Activation):
    def calculate(self, inputs):
        return np.maximum(0, inputs)
    def deriv(self, x):
        return np.where(x > 0, 1, 0)

class Softmax(Activation):
    def calculate(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # Subtract the maximum value for numerical stability
        return exp / np.sum(exp, axis=1, keepdims=True)
    def deriv(self, x):
        return 1