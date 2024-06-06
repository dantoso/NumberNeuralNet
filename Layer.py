import numpy as np
from Activation import *

class Layer:
    def __init__(self, inputNum: int, neuronNum: int, activation: Activation):
        self.weights = np.random.normal(loc=0.0, scale=0.01, size=(inputNum, neuronNum)) * 0.1
        self.biases = np.random.normal(loc=0.0, scale=0.01, size=(1, neuronNum)) * 0.1
        self.activation = activation
    
    def forward(self, inputs) -> None:
        result = np.dot(inputs, self.weights) + self.biases
        self.unactivatedResult = result
        self.output = self.activation.calculate(result)
