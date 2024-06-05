import numpy as np
from Activation import *

class Layer:
    def __init__(self, inputNum: int, neuronNum: int, activation: Activation):
        np.random.seed(0)
        self.weights = 0.10 * np.random.randn(inputNum, neuronNum)
        self.biases = np.zeros((1, neuronNum))
        self.activation = activation
    
    def forward(self, inputs) -> None:
        result = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation.calculate(result)
