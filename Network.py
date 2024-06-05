from Layer import *
from Activation import *
import numpy as np

class Network:
    def __init__(self, pixelNum: int):
        self.hiddenActivation = ReLU()
        self.outputActivation = Softmax()
        self.inputSize = pixelNum*pixelNum
        self.hiddenSize = 16

        self.hidden1 = Layer(self.inputSize, self.hiddenSize, self.hiddenActivation)
        self.hidden2 = Layer(self.hiddenSize, self.hiddenSize, self.hiddenActivation)
        self.outputLayer = Layer(self.hiddenSize, 10)
    
    def run(self, input):
        self.hidden1.forward(input)
        self.hidden2.forward(self.hidden1.output)
        self.outputLayer.forward(self.hidden2.output)
        self.predictions = np.argmax(self.outputLayer.output, axis=1)

    def test(self, testData, targets):
        self.run(testData)
        self.accuracy = np.mean(self.predictions == targets)
        return self.accuracy

    def fit(self, trainData, targets):
        self.run(trainData)
        loss = self.getLoss(targets)
        meanLoss = np.mean(loss)
        # call backprop
        # call optimizer
    
    def evaluate(self, input):
        self.run(input)
        return self.outputLayer.output
    
    def getLoss(self, targets):
        clippedVal = np.clip(self.outputLayer.output, 1e-7, 1-1e-7)
        numSamples = len(clippedVal)

        relevantConfidences = clippedVal[range(numSamples), targets]

        loss = -np.log(relevantConfidences)
        return loss

    def backprop(self, loss):

        return