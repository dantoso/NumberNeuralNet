from Layer import *
from Activation import *
import numpy as np

class Network:
    def __init__(self, pixelNum: int):
        self.hiddenActivation = ReLU()
        self.outputActivation = Softmax()
        self.inputSize = pixelNum
        self.hiddenSize = 16
        self.hiddenLayers = []

        self.hiddenLayers.append(Layer(self.inputSize, self.hiddenSize, self.hiddenActivation))
        self.hiddenLayers.append(Layer(self.hiddenSize, self.hiddenSize, self.hiddenActivation))

        self.outputLayer = Layer(self.hiddenSize, 10, self.outputActivation)
    
    def run(self, input):
        self.input = input
        forwarded = input

        for layer in self.hiddenLayers:
            layer.forward(forwarded)
            forwarded = layer.output
            
        self.outputLayer.forward(forwarded)
        self.predictions = np.argmax(self.outputLayer.output, axis=1)

    def test(self, testData, targets):
        self.run(testData)
        self.accuracy = np.mean(self.predictions == targets)
        return self.accuracy
    
    def fit(self, trainData, targets, iterations=1000, alpha=0.1):
        for i in range(iterations):
            self.run(trainData)
            self.accuracy = np.mean(self.predictions == targets)
            if i % 10 == 0:
                print(" ---------------------------- Iteration: ", i, " ---------------------------- ")
                print("Loss: ", self.getLoss(targets))
                print("Accuracy: ", self.accuracy)

            oneHot = self.oneHotEncode(targets)
            dParams = self.backProp(oneHot)
            self.updateParams(dParams, alpha)
    
    def evaluate(self, input):
        self.run(input)
        return self.predictions
    
    def oneHotEncode(self, targets):
        oneHot = np.zeros((len(targets), 10))
        for i in range(len(targets)):
            oneHot[i, targets[i]] = 1
        return oneHot
    
    def updateParams(self, dParams, alpha):
        for i in range(len(self.hiddenLayers)):
            index = -1-i
            layer = self.hiddenLayers[i]
            modifiers = dParams[index]
            self.updateLayer(layer, modifiers, alpha)
        self.updateLayer(self.outputLayer, dParams[0], alpha)
    
    def updateLayer(self, layer, modifiers, alpha):
        layer.weights = layer.weights - alpha * modifiers[0]
        layer.biases = layer.biases - alpha * modifiers[1]

    def backProp(self, oneHot):
        numSamples = len(oneHot)
        dParams = []
        nextLayer = self.outputLayer
        dZ = self.outputLayer.output - oneHot # m x 10
        
        # Backpropagation through hidden layers
        for i in range(len(self.hiddenLayers)):
            index = -1 - i
            layer = self.hiddenLayers[index]

            # Compute gradients for weights and biases
            dW = np.dot(layer.output.T, dZ) / numSamples
            dB = np.sum(dZ, axis=0, keepdims=True) / numSamples

            # Save gradients
            dParams.append([dW, dB])

            # Compute gradients for next layer
            dZ = np.dot(dZ, nextLayer.weights.T) * self.hiddenActivation.deriv(layer.unactivatedResult)
            nextLayer = layer
        
        # Backpropagation through input layer
        dW = np.dot(self.input.T, dZ) / numSamples
        dB = np.sum(dZ, axis=0, keepdims=True) / numSamples
        dParams.append([dW, dB])

        return dParams
    
    def getLoss(self, targets):
        clippedVal = np.clip(self.outputLayer.output, 1e-7, 1-1e-7)
        numSamples = len(clippedVal)

        relevantConfidences = clippedVal[range(numSamples), targets]

        loss = -np.log(relevantConfidences)

        return np.mean(loss)
