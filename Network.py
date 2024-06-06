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
            # layer.output -= np.max(layer.output, axis=1, keepdims=True)
            forwarded = layer.output
            
        self.outputLayer.forward(forwarded)
        self.predictions = np.argmax(self.outputLayer.output, axis=1)

    def test(self, testData, targets):
        self.run(testData)
        self.accuracy = np.mean(self.predictions == targets)
        return self.accuracy
    
    def fit(self, trainData, targets, iterations=1000):
        np.random.shuffle(trainData)

        for i in range(iterations):
            batch = trainData[range(32*i, 32*(i+1))]
            targetBatch = targets[range(32*i, 32*(i+1))]
            self.run(batch)
            self.accuracy = np.mean(self.predictions == targetBatch)
            if i % 10 == 0:
                print(" ---------------------------- Iteration: ", i, " ---------------------------- ")
                print("first 3 of output layer = ", self.outputLayer.output[:3])
                print("Loss: ", self.getLoss(targetBatch))
                print("Accuracy: ", self.accuracy)

            oneHot = self.oneHotEncode(targetBatch)
            dParams = self.backProp(oneHot)
            self.updateParams(dParams, 0.01)
    
    def evaluate(self, input):
        self.run(input)
        return self.outputLayer.output
    
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
