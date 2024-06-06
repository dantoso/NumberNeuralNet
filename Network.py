from Layer import *
from Activation import *
import numpy as np

class Network:
    def __init__(self, pixelNum: int):
        self.hiddenActivation = ReLU()
        self.outputActivation = Softmax()
        self.inputSize = pixelNum*pixelNum
        self.hiddenSize = 16
        self.hiddenLayers = []

        self.hiddenLayers.append(Layer(self.inputSize, self.hiddenSize, self.hiddenActivation))
        self.hiddenLayers.append(Layer(self.hiddenSize, self.hiddenSize, self.hiddenActivation))

        self.outputLayer = Layer(self.hiddenSize, 10)
    
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
    
    def fit(self, trainData, targets):
        self.run(trainData)
        oneHot = self.oneHotEncode(targets)
        dParams = self.backProp(oneHot)
        self.updateParams(dParams, 0.1)
    
    def evaluate(self, input):
        self.run(input)
        return self.outputLayer.output
    
    def oneHotEncode(targets):
        oneHot = np.zeros((10, len(targets)))
        for i in range(len(targets)):
            oneHot[i, targets[i]] = 1
        return oneHot
    
    def updateParams(self, dParams, alpha):
        for i in range(len(self.hiddenLayers)):
            layer = self.hiddenLayers[i]
            modifiers = dParams[i]
            self.updateLayer(layer, modifiers, alpha)
        self.updateLayer(self.outputLayer, dParams[-1], alpha)
    
    def updateLayer(self, layer, modifiers, alpha):
        layer.weights = layer.weights - alpha * modifiers[0]
        layer.biases = layer.biases - alpha * modifiers[1]

    def backProp(self, oneHot):
        numSamples = len(oneHot)
        dParams = np.array([])
        nextLayer = self.outputLayer
        dZ = self.outputLayer.output - oneHot # m x 10
        for i in range(len(self.hiddenLayers)):
            index = -1-i
            layer = self.hiddenLayers[index]

            if i != 0:
                # m x 16 = m x 10 * transpose(m x 10) * deriv(m x 16)
                dZ = np.dot(dZ, np.transpose(nextLayer.weights)) * self.hiddenActivation.deriv(layer.unactivatedResult)
                nextLayer = layer      
            # 16 x 10 = transpose(m x 16) * m x 10
            dW = np.dot(np.transpose(layer.output), dZ) * (1/numSamples)

            # 1 x 10
            dB = np.sum(dZ, 1) * (1/numSamples)           

            # save dParams
            dParams = np.append(dParams, [dW, dB])

        layer = self.hiddenLayers[0]
        dZ = np.dot(dZ, np.transpose(nextLayer.weights)) * self.hiddenActivation.deriv(layer.unactivatedResult)
        dW = np.dot(np.transpose(self.input), dZ) * (1/numSamples)
        dB = np.sum(dZ, 1) * (1/numSamples)           
        dParams = np.append(dParams, [dW, dB])

        return dParams