from Network import *
import numpy as np
from matplotlib import pyplot as plt

class NetGrapher:
    def showPredictions(self, predictions, images):
        for i in range(100):
            image = images[i].reshape((28, 28)) * 255
            plt.subplot(10, 10, i+1)
            plt.imshow(image, interpolation='nearest')
            plt.title(str(predictions[i]))
            plt.axis('off')
        plt.subplots_adjust(wspace=2, hspace=5)
        plt.show()
    def showBarChart(self, predictions, testTargets):
        correctCounts = {digit: 0 for digit in range(10)}
        totalCounts = {digit: 0 for digit in range(10)}
        for i in range(len(predictions)):
            totalCounts[testTargets[i]] += 1
            if predictions[i] == testTargets[i]:
                correctCounts[testTargets[i]] += 1

        percentages = {digit: (correctCounts[digit] / totalCounts[digit]) * 100 if totalCounts[digit] > 0 else 0 for digit in range(10)}

        digits = list(percentages.keys())
        percentages_values = list(percentages.values())

        plt.bar(digits, percentages_values)
        plt.xlabel('Digit')
        plt.ylabel('Percentage of Correct Predictions')
        plt.title('Percentage of Correct Predictions per Digit')
        plt.xticks(digits)
        plt.grid(axis='y')
        plt.show()
