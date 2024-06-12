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