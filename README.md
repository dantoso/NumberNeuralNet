# My first neural network
This was a personal project, where I coded a neural network from scratch using Python.

## How to run it
To see it in action, run the Tests.py file. If you just want a quick glance at the results, check the Charts folder. It contains charts and images with samples and results.

## Performance and info
The Tests.py file should run in about 30s.
For a more in depth look on performance, I recommend checking out the Charts folder.

### Training
- Training samples: 60 000 (MNIST dataset)
- Batch size: 60
- Epochs (per batch): 50
- Learning rate (per batch): 0.01

### Network
- Layer 1: 784x16, ReLU activation
- Layer 2: 16x16, ReLU activation
- Output Layer: 16x10, Softmax activation

### Testing
- Test samples: 10 000 (MNIST dataset)
- Tests: 10
- Average accuracy: 87%
