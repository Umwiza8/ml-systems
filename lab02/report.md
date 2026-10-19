# Lab 2: A Neural Network from Scratch

## 1. Objective

The aim of this lab was to build a neural network from scratch using NumPy. I implemented forward propagation, backpropagation, softmax, and cross-entropy loss. I also tested the gradients and compared different network sizes and learning rates.

## 2. Dataset and Network

I used the handwritten digits dataset from `sklearn.datasets.load_digits`. The images have 64 input features and 10 possible output classes. I divided the data into 70% training data and 30% test data, using `random_state=42`.

The network had two hidden layers with either 16 or 64 neurons per layer. I used ReLU for the hidden layers and softmax for the output layer. The weights were initialized with a normal distribution (mean 0, standard deviation 0.01), and the biases were set to zero.

## 3. Gradient Checking

I checked the manually calculated gradients against numerical gradients using a small value of \(h=10^{-5}\). All 10 tested weights passed the relative-error tolerance of \(10^{-6}\). This gave me confidence that my backpropagation implementation was working correctly for the tested weights.

## 4. Training Results

I trained the network for 50 epochs with a batch size of 32 and tested three learning rates.

| Hidden neurons | Learning rate | Final loss | Test accuracy |
|---:|---:|---:|---:|
| 16 | 0.001 | 2.302550 | 10.19% |
| 16 | 0.01 | 2.302476 | 10.19% |
| 16 | 0.1 | 1.597017 | 32.59% |
| 64 | 0.001 | 2.302526 | 10.19% |
| 64 | 0.01 | 2.302343 | 10.19% |
| 64 | 0.1 | 0.168405 | 93.89% |

The best result came from the network with 64 neurons per hidden layer and a learning rate of 0.1, which achieved 93.89% test accuracy. The smaller learning rates did not improve the results much during the 50 epochs.

## 5. Performance

I measured the median forward and backward pass times over 100 runs.

| Hidden neurons | Forward pass | Backward pass |
|---:|---:|---:|
| 16 | 0.0240 ms | 0.0275 ms |
| 64 | 0.0391 ms | 0.0532 ms |

The backward pass took longer than the forward pass because it also calculates gradients. The network with 64 neurons was slower because it performs more computations.

## 6. Conclusion

In this lab, I learned how the main steps of neural network training work by implementing them myself. I found that the network size and learning rate made a big difference in the results. The best configuration achieved 93.89% test accuracy, while the gradient checks helped me verify my implementation. This lab also showed me that a larger network can perform better but requires more computation.