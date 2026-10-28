# Lab 3: Comparison of Neural Network Architectures



## 1. Objective



The objective of this laboratory was to implement, train, and compare three neural network architectures for Fashion-MNIST image classification: a Multilayer Perceptron (MLP), a Convolutional Neural Network (CNN), and an Attention model based on Transformer encoder layers.



The comparison considers classification accuracy, test loss, number of trainable parameters, training time, inference latency, and computational complexity measured in floating-point operations (FLOPs).



## 2. Experimental Setup



The Fashion-MNIST dataset contains grayscale images of clothing items with dimensions of 28 × 28 pixels and ten target classes.



The images were converted to tensors and normalized using a mean of 0.2860 and a standard deviation of 0.3530.



The training dataset was divided into 55,000 training images and 5,000 validation images. The separate test dataset contained 10,000 images.



The experiments used the following configuration:



- Random seed: 42

- Batch size: 128

- Number of epochs: 5

- Optimizer: Adam

- Learning rate: 0.001

- Loss function: Cross-entropy loss

- Device: CPU



The same dataset and training configuration were used for all three architectures.



## 3. Model Architectures



### 3.1 Multilayer Perceptron (MLP)



The MLP flattens each 28 × 28 image into a vector containing 784 values. It contains three fully connected layers with dimensions 784 → 256 → 128 → 10. ReLU activation functions are applied after the first two layers.



### 3.2 Convolutional Neural Network (CNN)



The CNN contains two convolutional layers. The first convolution changes the number of channels from 1 to 16, while the second changes it from 16 to 32. Both use 3 × 3 kernels and ReLU activation.



Each convolutional layer is followed by 2 × 2 max pooling. The resulting 32 × 7 × 7 feature representation is flattened and passed to a fully connected classification layer with ten outputs.



### 3.3 Attention Model



The Attention model divides each image into 16 non-overlapping patches of size 7 × 7 pixels. Each patch contains 49 values and is projected into a 64-dimensional feature vector.



The model uses two Transformer encoder layers, each with four attention heads and a feed-forward dimension of 128. The encoded patch representations are averaged and passed to a final linear classification layer with ten outputs.



## 4. Trainable Parameter Calculations



For a fully connected layer, the number of trainable parameters is:



Parameters = (number of inputs × number of outputs) + number of biases.



### 4.1 MLP



The three fully connected layers contain:



- First layer: 784 × 256 + 256 = 200,960

- Second layer: 256 × 128 + 128 = 32,896

- Output layer: 128 × 10 + 10 = 1,290



Total parameters:



200,960 + 32,896 + 1,290 = 235,146



### 4.2 CNN



For a convolutional layer, the parameter count is:



Parameters = (kernel height × kernel width × input channels × output channels) + output-channel biases.



- First convolution: 3 × 3 × 1 × 16 + 16 = 160

- Second convolution: 3 × 3 × 16 × 32 + 32 = 4,640

- Classification layer: 1,568 × 10 + 10 = 15,690



Total parameters:



160 + 4,640 + 15,690 = 20,490



### 4.3 Attention Model



The patch projection contains:



49 × 64 + 64 = 3,200 parameters.



Each Transformer encoder layer contains 33,472 trainable parameters, including its attention projections, feed-forward network, biases, and normalization parameters. With two encoder layers, this gives:



2 × 33,472 = 66,944



The final classification layer contains:



64 × 10 + 10 = 650 parameters.



Total parameters:



3,200 + 66,944 + 650 = 70,794



The total is 3,200 + 66,944 + 650 = 70,794, matching the count measured from the actual PyTorch model.

## 5. FLOPs Calculations



FLOPs estimate the number of floating-point operations needed for a forward pass. For matrix multiplication, this report uses the convention that one multiply-accumulate (MAC) operation is approximately two FLOPs: one multiplication and one addition.



Bias additions, activation functions, normalization, pooling, and other smaller operations are excluded from the simplified calculations.



### 5.1 MLP



The MAC count for the three linear layers is:



784 × 256 + 256 × 128 + 128 × 10 = 234,752 MACs



Therefore:



FLOPs ≈ 2 × 234,752 = 469,504



### 5.2 CNN



The first convolution requires:



28 × 28 × 16 × 3 × 3 × 1 = 112,896 MACs



The second convolution operates on a 14 × 14 feature map:



14 × 14 × 32 × 3 × 3 × 16 = 903,168 MACs



The final classification layer requires:



1,568 × 10 = 15,680 MACs



Total:



112,896 + 903,168 + 15,680 = 1,031,744 MACs



Therefore:



FLOPs ≈ 2 × 1,031,744 = 2,063,488



### 5.3 Attention Model



The hand-calculation script estimates 2,329,856 FLOPs per image. This includes the patch projection, the major Transformer matrix multiplications, and the final classification layer.



The THOP profiler reports 1,182,976 FLOPs for the Attention model. This differs from the hand-calculated estimate because automated profilers may not count all Transformer operations in the same way. The two values should therefore be treated as estimates using different counting methods, not as identical measurements.



## 6. Experimental Results



The following table summarizes the measured results.



| Metric | MLP | CNN | Attention |

|---|---:|---:|---:|

| Trainable parameters | 235,146 | 20,490 | 70,794 |

| Test loss | 0.3355 | 0.2819 | 0.4627 |

| Test accuracy | 87.77% | 90.11% | 83.75% |

| Training time | 34.46 s | 59.80 s | 111.93 s |

| Batch-1 inference latency | 0.0559 ms | 0.3255 ms | 0.4559 ms |

| Hand-calculated FLOPs | 469,504 | 2,063,488 | 2,329,856 |

| THOP-estimated FLOPs | 469,504 | 2,063,488 | 1,182,976 |



The training times cover five training epochs on the CPU. Inference latency was measured separately after warm-up runs. Batch-1 latency represents a single image, while batch-128 latency represents a batch of 128 images.



## 7. Discussion



The CNN achieved the highest test accuracy at 90.11% and the lowest test loss at 0.2819. It also had the fewest trainable parameters, with 20,490. These results indicate that convolutional feature extraction was effective for this image-classification task.



The MLP had the shortest training time at 34.46 seconds and the lowest single-image inference latency at 0.0559 milliseconds. However, its test accuracy of 87.77% was lower than that of the CNN.



The Attention model achieved 83.75% test accuracy and required the longest training time, at 111.93 seconds. Its single-image inference latency was also the highest of the three models. In this experiment, the additional Transformer operations did not lead to better classification performance.



The results demonstrate that the number of parameters alone does not determine computational cost or inference speed. The CNN had fewer parameters than the MLP but higher measured inference latency. Architecture, operation type, and implementation all affect performance.



The FLOPs results also illustrate a limitation of automated profiling. Although the profiler and hand-calculation estimates agree for the MLP and CNN, they differ for the Attention model. A more detailed operation-by-operation analysis would be needed to establish a fully consistent FLOPs count for the Transformer.



## 8. Limitations



The experiments were conducted on a CPU and used five training epochs. Training times and inference latency may differ on other hardware, with different software configurations, or after further optimization.



The reported latency values are measurements from a particular environment rather than universal performance guarantees. The FLOPs calculations are simplified estimates and do not include every operation, such as all normalization, activation, pooling, and tensor-manipulation operations.



The experiment also compares one implementation of each architecture. The results do not establish that one architecture is always superior for every image-classification task.



## 9. Conclusion



This laboratory compared an MLP, a CNN, and a Transformer-based Attention model on Fashion-MNIST.



The CNN achieved the best test accuracy and used the fewest trainable parameters. The MLP was the fastest to train and had the lowest single-image inference latency. The Attention model was the slowest to train and achieved the lowest test accuracy in this experiment.



The results show the importance of evaluating neural network architectures using several criteria, including accuracy, parameter count, computational complexity, training time, and inference latency. No single metric provides a complete assessment of model efficiency.



The experiment also highlighted the importance of validating automated profiler outputs against hand calculations, particularly for architectures containing Transformer layers.

S

