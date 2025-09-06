This project aims to build a neural network from the ground up, without relying on any machine learning libraries or frameworks. Working with the classic MNIST dataset, which contains 60,000 training and 10,000 test grayscale images (28×28 pixels) of handwritten digits (0–9).


Objective

Building a model that can accurately classify handwritten digits by implementing all the fundamental components of a neural network manually. This includes:

- Creating custom structures for neurons and layers
- Writing code for forward propagation
- Implementing backpropagation to compute gradients and update weights
- Experimenting with different activation functions and loss functions
- Training using gradient descent and observing how various hyperparameters affect performance


Core Components

- Neurons & Layers: Create classes to represent individual neurons and layer connections.
- Forward Propagation: Manually compute outputs as data flows through layers using chosen activation functions.
- Backpropagation: Calculate gradients for each weight and bias using the chain rule and update parameters.

Features to Experiment With

- Activation Functions:
    - Softmax
    - ReLU
- Loss Functions:
    - Mean Squared Error (MSE)
    - Cross-Entropy Loss
- Optimization:
    - Stochastic Gradient Descent (SGD)
- Initialization:
    - Random weight and bias initialization
 

