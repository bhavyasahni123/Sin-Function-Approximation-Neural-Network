# Project 4 --- Sin(x) Function Approximation Using Neural Network

A neural network implemented **from scratch in C++** to approximate the
mathematical function:

\[ y = `\sin`{=tex}(x) \]

The trained parameters are then used in **Python** to generate
predictions and visualize the neural-network approximation against the
actual `sin(x)` curve.

No PyTorch, TensorFlow, or scikit-learn was used for the neural-network
implementation.

------------------------------------------------------------------------

## 1. Objective

The objective of this project is to train a small feed-forward neural
network to learn the relationship:

\[ x `\rightarrow `{=tex}`\sin`{=tex}(x) \]

Instead of explicitly programming the neural network to calculate
`sin(x)`, the network learns an approximation from generated
input-output examples.

The project demonstrates:

-   Dataset generation
-   Feed-forward neural networks
-   Sigmoid activation
-   Linear output activation
-   Forward propagation
-   Backpropagation
-   Batch gradient descent
-   SSE and MSE calculation
-   Function approximation
-   C++ implementation
-   Python visualization

------------------------------------------------------------------------

## 2. Neural Network Architecture

The network uses the following architecture:

``` text
Input Layer       Hidden Layer          Output Layer

    x          ┌── h1 ──┐
               ├── h2 ──┤
               ├── h3 ──┤
               ├── h4 ──┤──────► ŷ
               └── h5 ──┘

             1 : 5 : 1
```

### Architecture

  Layer      Neurons Activation
  -------- --------- ------------
  Input            1 None
  Hidden           5 Sigmoid
  Output           1 Linear

Total trainable parameters:

\[ 5 + 5 + 5 + 1 = 16 \]

There are:

-   5 input-to-hidden weights
-   5 hidden biases
-   5 hidden-to-output weights
-   1 output bias

------------------------------------------------------------------------

## 3. Dataset Generation

The dataset is generated directly inside the C++ program.

The input range is:

\[ x `\in [-\pi,\pi]`{=tex}\]

For every randomly generated input:

\[ y = `\sin`{=tex}(x) \]

The program generates:

**50,000 training data points**

using a uniform random distribution between `-π` and `π`.

The random engine is initialized as:

``` cpp
default_random_engine re(1);
```

Using a fixed seed makes the generated dataset reproducible.

------------------------------------------------------------------------

## 4. Training Method

This project uses **batch learning**.

For every epoch:

1.  Every training sample is passed through the network.
2.  Forward propagation is performed.
3.  The error is calculated.
4.  Gradients are accumulated for all 50,000 samples.
5.  The accumulated gradients are averaged.
6.  Parameters are updated once at the end of the epoch.

Therefore, parameters are **not updated after every individual sample**.

The update is:

\[ w `\leftarrow `{=tex}w + `\eta `{=tex}`\frac{\Delta w}{N}`{=tex} \]

where:

-   (`\eta`{=tex}) = learning rate
-   \(N\) = number of training samples

------------------------------------------------------------------------

## 5. Forward Propagation

For each input (x):

### Hidden layer

\[ a_j = w_jx+b_j \]

The sigmoid activation is:

\[ `\sigma`{=tex}(a)=`\frac{1}{1+e^{-a}}`{=tex} \]

Therefore:

\[ z_j=`\sigma`{=tex}(a_j) \]

for the five hidden neurons.

### Output layer

The output neuron uses a linear activation:

\[ `\hat `{=tex}y = v_1z_1+v_2z_2+v_3z_3+v_4z_4+v_5z_5+v_0 \]

The network prediction is therefore:

\[ `\boxed{\hat y = NN(x)}`{=tex} \]

------------------------------------------------------------------------

## 6. Error and Loss

For each training example:

\[ e = y-`\hat `{=tex}y \]

The Sum of Squared Errors is:

\[ SSE=`\sum`{=tex}\_{i=1}^{N}(y_i-`\hat `{=tex}y_i)^2 \]

The Mean Squared Error is:

\[ MSE=`\frac{SSE}{N}`{=tex} \]

The final training MSE was approximately:

\[ `\boxed{0.000403}`{=tex} \]

------------------------------------------------------------------------

## 7. Backpropagation

The output-layer error is propagated back through the hidden layer.

For the hidden neurons:

\[ `\delta`{=tex}\_j = e,v_j,z_j(1-z_j) \]

The accumulated gradients are then calculated for:

### Input-to-hidden weights

\[ `\Delta `{=tex}w_j = `\delta`{=tex}\_jx \]

### Hidden biases

\[ `\Delta `{=tex}b_j = `\delta`{=tex}\_j \]

### Hidden-to-output weights

\[ `\Delta `{=tex}v_j = ez_j \]

### Output bias

\[ `\Delta `{=tex}b_0=e \]

The gradients are accumulated over the complete training dataset and
then averaged before updating the parameters.

------------------------------------------------------------------------

## 8. Training Configuration

  Parameter                                 Value
  ---------------------- ------------------------
  Function                               `sin(x)`
  Input range                           `[-π, π]`
  Training samples                         50,000
  Hidden neurons                                5
  Output neurons                                1
  Hidden activation                       Sigmoid
  Output activation                        Linear
  Learning rate                               0.1
  Maximum epochs                          100,000
  Training method          Batch gradient descent
  Final epoch reached                     100,000
  Final MSE                            \~0.000403
  Early-stop threshold                    0.00001
  Random seed                                   1

The early stopping condition was:

``` cpp
if(MSE < 0.00001){
    break;
}
```

The threshold was not reached, so training continued until the maximum
of **100,000 epochs**.

------------------------------------------------------------------------

## 9. Final Trained Parameters

### Input → Hidden Weights

``` text
w1 =  0.524800
w2 =  0.406670
w3 = -0.473325
w4 = -1.840675
w5 =  1.654361
```

### Hidden Biases

``` text
w01 =  0.944361
w02 = -0.164548
w03 =  0.813894
w04 = -1.510909
w05 = -1.111656
```

### Hidden → Output Weights

``` text
v1 = -3.943300
v2 = -4.031764
v3 =  4.857428
v4 = -3.292040
v5 =  4.360074
```

### Output Bias

``` text
v0 = 0.835396
```

------------------------------------------------------------------------

## 10. Visualization

The Python program uses the trained parameters from the C++
implementation.

The full mathematical curve is plotted using:

\[ y=`\sin`{=tex}(x) \]

over 100,000 points.

The neural network is evaluated at the same:

**100,000 test points**

and its predictions are plotted against the actual function.

![Sin(x) Approximation](sin_approximation.png)

The close overlap between the two curves demonstrates that the neural
network has successfully learned a good approximation of `sin(x)` over
the interval `[-π, π]`.

------------------------------------------------------------------------

## 11. Why Use Two Languages?

### C++

The C++ implementation is used for:

-   Neural-network mathematics
-   Dataset generation
-   Forward propagation
-   Backpropagation
-   Batch gradient descent
-   Training
-   Parameter calculation

### Python

Python is used for:

-   Loading the trained parameters
-   Running the forward pass
-   Generating dense test points
-   Visualization
-   Comparing the learned function with the actual function

The neural-network prediction in Python reproduces the same mathematical
forward pass used during C++ training.

------------------------------------------------------------------------

## 12. Project Structure

``` text
Project_4_Sin_Approximation/
│
├── cpp/
│   └── main.cpp
│
├── python/
│   └── graph.py
│
├── results/
│   └── sin_approximation.png
│
└── README.md
```

------------------------------------------------------------------------

## 13. Technologies Used

-   C++
-   Python
-   C++ STL `vector`
-   C++ `<random>`
-   C++ `<cmath>`
-   Python `math`
-   Python `matplotlib`

No machine-learning framework was used.

------------------------------------------------------------------------

## 14. Key Learning Outcomes

This project demonstrates the complete training process of a neural
network without relying on a machine-learning library.

The main concepts implemented manually are:

``` text
Dataset
   ↓
Forward Propagation
   ↓
Prediction
   ↓
Error
   ↓
Backpropagation
   ↓
Gradient Accumulation
   ↓
Batch Parameter Update
   ↓
Next Epoch
   ↓
Final Model
   ↓
Function Approximation
```

The project also demonstrates the difference between **sequential/online
learning** and **batch learning**.

In the earlier sequential version, parameters were updated after
individual examples.

In this version, gradients from the entire dataset are accumulated
first, and parameters are updated once per epoch.

------------------------------------------------------------------------

## 15. Result

The trained `1:5:1` neural network successfully learned a close
approximation of:

\[ `\boxed{y=\sin(x)}`{=tex} \]

using only:

-   5 hidden neurons
-   16 trainable parameters
-   50,000 generated training examples
-   batch gradient descent
-   sigmoid hidden neurons
-   a linear output neuron

After 100,000 epochs, the training MSE reached approximately:

\[ `\boxed{MSE \approx 0.000403}`{=tex} \]

The visualization shows that the predicted function closely follows the
actual `sin(x)` curve throughout most of the interval.

------------------------------------------------------------------------

## 16. Conclusion

This project implements a complete feed-forward neural network from
scratch for nonlinear function approximation.

The network does not know the mathematical formula for `sin(x)` during
prediction. Instead, it learns the relationship between input `x` and
target `sin(x)` from the generated training dataset.

The project provides a foundation for moving from manually implemented
small neural networks toward more advanced architectures and
machine-learning systems.
