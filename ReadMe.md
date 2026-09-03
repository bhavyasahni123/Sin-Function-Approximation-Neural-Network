# Sin(x) Function Approximation Using Neural Network

A neural network implemented **from scratch in C++** to approximate the mathematical function:

$$
y = \sin(x)
$$

The project uses a **single hidden-layer feedforward neural network** trained using **batch gradient descent and backpropagation**. No machine-learning frameworks are used.

---

## Neural Network Architecture

* **Input layer:** 1 neuron
* **Hidden layer:** 5 neurons
* **Output layer:** 1 neuron
* **Architecture:** 1 : 5 : 1
* **Hidden activation:** Sigmoid
* **Output activation:** Linear
* **Loss:** Mean Squared Error (MSE)
* **Optimization:** Batch Gradient Descent
* **Training method:** Backpropagation

---

## Dataset

The dataset is generated directly inside the C++ program.

For every randomly generated input:

$$
x \in [-\pi,\pi]
$$

the corresponding target is calculated as:

$$
y = \sin(x)
$$

### Dataset Size

**50,000 training samples**

The random number generator is initialized with:

```cpp
default_random_engine re(1);
```

Using a fixed seed makes the generated dataset reproducible.

---

## Forward Propagation

For each input \(x\), the five hidden neurons calculate:

$$
a_1=w_1x+b_1
$$

$$
a_2=w_2x+b_2
$$

$$
a_3=w_3x+b_3
$$

$$
a_4=w_4x+b_4
$$

$$
a_5=w_5x+b_5
$$

The sigmoid activation function is:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

Therefore:

$$
z_j=\sigma(a_j)
$$

The output neuron uses a linear activation:

$$
\hat{y}
=
v_1z_1+
v_2z_2+
v_3z_3+
v_4z_4+
v_5z_5+
v_0
$$

---

## Error Calculation

For each training sample:

$$
e=y-\hat{y}
$$

The Sum Squared Error is:

$$
SSE=\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

and the Mean Squared Error is:

$$
MSE=\frac{SSE}{N}
$$

Training was configured to stop early if:

$$
MSE < 0.00001
$$

In the final run, the network did not reach this threshold and completed the full **100,000 epochs**, achieving a final MSE of approximately:

```text
0.000403
```

---

## Training

The network uses **batch learning**.

During each epoch:

1. All 50,000 training samples are processed.
2. Forward propagation is performed.
3. The error for every sample is calculated.
4. Gradients are accumulated.
5. The parameters are updated once after the complete dataset has been processed.

The learning rate used was:

```text
0.1
```

Unlike sequential/online learning, the weights are **not updated after every individual sample**.

---

## Backpropagation

For the output layer:

$$
\Delta v_j=e z_j
$$

$$
\Delta v_0=e
$$

For the hidden layer:

$$
\delta_j=e v_j z_j(1-z_j)
$$

The corresponding gradients are:

$$
\Delta w_j=\delta_jx
$$

$$
\Delta b_j=\delta_j
$$

After processing the complete dataset, the parameters are updated using:

$$
w_j \leftarrow w_j+\eta\frac{\Delta w_j}{N}
$$

where:

* \(\eta\) = learning rate
* \(N\) = number of training samples

---

## Final Training Results

| Parameter                |          Value |
| ------------------------ | -------------: |
| Function                 |  \(y=\sin(x)\) |
| Input range              | \([-\pi,\pi]\) |
| Training samples         |         50,000 |
| Hidden neurons           |              5 |
| Architecture             |      1 : 5 : 1 |
| Hidden activation        |        Sigmoid |
| Output activation        |         Linear |
| Learning rate            |            0.1 |
| Maximum epochs           |        100,000 |
| Actual epochs            |        100,000 |
| Early stopping threshold |  MSE < 0.00001 |
| Final MSE                |      ~0.000403 |

---

## Final Weights

### Input → Hidden Layer

```text
w1 =  0.524800
w2 =  0.406670
w3 = -0.473325
w4 = -1.840675
w5 =  1.654361
```

### Hidden Layer Biases

```text
w01 =  0.944361
w02 = -0.164548
w03 =  0.813894
w04 = -1.510909
w05 = -1.111656
```

### Hidden → Output Layer

```text
v1 = -3.943300
v2 = -4.031764
v3 =  4.857428
v4 = -3.292040
v5 =  4.360074
```

### Output Bias

```text
v0 = 0.835396
```

---

## Visualization

The trained network was implemented in Python using the final learned weights.

The graph contains:

* **Blue curve:** Actual \(\sin(x)\)
* **Red curve:** Neural network prediction

The prediction was evaluated at **100,000 test points** uniformly distributed across:

$$
[-\pi,\pi]
$$

### Result

![Sin(x) Function Approximation](sinx_graph.jpg)

The graph shows the neural network prediction following the target sine curve across the specified range.

---

## Test Data

For visualization, **100,000 equally spaced test points** were generated from:

$$
-\pi \rightarrow \pi
$$

For each point:

```text
Actual    = sin(x)
Predicted = NeuralNetwork(x)
```

The predicted values were then plotted against the original sine curve.

---

## Technologies Used

* **C++**
* **Python**
* **Mathematics**
* **Artificial Neural Networks**
* **Backpropagation**
* **Batch Gradient Descent**
* **Mean Squared Error**
* **Matplotlib**
