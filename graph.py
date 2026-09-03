import math
import matplotlib.pyplot as plt

def sigmoid (x):
    return 1/(1+math.exp(-x))

w1 = 0.524800
w2 = 0.406670
w3 = -0.473325
w4 = -1.840675
w5 = 1.654361

w01 = 0.944361
w02 = -0.164548
w03 = 0.813894
w04 = -1.510909
w05 = -1.111656

v1 = -3.943300
v2 = -4.031764
v3 = 4.857428
v4 = -3.292040
v5 = 4.360074
v0 = 0.835396

def predict(x):

    a1 = w1 * x + w01
    a2 = w2 * x + w02
    a3 = w3 * x + w03
    a4 = w4 * x + w04
    a5 = w5 * x + w05

    z1 = sigmoid(a1)
    z2 = sigmoid(a2)
    z3 = sigmoid(a3)
    z4 = sigmoid(a4)
    z5 = sigmoid(a5)

    y_hat = (v1 * z1 + v2 * z2 + v3 * z3 + v4 * z4 + v5 * z5 + v0)

    return y_hat

x_curve= []
y_curve= []
y_predicted = []
x_test = []

for i in range(100000):
    x=-math.pi + (2*math.pi*i)/99999
    x_curve.append(x)
    y_curve.append(math.sin(x))

for i in range(100000):
    x=-math.pi + (2*math.pi*i)/99999
    x_test.append(x)
    y_predicted.append(predict(x))


plt.plot(x_curve, y_curve , label='sin(x)', color='blue')
plt.plot(x_test, y_predicted, label='Predicted', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function Approximation of Sin(x) Using Neural Network')
plt.legend()
plt.show()