Completed k-Nearest Neighbor, Training a support vector machine, Implement softmax classifier from assignment 1, Things remaining are 2 layer neural network and Higher level representations : image features ie Q3 & Q4

# Backpropagation and Neural Networks

## Gradient descent

$$ \frac{df(x)}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} $$

Numerical graident: slow :(, approximate :(, easy to write :)

Analytic gradient: fast :), exact :), error-prone :(

In practice: Derive analytic gradient, check your implmentation with numerical gradient

## How to compute analytic gradient using computational graphs

$$ f=Wx $$
$$ L_i = \sum_{j \neq y_i}max (0, s_j - s_{y_{i}} + 1) $$

Then we can use backpropagation to compute the gradient wrt ever variable in the graph.

## How does backpropagation work?

The first step is to take the function $f$ and represent in computational graph and then we do a forward pass. The process involves:
- Breaking down complex operations into simple location operations
- Computing intermediate values by passing inputs through the network
- Calculating the final ouput using these intermediate operations

Backward Pass:
The key to backpropagation is chain rule.
- Start with the output layer and compute loss (difference between predicted and actual output)
- Recursively compute local graidents for each operation in the network
- Propagate the gradient backward through the computation graph
- Calculate how much each weight and bias contributes to the total erorr

We can have a sigmoid function here
$$ \sigma(x) = \frac{1}{i+e^{-x}} $$

$$ \frac{d\sigma(x)}{dx} = \frac{e^{-x}}{(1 + e^{-x})^2} = (\frac{1 + e^{-x} - 1}{1 + e^{-x}}) (\frac{1}{1 + e^{-x}}) = (1 - \sigma(x))\sigma(x) $$

We can have the sigmoid function anywhere in computational graph.

Anytime we have problem computing the gradients, just think about it as a computational graph, break it dow in parts and use the chain rule.

A vectorized example: $f(x, W) = ||W.x||^2 = \sum^{n}_{i=1} (W. x)_i^2 $

where x $ \in \mathbb{R}^n $ and W $\in \mathbb{R}^{nxn} $

$ q = W.x $

$f(q) = ||q||^2 = q_1^2 + ....+ q_n^2 $

Remember, Always check: The gradient with respect to a variable should have the same shape as the variable

Modularized implmentation: Forward/Backward API

Graph (or Net) object (rough psuedo code)
```python
class ComputationalGraph(object):
    # ...
    def forward(inputs):
        # 1. [pass inputs to input gates...]
        # 2. forward the computational graph:
        for gate in self.graph.nodes_topographically_sorted():
            gate.forward()
        return loss # the final gate in the graph outputs the loss
    
    def backward():
        for gate in reversed(self.graph.nodes_topographically_sorted()):
            gate.backward() # little piece of backprop (chain rule applied)
        return inputs_gradients    
```

Modularized implmentation: forward/backward API 
```python
class MultiplyGate(object):
    def forward(object):
        z = x * y
        self.x = x
        self.y = y
        return z
    def backward(dz):
        dx = self.y * dz # [dz/dx * dL/dz]
        dy = self.x * dx # [dz/dy * dL/dz]
        return [dx, dy]
```

### Summary so far...

- neural nets will be very large: impractical to write down gradient formula by hand for all parameters
- **backpropagation** = recursive application of the chain rule along a computational grpah to compute the gradients of all inputs/parameters/intermediates
- implmentations maintain a graph structure, where the nodes implement the **forward()**/**backward()** API
- **forward**: compute result of an operation and save any intermediates needed for gradient computation in memory
- **backward**: apply the chain rule to compute the gradient of the loss function with respect to the inputs

## Neural Networks

Neural networks: without the brain stuff

(Before) Linear score function: $f = Wx$

(Now) 2-layer Neural Network: $f = W_2max(0,W_1x)$

Full implmentation of training a 2-layer neural network needs ~20 lines
```python
import numpy as np
from numpy.random import randn

# defining the neural network
X, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(D, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

# forward pass
for t in range(2000):
    h = 1 / (1 + np.exp(-x.dot(w1)))
    y_pred = h.dot(w2)
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # calculate analytic gradient
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1 - h))

    # gradient descent
    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2
```
ps: in assignment1 we have to implement 2layer neural network

