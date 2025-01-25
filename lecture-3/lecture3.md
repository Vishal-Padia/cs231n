Assignment-1 released, need to do it!

# Loss Functions and Optimizations

## Loss Function
Loss function basically quantifies out unhappiness with the scores across the training data. Optimization is a way to efficiently finding the parameters that minimize the loss function.

A loss function tell how good our current classifier is. Given a dataset of examples

$$ {(x_i, y_i)}^{N}_{i=1} $$

Where $x_i$ is the image and $y_i$ is (integer) label

Loss over the dataset is a sum of loss over examples:

$$ L = \frac{1}{N} \sum_i{L_i (f(x_i, W), y_i)} $$

## Multiclass SVM loss

Given an example $(x_i, y_i)$ where $x_i$ is the image and $y_i$ is integer label and using shorthand for the scores vector: $ s = f(x_i, W) $

The SVM loss has the form:

$$ L_i = \sum_{j\ne y_i}max(0, s_j-s_{yi}+1) $$

```python
def L_i_vectorized(x, y, w):
    scores = W.dot(x)
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i
```

This also referred to as hinge loss (this is because when we plot it, it looks like hinge). 

Q: What is the min/max of SVM Loss?

Ans: Min is 0, Max is infinity ($ \infin $)

Eg Suppose that we found a W such that L=0, will there be other Ws?

Ans: Yes

We don't really care about the score on the training data, all we care is the score on the test data.

## Softmax Classifier (Multinomial Logistic Regression)

scores = unnormalized log probabilities of the classes.

$$ P(Y=k|X=X_i) = \frac{e^sk}{\sum_j e^sj} $$ 

where $s=f(x_i;W)$

Want to minimize the log likelihood or (for a loss function) to minimize the negative log likelihood of the correct class:
$$ L_i = -log P(Y=y_i | X=x_i) $$

In summary 
$$ L_i = -log(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}) $$

Q: What's the min and max of possible loss L_i?

Ans: Min is zero and Max is infinity ($\infin$)

Recap:
$$ L = \frac{1}{N}\sum_{i=1}^N {L_i + R(W)} $$

How do we find the `W` that minimizes the loss? That's where Optimization comes in!

## Optimization

Basically trying to find the local minima

Strategy 1: A first very bad idea solution: Random Search
```python
# assume X_train is the data where each column is an example
# assusm Y_train are the labels
# assume the function L evaluates the loss function

bestloss = float("-inf")
for num in xrange(1000):
    W = np.random.randn(10, 3073) * 0.0001 # generate random parameters
    loss = L(X_train, Y_train, W)
    if loss < bestloss:
        bestloss = loss
        bestW = W
    print(f"In attempt {num} the loss was {loss}, best {bestloss}")
```

Spoiler alert, this is really bad algorithm :)

Strategy 2: Follow the slope

Take small steps trying to go towards the minima

In 1-dimension, the derivative of a function

$$ \frac{df(x)}{dx} = \displaystyle \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} $$

In multiple dimensions, the **gradient** is the vector of (partial derivatives) along each dimension. The slope in any direction is the **dot product** of the direction with the gradient. The direction of steepset descent is the **negative gradient**.

This is super slow, so we'll use **analytic gradient**

_In practice: Always use analytic gradient, but check implementation with numerical gradient. This is called gradient check_

### Gradient Descent

```python
# Vanilla Gradient Descent
while True:
    weights_grad = evaluate_gradient(loss_fun, data, weights)
    weights += - step_size * weights_grad # perform parameter update
```

Here the step_size is hyper parameter, it's also know an learning_rate, setting the learning_rate as the first hyper parameter is a good strategy. step_size is basically once we find a point, how much step_size do we need to take inorder to reach the local minima

In practice we use Stochastic Gradient Descent (SGD),

Full sum expensive when N is large!
Approximate sum using a minibatch of examples 32/64/128 commom

```python
# vanilla Minibatch Gradient Descent

while True:
    data_batch = sample_training_data(data, 256) # sample 256 examples
    weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
    weights += - step_size * weights_grad # perform parameter update
```

