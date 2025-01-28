Submitted Assignment 1 before I watched this lecture, less goo

# Convolution Neural Networks

### A bit of history

The **Mark I Perceptron** machine was the first implementation of the perceptron algorithm. 

The machien was connected to a camera that used 20x20 cadium sulfide photocells to produce a 400-pixel image.

$$ 
f(x) =
\begin{cases}
1 & \text{if } w \cdot x + b > 0 \\
0 & \text{otherwise}
\end{cases} 
$$

update rule: $ w_i(t+1) = w_i(t) + \alpha(d_j - y_j(t))x_{j,i} $

---

Fully connected layer:

32x32x3 image -> strectch to 3072x1

### Convolution layer

32x32x3 image -> preserve spatial strucutre (32 height, 32 width, 3 depth)

our weights are going to be small filters like 5x5x3 filter, and we are going to **convolve** the filter with the image ie "slide over the image spatially computing the dot products" ie 5 * 5 * 3 = 75 - dimensional dot product + bias

When we are working with CNNs we would want multiple filters.

We can have as many filters as we want to have, for example, if we had 6 5x5 filters, we'll get 6 seperate activation maps. We stack these up to get a "new image" of size 28x28x6!

Once we have more layers in our ConvNet, the earlier layers usually represent low-level features, in the mid layers it represents more complex kind of features and then at higher level layers it represents things that start to more resembles blobs and all. 

We are going to have an image and pass it to our network (conv, relu, conv, relu, pool, conv, relu, conv, relu, pool, conv, relu, conv, relu, pool, Fully Connected layer) and use it get the final score/output.

Q: Why do we do zero padding?

Because we want to maintain the same input sizes as before

Example time:

Input volume: 32x32x3

10 5x5 filters with stride 1, padding 2

Output volume size? 

Ans: (32(ie input) + 2 * 2(ie padding) - 5(ie filter)) / 1(ie stride) + 1 = 32 spatially, so 32x32x10

Number of parameters in this layer? 

Ans: each filter has  5 * 5 * 3 + 1 = 76 params => 76 * 10 = 760 

Summary: To summarize the Conv layer:
- Accepts a volume of size $W_1$ x $H_1$ x $D_1$
- Requires four hyperparameters:
    - Number of filters $K$,
    - their spatial extent $F$,
    - the stride $S$,
    - the amount of zero padding $P$.
- Produces a volume of size $W_2$ x $H_2$ x $D_2$ where:
    - $W_2$ = $ (W_1 - F + 2P)/S + 1 $
    - $H_2$ = $ (H_1 - F + 2P)/S + 1 $ (ie width and height are computed equally by symmetry)
    - $D_2$ = $K$
- With parameter sharing, it introduces $F \cdot F \cdot D_1$ weights per filter for a total of $(F \cdot F \cdot D_1) \cdot K$ weights and $K$ biases.
- In the output volume, the $d$-th depth slice (of size $W_2$ x $H_2$) is the result of performing a valid convolution of the $d$-th filter over the input volume with a stride of $S$, and then offset by $d$-th bias.

Common settings:

K = (powers of 2 eg 32, 64, 128, 512)
 - F = 3, S = 1, P = 1
 - F = 5, S = 1, P = 2
 - F = 5, S = 2, P = ? (whatever fits)
 - F = 1, S = 1, P = 0

Pooling Layer:
- makes the representations smaller and more manageable
- operates over each activation map independently

Common way to do this is Max Pooling.

- Accepts a volume of size $W_1$ x $H_1$ x $D_1$
- Requires three hyperparameters:
    - their spatial extend $F$
    - the stride $S$
- Produces a volume of size $W_2$ x $H_2$ x $D_2$ where:
    - $W_2$ = $ W_1 - F)/ S + 1 $
    - $H_2$ = $ (H_1 - F)/S + 1 $
    - $D_2$ = $D_1$
- Introduces zero parameters since it computes a fixed function of the input
- Note that it is not common to use zero-padding for Pooling layers

Common Setting for pooling layer: 

F = 2, S = 2

F = 3, S = 2

Summary:

- ConvNets stack CONV, POOL, FC layers
- Trends towards smaller filters and deeper architectures
- Trends towards getting rid of POOL/FC layers (just CONV)
- Typical architectures look like:
   - [(CONV-RELU) * N-POOL?] * M - (FC - RELU) * K, SOFTMAX
   - where N is usually upto ~5, M is large, 0 <= k <= 2
   - but recent advances such as ResNet/GoogLeNet challenge this paradigm


