Need to complete remaining questions from assignment2

# DeepLearning Software

- CPU vs GPU
- Deep Learning Frameworks
    - Caffe/Caffe2
    - Theano/Tensorflow
    - Torch/PyTorch

## CPU vs GPU

CPU: Fewer cores, but each core is much faster and much more capable; great at sequential tasks.

GPU: More core, but each core is much slower and "dumber"; great for parallel tasks.

Programming GPUs:
- CUDA (NVIDIA Only)
    - Write C-like code that runs directly on the GPU
    - Higher-level APIs: cuBLAS, cuFFT, cuDNN, etc
- OpenCL
    - Similar to CUDA but runs on anything
    - Usually slower :(
- Udacity: Intro to Parallel Programming
    - For deep learning just use existing libraries

just use cuDNN, it will give better results compared vanilla cuda code.

## Deep Learning Frameworks

Mostly people use PyTorch or Tensorflow. People in academia use PyTorch and also the professor has bias towards PyTorch.

The point of DL Frameworks:
1. Easily build big computational graphs
2. Easily compute gradients in computational graphs
3. Run it all efficiently on GPU (wrap cuDNN, cuBLAS, etc)

**Computational Graphs** 

**Numpy:**
```python
import numpy as np

np.random.seed(0)

N, D = 3, 4

x = np.random.randn(N, D)
y = np.random.randn(N, D)
z = np.random.randn(N, D)

a = x * y
b = a + z
c = np.sum(b)

# computing gradients with c wrt x, y, z
grad_c = 1.0
grad_b = grad_c * np.ones((N, D))
grad_a = grad_b.copy()
grad_z = grad_b.copy()
grad_x = grad_a * y
grad_y = grad_a * x
```
Problems:
- Can't run on GPU
- Have to compute our own gradients

**Tensorflow:**
```python
import numpy as np
import tensoflow as tf

np.random.seed(0)

N, D = 3, 4

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32)

a = x * y
b = a + z
c = tf.placeholder(tf.float32)

grad_x, grad_y, grad_z = tf.gradients(x, y, z)

with tf.Session() as sess:
    values = {
        x : np.random.randn(N, D),
        y : np.random.randn(N, D),
        z : np.random.randn(N, D)
    }

    out = sess.run([c, grad_x, grad_y, grad_z], feed_dict=values)
    c_val, grad_x_val, grad_y_val, grad_z_val = out
```

Moving between CPU and GPU:
```python
import numpy as np
import tensoflow as tf

np.random.seed(0)

N, D = 3, 4

with tf.device('/cpu:0'): # use /gpu:0 for using GPU
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32)

    a = x * y
    b = a + z
    c = tf.placeholder(tf.float32)

grad_x, grad_y, grad_z = tf.gradients(x, y, z)

with tf.Session() as sess:
    values = {
        x : np.random.randn(N, D),
        y : np.random.randn(N, D),
        z : np.random.randn(N, D)
    }

    out = sess.run([c, grad_x, grad_y, grad_z], feed_dict=values)
    c_val, grad_x_val, grad_y_val, grad_z_val = out
```

**PyTorch:**
```python
import numpy as np
from torch.autograd import Variable

N, D = 3, 4

x = Variable(torch.randn(N, D), requires_grad=True)
y = Variable(torch.randn(N, D), requires_grad=True)
z = Variable(torch.randn(N, D), requires_grad=True)

a = x * y
b = a + z
c = torch.sum(b)

c.backward()

print(x.grad.data)
print(y.grad.data)
print(z.grad.data)
```

Moving to GPU:
```python
import numpy as np
from torch.autograd import Variable

N, D = 3, 4

x = Variable(torch.randn(N, D).cuda(), requires_grad=True)
y = Variable(torch.randn(N, D).cuda(), requires_grad=True)
z = Variable(torch.randn(N, D).cuda(), requires_grad=True)

a = x * y
b = a + z
c = torch.sum(b)

c.backward()

print(x.grad.data)
print(y.grad.data)
print(z.grad.data)
```

Then Prof Justin went through how to build and run models using Tensorflow, Keras, PyTorch, Caffe.

Everything can be found easily in the docs!
