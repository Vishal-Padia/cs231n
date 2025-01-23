All the assignments will be done in Numpy, I think this is crazy good becasue rather than using high level library (like: scikit learn) I can understand the core concepts of the machine learning algorithms.

## Image Classification

System will receive input image, the system is trained on same labels cat, dog, etc and then system will classify the input image.

The system doesn't perceive it as "a cat image" the computer perceives it as a big grid of numbers between [0, 255], eg. 8000 x 600 x 3 (3 channels of RGB)

As and when the camera moves the pixels changes, the algorithm should be robust to camera angles, illumination, deformation, occlusion, background clutter, intraclass variation (shapes, size, colors, etc). There's no obivious way to hard-code the algorithm for recognizing a cat or other classes. 

Attempts have been made to recognize cats: Like Finding edges, corners like a specific pattern, but writing specifc algorithms for recognizing the dogs, trucks and so on would be difficult task as we would have to find patterns in each image and specific hard-coded algo is a tedious and time consuming task.

That's why we have taken a **data-driven approach**: Collect dataset of images and labels, Use ML to train a classifier and Evaluate the classifier on new images. 

```python
def train(images, labels):
    # insert ML magic here
    return model

def predict(model, images):
    # insert prediction logic here
    return prediction
```

## First Classifier: Nearest Neighbor

In the train function we just memorize all the data and labels. And in the predict function we predict the labels of the most similar training image

```python
def train(images, labels):
    # inset ML magic here
    return model

def predit(model, test_images):
    # Use model to predict labels
    return test_labels
```

Using CIFAR10 dataset, we can use KNN to classify images! But it won't work very well :) We use L1 distance to compare images

L1 Distance: $$d_1(I_1, I_2) = \sum_{p} | I^{P}_1 - I^{P}_2 | $$

This gives a concrete way to calculate the distance between images

```python
import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass
    
    def train(self, X, y):
        """ X is NxD where each row is an example. Y is 1-dimension of size N """

        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.Ytr = y
    
    def predict(self, X):
        """ X is NxD where each row is an exmaple we wish to predict label for """
        num_test = X.shape[0]

        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = selef.ytr.dtype)

        # loop over all the test rows
        for i in range(num_test):
            # find the nearest training image to the i^th test images
            # using the L1 distance (sum of absolute value difference)
            distances = np.sum(np.abs(self.Xtr - X[i, :], axis=1))
            min_index = np.argmin(distances) # get the index with the smallest distance
            Ypred[i] = self.ytr[min_index] # predict the label of the nearset sample
        
        return Ypred
```

Q: With N examples how fast are training and prediction?

A: Train O(1), predict O(N)

This is bad: we want classifiers that are fast at prediction; slow for training is ok!

We can also use L2 (Euclidean Distance)

$$ d_2(I_1, I_2) = \sqrt{\sum_p (I^{P}_1 - I^{P}_2)^2 } $$

Different distance metric makes different assumption about the underlying geometry/topolgy

L1 distance depends on your choice of co-ordinate system, if you were to rotate the co-ordinate system, the L1 distances changes too

Changing the co-ordinate frame in L2 doesn't matter, the distance would be the same.

### Hyperparameters
What is the best value of k to use? What is the best distance to use?

These are hyperparameters: choices baout the algorithm that we set rather tha learn, it's also very problem dependent, start with a random number and see how it goes.

Setting Hyperparameters:

Idea 1: Choose hyperparameters that work best on the data. BAD: K=1 always works perfectly on training data

Idea 2: Split data into train and test, choose hyperparameters that work best on test data. BAD: No idea how algorithm will perform on new data

Idea 3: Split data into train, val and test, choose hyperparameters on val and evaluate on test. BETTER!

Idea 4: Cross validation: Split data into folds try each fold as validation and average the results. **Useful for small datasets, but not used too frequently in deep learning!**

k-Nearest Neighbor on images are never used
- Ver slow at test time
- Distance metrics on pixels are not informatiive
- Curse of dimensionality

## Summary
In image classification we start with a training set of images and labels and must predict labels on the test set. 

The K-Nearest Neighbors classifier predicts labels based on nearest training exmaples. 

Distance metric and K are hyperparameters.

Choose hyperparameters using the validation set; only run on the testset once at the very end! 

## Linear Classification

Neural Networks are like Lego-Blocks, building a large structure using small blocks. 

Parametric Approach

$$ f(x, W) = Wx + b $$

Image
Array of 32x32x3 numbers -> f(x, W) -> 10 numbers giving class scores

(3072 numbers total)        W: parameters 
                            or weights

We added bias to balance things out!

So the problem is that the linear classifer is only learning one template per each class

