---
layout: post
title: Image Classification with Convolutional Neural Network
subtitle:
tags: [CNN, deep learning, python, tensorflow, matplotlib]
---

The purpose of the project is to develop a Convolutional Neural Network (CNN) to classify images. We will be using the TensorFlow package and following similar steps in [MIT's Deep Learning lab](https://github.com/aamini/introtodeeplearning/blob/master/lab2/Part1_MNIST.ipynb).

__Methodology__:

1. Load the data and flatten the input to feed into the model using tf.keras.layers.Flatten()
2. Compile the model using model.compile()
3. Train the model with the training data and training labels using model.fit()
4. Evaluate the model with the test dataset and print a few of the test image labels with predictions to test accuracy

__CIFAR-10 Data__:

Data set retrieved from Keras package. The CIFAR-10 data set consists of 60,000 color images in 10 classes. The data set contains 50,000 training images and 10,000 test images.

Image Classes:
* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

Relevant data set links:  
[https://keras.io/datasets/](https://keras.io/datasets/)    
[https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

__Results__:     
A model with epoch of 10 and a batch size of 64 with a test loss of 1.5843.

### 1. Loading Relevant Libraries and CIFAR-10 Data
<!--
{% highlight python linenos %}
import tensorflow as tf
import mitdeeplearning as mdl
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from keras.datasets import cifar10
{% endhighlight %} -->

<!--
testing

{% highlight python linenos %}
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
{% endhighlight %}

testing

{% endhighlight %}
np.shape(x_train)
{% endhighlight %}

**Output:**
```
(50000, 32, 32, 3)
```

There are 50,000 32x32 colored images in the training set.

Let's visualize a few of the training images with corresponding training labels below.

The training and testing labels are numeric so we will need to
pair them with a `class_names` vector. -->
