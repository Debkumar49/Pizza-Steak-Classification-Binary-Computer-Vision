<h1>Pizza Steak Binary Classification(Computer Vision)</h1>
Binary classification using CNN

<h2> What is Neural Network ?</h2>
 It’s a technique for building a computer program that learns from data. It is based very loosely on how we think the human brain works. First, a collection of software “neurons” are created and connected together, allowing them to send messages to each other. Next, the network is asked to solve a problem, which it attempts to do over and over, each time strengthening the connections that lead to success and diminishing those that lead to failure.

<h2> What is a Convolutional Neural Network?</h2>
In machine learning, a classifier assigns a class label to a data point. For example, an image classifier produces a class label (e.g, bird, plane) for what objects exist within an image. A convolutional neural network, or CNN for short, is a type of classifier, which excels at solving this problem!

A CNN is a neural network: an algorithm used to recognize patterns in data. Neural Networks in general are composed of a collection of neurons that are organized in layers, each with their own learnable weights and biases. Let’s break down a CNN into its basic building blocks.

✔ A <b>tensor</b> can be thought of as an n-dimensional matrix. In the CNN above, tensors will be 3-dimensional with the exception of the output layer.<br>
✔ A <b>neuron</b> can be thought of as a function that takes in multiple inputs and yields a single output.<br>
✔ A <b>layer</b> is simply a collection of neurons with the same operation, including the same hyperparameters.<br>
✔ <b>Kernel weights and biases</b>, while unique to each neuron, are tuned during the training phase, and allow the classifier to adapt to the problem and dataset provided.<br>
✔ A CNN conveys a differentiable score function, which is represented as class scores in the visualization on the output layer.<br>

<h2>What does each layer of the network do?</h2>
  <h3>Input Layer</h3>
  The input layer (leftmost layer) represents the input image into the CNN. Because we use RGB images as input, the input layer has three channels, corresponding to the    red, green, and blue channels, respectively, which are shown in this layer.
  <h2>Convolutional Layers.</h3>
  The convolutional layers are the foundation of CNN, as they contain the learned kernels (weights), which extract features that distinguish different images from one      another—this is what we want for classification! As you interact with the convolutional layer, you will notice links between the previous layers and the convolutional    layers. Each link represents a unique kernel, which is used for the convolution operation to produce the current convolutional neuron’s output or activation map.
  The convolutional neuron performs an elementwise dot product with a unique kernel and the output of the previous layer’s corresponding neuron. This will yield as many intermediate results as there are unique kernels. The convolutional neuron is the result of all of the intermediate results summed together with the learned bias.For example, let’s look at the first convolutional layer in the Tiny VGG architecture above. Notice that there are 10 neurons in this layer, but only 3 neurons in the previous layer. In the Tiny VGG architecture, convolutional layers are fully-connected, meaning each neuron is connected to every other neuron in the previous layer. Focusing on the output of the topmost convolutional neuron from the first convolutional layer, we see that there are 3 unique kernels.
  
  
  
<img src="https://user-images.githubusercontent.com/47305904/191323592-dc2f6cd9-b0f8-4700-ab96-828011d5e7d1.gif" width = "600" height="500" align="center">

The size of these kernels is a hyper-parameter specified by the designers of the network architecture. In order to produce the output of the convolutional neuron (activation map), we must perform an elementwise dot product with the output of the previous layer and the unique kernel learned by the network. In TinyVGG, the dot product operation uses a stride of 1, which means that the kernel is shifted over 1 pixel per dot product, but this is a hyperparameter that the network architecture designer can adjust to better fit their dataset. We must do this for all 3 kernels, which will yield 3 intermediate results.

<img src="https://user-images.githubusercontent.com/47305904/191324397-59ad7609-fa3a-4ff2-a4a4-0b9a8fbf73d7.gif" style="width:400px;height:300px;">

Then, an elementwise sum is performed containing all 3 intermediate results along with the bias the network has learned. After this, the resulting 2-dimensional tensor will be the activation map viewable on the interface above for the topmost neuron in the first convolutional layer. This same operation must be applied to produce each neuron’s activation map.

With some simple math, we are able to deduce that there are 3 x 10 = 30 unique kernels, each of size 3x3, applied in the first convolutional layer. The connectivity between the convolutional layer and the previous layer is a design decision when building a network architecture, which will affect the number of kernels per convolutional layer. Click around the visualization to better understand the operations behind the convolutional layer. See if you can follow the example above!

<h2>Activation Functions</h2>
Neural networks are extremely prevalent in modern technology—because they are so accurate! The highest performing CNNs today consist of an absurd amount of layers, which are able to learn more and more features. Part of the reason these groundbreaking CNNs are able to achieve such tremendous accuracies is because of their non-linearity. ReLU applies much-needed non-linearity into the model. Non-linearity is necessary to produce non-linear decision boundaries, so that the output cannot be written as a linear combination of the inputs. If a non-linear activation function was not present, deep CNN architectures would devolve into a single, equivalent convolutional layer, which would not perform nearly as well. The ReLU activation function is specifically used as a non-linear activation function, as opposed to other non-linear functions such as Sigmoid because it has been empirically observed that CNNs using ReLU are faster to train than their counterparts.

The ReLU activation function is a one-to-one mathematical operation:
<img src="https://user-images.githubusercontent.com/47305904/191326211-60ced00c-e98b-4155-8679-349b48d62647.png" style="width:400px;height:300px;">

This activation function is applied elementwise on every value from the input tensor. For example, if applied ReLU on the value 2.24, the result would be 2.24, since 2.24 is larger than 0.The Rectified Linear Activation function (ReLU) is performed after every convolutional layer in the network architecture outlined above. Notice the impact this layer has on the activation map of various neurons throughout the network!

<h2>Pooling Layers</h2>
There are many types of pooling layers in different CNN architectures, but they all have the purpose of gradually decreasing the spatial extent of the network, which reduces the parameters and overall computation of the network. The type of pooling used in the Tiny VGG architecture above is Max-Pooling.The Max-Pooling operation requires selecting a kernel size and a stride length during architecture design. Once selected, the operation slides the kernel with the specified stride over the input while only selecting the largest value at each kernel slice from the input to yield a value for the output.

<h2>Flatten Layer</h2>
This layer converts a three-dimensional layer in the network into a one-dimensional vector to fit the input of a fully-connected layer for classification. For example, a 5x5x2 tensor would be converted into a vector of size 50.

