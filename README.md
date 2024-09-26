# Digit Recongnition in C++
I created a neural network from scratch, without using any libraries, to recognize handwritten digits.  
Here is a simple description of how it works.
![alt text](https://github.com/vittoriosironi/Digit_Recognition_ML-Neural_Network/blob/main/Image/5.png?raw=true)

## 🚀 How does the code work?
The main file is `ml_dig.cpp`. This file contains the code for initializing the neural network, reading the data, and training the network.  
In the first part of the main function, the code initializes a network with two layers, consisting of 30 and 10 neurons, respectively. The digits are then read from the file `digits_dataSet.txt`, with 80% of the data allocated for training and the remaining 20% for testing. To reduce the data size, convolution is applied: the code selects small sections of the image and averages the data within each section. The size of these sections can be adjusted by modifying the `KRES` variable.  
Next, the code loops `REP` times to train the model. Initially, the predictions will be incorrect, but they improve after a few cycles. During training, the `evo` function (explained later) uses backpropagation on all training samples. Once a training cycle is completed, the `test` function is called, selecting random digits and displaying the results on the terminal. This loop repeats `REP` times.  
After the training phase is completed, the model is ready, and the last part of the code performs the actual digit recognition.

## 🥫 How is the neural network implemented?
The secret sauce resides in the `ML.h` file, which contains the classes for the network.  
There are three main classes:
 - `Neuron`
 - `NeuralLayer`
 - `NeuralNetwork`

### Neuron
This class has four attributes:
- `N` (number of connections of the neuron)
- `activationFunction`
- `weight`
- `bias`

Each neuron contains a vector of connection weights and a bias. The user can specify the number of connections and the activation function for the neuron. The main constructor initializes the weights and bias with random values between -0.5 and 0.5, using the formula: $$-0.5 + \frac{rand()}{RANDMAX}$$.  
Then, the activation function is applied to calculate the weighted sum of the neuron's inputs.

### NeuralLayer
This class represents a single layer in the network.  
It has three attributes:
- `NL` (number of neurons in the layer)
- `N`
- `neuron`
  
There are two primary methods used to evaluate the connections of each neuron in the layer, calling the activation function of each neuron.

### Neural Network
This class represents the entire neural network.  
It has two attributes:
- `Inputs` (size of the input data)
- `NeuronsPerLayer`
  
The constructor initializes the network using the classes described above. It also contains methods to evaluate all the values of the model by leveraging the `NeuralLayer` class methods.  
However, the core of the implementation (🥫) lies in the `evo` function, where all the neural network mathematics (in particulary the one used for backpropagation) is handled. This function should deserves a detailed explanation by itself. I recommend to give it a look. Additionally, there are functions for calculating the cost function and printing the neural network's parameters to a file.

## 💻 How can you use it?
You can compile it as a regular C++ file using:
```
g++ ml_dig.cpp
```
Then execute it:
```
./a.out
```
Initially, the digit recognition will be inaccurate. However, after training, it will improve over time.










