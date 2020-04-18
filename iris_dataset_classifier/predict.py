# NOTE: This is still a WIP and incomplete

import os

import numpy as np
import pandas as pd

# Described path here inorder to avoid path conflicts
path = os.path.dirname(os.path.realpath(__file__))
# Modify this path to pass any other file
col = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
iris = pd.read_csv(f'{path}/tests/test_iris.xlsx', names=col)


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((4, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Do not forget this gave me some good trouble :(
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def rectified(self, x):
        print(x)
        return x

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            outputs = self.neuron(training_inputs)
            error = training_outputs - outputs
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(outputs))
            self.synaptic_weights = np.add(self.synaptic_weights, adjustments)

    def neuron(self, inputs):
        inputs = inputs.astype(float)
        outputs = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return outputs


if __name__ == '__main__':
    neural_network = NeuralNetwork()
    print('Random synaptic weights:')
    print(neural_network.synaptic_weights)
    training_inputs = iris.drop('type', axis=1).to_numpy()
    # TODO substitue this temp_array
    temp_array = [0.2 for i in range(50)] + [0.5 for j in range(50)] + [1 for n in range(50)]
    training_outputs = np.array([temp_array]).T
    # Taking mean
    # mean = [5.84, 3.05, 3.76, 1.20]
    # mean_array = np.array([mean for i in range(50)])
    neural_network.train(training_inputs, training_outputs, 10)
    print('Synaptic weights after training')
    print(neural_network.synaptic_weights)

    print('Training complete now testing Begins :)')
    sepal_length = float(input('sepal length: '))
    sepal_width = float(input('sepal width: '))
    petal_length = float(input('petal length: '))
    petal_width = float(input('petal width: '))
    print('Entered data:', sepal_length, sepal_width, petal_length, petal_width)
    type = neural_network.neuron(np.array([sepal_length, sepal_width, petal_length, petal_width]))
    print(type)

    # TODO : check the variation about mean
