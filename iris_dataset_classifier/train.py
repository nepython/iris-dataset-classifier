import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Described path here inorder to avoid path conflicts
path = os.path.dirname(os.path.realpath(__file__))
# Modify this path inorder to pass any other file
col = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Name']
iris = pd.read_csv(f'{path}/tests/test_iris.xlsx', names=col)
iris.loc[iris['Name'] == 'Iris-virginica', 'species'] = 0
iris.loc[iris['Name'] == 'Iris-versicolor', 'species'] = 1
iris.loc[iris['Name'] == 'Iris-setosa', 'species'] = 2

# Reason for selecting only these two species explained in README
iris = iris[iris['species'] != 2]

# These two were the most distinct features hence I have selected them
training_inputs = iris[['petal_length', 'petal_width']].values.T
output = iris[['species']].values.T
output = output.astype('uint8')


def plot_input_data():
    iris = pd.read_csv(f'{path}/iris.csv')
    iris.loc[iris['Name'] == 'Iris-virginica', 'species'] = 0
    iris.loc[iris['Name'] == 'Iris-versicolor', 'species'] = 1
    iris.loc[iris['Name'] == 'Iris-setosa', 'species'] = 2
    training_inputs = iris[['petal_length', 'petal_width']].values.T
    output = iris[['species']].values.T
    output = output.astype('uint8')
    plt.scatter(training_inputs[0, :], training_inputs[1, :], c=output[0, :], s=40, cmap=plt.cm.Spectral)
    plt.title('IRIS DATA | Blue - Versicolor, Red - Virginica ')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.show()


class NeuralNetwork():
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def initialize_parameters(self, input_size, layer_size, output_size):
        # Selected it randomly though it does affects the prediction slightly
        np.random.seed(2)
        input_weight = np.random.randn(layer_size, input_size) * 0.01
        input_bias = np.zeros(shape=(layer_size, 1))
        output_weight = np.random.randn(output_size, layer_size) * 0.01
        output_bias = np.zeros(shape=(output_size, 1))
        parameters = {'input_weight': input_weight,
                      'input_bias': input_bias,
                      'output_weight': output_weight,
                      'output_bias': output_bias}
        return parameters

    def layer_sizes(self, training_inputs, output):
        input_size = training_inputs.shape[0]
        layer_size = 6
        output_size = output.shape[0]
        return (input_size, layer_size, output_size)

    # Here, `tanh` has been used as the first activator followed by sigmoid
    # TODO: Migrate to ReLU if feasible
    def forward_propagation(self, training_inputs, parameters):
        input_weight = parameters['input_weight']
        input_bias = parameters['input_bias']
        output_weight = parameters['output_weight']
        output_bias = parameters['output_bias']
        # Implement Forward Propagation to calculate A2 (probability)
        Z1 = np.dot(input_weight, training_inputs) + input_bias
        A1 = np.tanh(Z1)
        Z2 = np.dot(output_weight, A1) + output_bias
        A2 = self.sigmoid(Z2)
        cache = {'Z1': Z1,
                 'A1': A1,
                 'Z2': Z2,
                 'A2': A2}
        return A2, cache

    def compute_cost(self, A2, output, parameters):
        output_size = output.shape[1]
        # TODO: Understand it, didn't understand this much right now :(
        logprobs = np.multiply(np.log(A2), output) + np.multiply((1 - output), np.log(1 - A2))
        cost = - np.sum(logprobs) / output_size
        return cost

    def backward_propagation(self, parameters, cache, training_inputs, output):
        input_size = training_inputs.shape[1]
        output_weight = parameters['output_weight']
        A1 = cache['A1']
        A2 = cache['A2']
        # Backward propagation: calculate d(input_weight), d(input_bias), d(output_weight), d(output_bias)
        dZ2 = A2 - output
        doutput_weight = (1 / input_size) * np.dot(dZ2, A1.T)
        # REMEMBER: `keepdims` returns sum in `array` form
        doutput_bias = (1 / input_size) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(output_weight.T, dZ2), 1 - np.power(A1, 2))
        dinput_weight = (1 / input_size) * np.dot(dZ1, training_inputs.T)
        dinput_bias = (1 / input_size) * np.sum(dZ1, axis=1, keepdims=True)
        grads = {'dinput_weight': dinput_weight,
                 'dinput_bias': dinput_bias,
                 'doutput_weight': doutput_weight,
                 'doutput_bias': doutput_bias}
        return grads

    def update_parameters(self, parameters, grads, learning_rate=1.2):
        input_weight = parameters['input_weight']
        input_bias = parameters['input_bias']
        output_weight = parameters['output_weight']
        output_bias = parameters['output_bias']
        dinput_weight = grads['dinput_weight']
        dinput_bias = grads['dinput_bias']
        doutput_weight = grads['doutput_weight']
        doutput_bias = grads['doutput_bias']
        # Update rule for each parameter
        input_weight = input_weight - learning_rate * dinput_weight
        input_bias = input_bias - learning_rate * dinput_bias
        output_weight = output_weight - learning_rate * doutput_weight
        output_bias = output_bias - learning_rate * doutput_bias
        parameters = {'input_weight': input_weight,
                      'input_bias': input_bias,
                      'output_weight': output_weight,
                      'output_bias': output_bias}
        return parameters

    def nn_model(self, training_inputs, output, layer_size, num_iterations=10000, print_cost=False):
        np.random.seed(3)
        input_size = self.layer_sizes(training_inputs, output)[0]
        output_size = self.layer_sizes(training_inputs, output)[2]
        parameters = self.initialize_parameters(input_size, layer_size, output_size)

        for i in range(0, num_iterations):
            # Forward propagation. Inputs: 'training_inputs, parameters'. Outputs: 'A2, cache'.
            A2, cache = self.forward_propagation(training_inputs, parameters)
            # Cost function. Inputs: 'A2, output, parameters'. Outputs: 'cost'.
            cost = self.compute_cost(A2, output, parameters)
            if i is 0 and not print_cost:
                print(f'Before training:\n  cost = {cost}')
            # Backpropagation. Inputs: 'parameters, cache, training_inputs, output'. Outputs: 'grads'.
            grads = self.backward_propagation(parameters, cache, training_inputs, output)
            # Gradient descent parameter update. Inputs: 'parameters, grads'. Outputs: 'parameters'.
            parameters = self.update_parameters(parameters, grads)
            # Prints improvement every 1000 iterations
            if print_cost and i % 1000 is 0:
                print(f'Cost after iteration {i}: {cost}')
        print(f'After training:\n   cost = {cost}')
        return parameters, layer_size

    def plot_decision_boundary(self, model, training_inputs, output):
        # Set min and max values and gave it some padding
        x_min, x_max = training_inputs[0, :].min() - 0.20, training_inputs[0, :].max() + 0.20
        y_min, y_max = training_inputs[1, :].min() - 0.20, training_inputs[1, :].max() + 0.20
        h = 0.01
        # Generate a grid of points with distance h between them
        x_co_ordinates, y_co_ordinates = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        Z = model(np.c_[x_co_ordinates.ravel(), y_co_ordinates.ravel()])
        Z = Z.reshape(x_co_ordinates.shape)
        # Plot the contour and training examples
        plt.contourf(x_co_ordinates, y_co_ordinates, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(training_inputs[0, :], training_inputs[1, :], c=output)
        plt.title('Boundary created separating Versicolor from Virginica')
        plt.xlabel('Petal Length')
        plt.ylabel('Petal Width')
        plt.show()

    def predict(self, parameters, training_inputs):
        A2, cache = nn.forward_propagation(training_inputs, parameters)
        predictions = np.round(A2)
        return predictions


if __name__ == '__main__':
    nn = NeuralNetwork()
    # Pass `print_cost=True` if you want to see improvements every 1000 trials
    parameters, layer_size = nn.nn_model(training_inputs, output, layer_size=6, num_iterations=10000)
    nn.plot_decision_boundary(lambda x: nn.predict(parameters, x.T), training_inputs, output[0, :])
