import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Described path here inorder to avoid path conflicts
col = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Name']
path = os.path.dirname(os.path.realpath(__file__))
# Modify this path to pass any other file
iris = pd.read_csv(f'{path}/tests/test_iris.xlsx', names=col)


class DatasetClassifier():
    # All three species
    iris_setosa = iris.loc[iris['Name'] == 'Iris-setosa']
    iris_virginica = iris.loc[iris['Name'] == 'Iris-virginica']
    iris_versicolor = iris.loc[iris['Name'] == 'Iris-versicolor']

    # Displays plots comparing the 3 species across all 4 features
    # Not required for the task
    def compare_all_species_per_feature(self):
        plt.style.use('ggplot')
        features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
        for feature in features:
            f = plt.figure(feature)
            plt.hist(iris[feature].to_numpy()[:50], bins=10, color='black', label='setosa')
            plt.hist(iris[feature].to_numpy()[50:100], bins=10, color='darkblue', label='versicolor')
            plt.hist(iris[feature].to_numpy()[100:150], bins=10, color='green', label='virginica')
            plt.legend()
            f.show()
        plt.show()

    def plot_iris_setosa(self):
        counts, bin_edges = np.histogram(self.iris_setosa['petal_length'], bins=10, density=True)
        pdf = counts / (sum(counts))
        print(pdf)
        print(bin_edges)
        cdf = np.cumsum(pdf)
        plt.plot(bin_edges[1:], pdf)
        plt.plot(bin_edges[1:], cdf)
        plt.show()

    def plot_iris_virginica(self):
        counts, bin_edges = np.histogram(self.iris_virginica['petal_length'],
                                         bins=10, density=True)
        pdf = counts / (sum(counts))
        print(pdf)
        print(bin_edges)
        cdf = np.cumsum(pdf)
        plt.plot(bin_edges[1:], pdf)
        plt.plot(bin_edges[1:], cdf)
        plt.show()

    def plot_iris_versicolor(self):
        counts, bin_edges = np.histogram(self.iris_versicolor['petal_length'],
                                         bins=10, density=True)
        pdf = counts / (sum(counts))
        print(pdf)
        print(bin_edges)
        cdf = np.cumsum(pdf)
        plt.plot(bin_edges[1:], pdf)
        plt.plot(bin_edges[1:], cdf)
        plt.show()
