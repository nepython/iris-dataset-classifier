import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Described path here inorder to avoid path conflicts
col = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
path = os.path.dirname(os.path.realpath(__file__))
# Modify this path to pass any other file
iris = pd.read_csv(f"{path}/tests/test_iris.xlsx", names=col)


class DatasetClassifier():
    # All three species
    iris_setosa = iris.loc[iris["type"] == "Iris-setosa"]
    iris_virginica = iris.loc[iris["type"] == "Iris-virginica"]
    iris_versicolor = iris.loc[iris["type"] == "Iris-versicolor"]

    # Displays plots comparing the 3 species across all 4 features
    def compare_all_species_per_feature(self):
        sns.FacetGrid(iris, hue="type", size=3).map(sns.distplot, "petal_length").add_legend()
        sns.FacetGrid(iris, hue="type", size=3).map(sns.distplot, "petal_width").add_legend()
        sns.FacetGrid(iris, hue="type", size=3).map(sns.distplot, "sepal_length").add_legend()
        sns.FacetGrid(iris, hue="type", size=3).map(sns.distplot, "sepal_width").add_legend()
        plt.show()

    def plot_iris_setosa(self):
        counts, bin_edges = np.histogram(self.iris_setosa["petal_length"], bins=10, density=True)
        pdf = counts/(sum(counts))
        print(pdf)
        print(bin_edges)
        cdf = np.cumsum(pdf)
        plt.plot(bin_edges[1:], pdf)
        plt.plot(bin_edges[1:], cdf)
        plt.show()

    def plot_iris_virginica(self):
        counts, bin_edges = np.histogram(self.iris_virginica["petal_length"],
                                         bins=10, density=True)
        pdf = counts/(sum(counts))
        print(pdf)
        print(bin_edges)
        cdf = np.cumsum(pdf)
        plt.plot(bin_edges[1:], pdf)
        plt.plot(bin_edges[1:], cdf)
        plt.show()

    def plot_iris_versicolor(self):
        counts, bin_edges = np.histogram(self.iris_versicolor["petal_length"],
                                         bins=10, density=True)
        pdf = counts/(sum(counts))
        print(pdf)
        print(bin_edges)
        cdf = np.cumsum(pdf)
        plt.plot(bin_edges[1:], pdf)
        plt.plot(bin_edges[1:], cdf)
        plt.show()
