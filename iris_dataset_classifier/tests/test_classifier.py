import os
from unittest import TestCase, mock

import pandas as pd

from iris_dataset_classifier.classifier import DatasetClassifier, iris

# Useful For Brief Info of Dataset 
print("First five rows")
print(iris.head())
print("*********")
print("columns", iris.columns)
print("*********")
print("shape:", iris.shape)
print("*********")
print("Size:", iris.size)
print("*********")
print("no of samples available for each type")
print(iris["type"].value_counts())
print("*********")
print(iris.describe())


class TestClassifier(TestCase):

    @mock.patch('iris_dataset_classifier.classifier.plt')
    def test_compare_all_species_per_feature(self, mock):
        DatasetClassifier().compare_all_species_per_feature()
        print("Here")
        self.assertEqual(mock.call_count, 4)

    @mock.patch('iris_dataset_classifier.classifier.plt')
    def test_plot_iris_setosa(self, mock):
        DatasetClassifier().plot_iris_setosa()
        self.assertEqual(mock.call_count, 1)

    @mock.patch('iris_dataset_classifier.classifier.plt')
    def test_plot_iris_virginica(self, mock):
        DatasetClassifier(iris).plot_iris_virginica()
        self.assertEqual(mock.call_count, 1)

    @mock.patch('iris_dataset_classifier.classifier.plt')
    def test_plot_iris_versicolor(self, mock):
        DatasetClassifier(iris).plot_iris_versicolor()
        self.assertEqual(mock.call_count, 1)