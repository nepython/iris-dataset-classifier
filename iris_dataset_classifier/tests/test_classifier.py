import unittest
from unittest import TestCase, mock

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
print("no of samples available for each species")
print(iris["Name"].value_counts())
print("*********")
print(iris.describe())


class TestClassifier(TestCase):

    @mock.patch('iris_dataset_classifier.classifier.plt.show')
    @mock.patch('iris_dataset_classifier.classifier.plt.hist')
    @mock.patch('iris_dataset_classifier.classifier.plt.legend')
    def test_compare_all_species_per_feature(self, mock_legend, mock_hist, mock_plot):
        DatasetClassifier().compare_all_species_per_feature()
        self.assertEqual(mock_legend.call_count, 4)
        self.assertEqual(mock_hist.call_count, 12)
        self.assertEqual(mock_plot.call_count, 1)

    @mock.patch('iris_dataset_classifier.classifier.plt.show')
    def test_plot_iris_setosa(self, mock):
        DatasetClassifier().plot_iris_setosa()
        self.assertEqual(mock.call_count, 1)

    @mock.patch('iris_dataset_classifier.classifier.plt.show')
    def test_plot_iris_virginica(self, mock):
        DatasetClassifier().plot_iris_virginica()
        self.assertEqual(mock.call_count, 1)

    @mock.patch('iris_dataset_classifier.classifier.plt.show')
    def test_plot_iris_versicolor(self, mock):
        DatasetClassifier().plot_iris_versicolor()
        self.assertEqual(mock.call_count, 1)


if __name__ == '__main__':
    unittest.main()
