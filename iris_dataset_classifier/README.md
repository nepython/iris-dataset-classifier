## Brief Description about each file

### ``train.py``

**INFO**: The reason for not including `setosa` species is as it's already quite distinct from other two species.
It can be easily concluded from below figure

![input_data](https://github.com/nepython/iris-dataset-classifier/blob/master/iris_dataset_classifier/readme_images/input_data.png)

Since,`petal_length` and `petal_width` were the most distributed features hence I have selected them for classification

![feature-distribution](https://github.com/nepython/iris-dataset-classifier/blob/master/iris_dataset_classifier/readme_images/feature_distribution.png)

**Final result**

![final-result](https://github.com/nepython/iris-dataset-classifier/blob/master/iris_dataset_classifier/readme_images/final_result.png)

### ``predict.py``
**NOTE**: It is incomplete and still work in progress.
It's purpose is to study the input dataset and find suitable `synaptic_weights`.

After training finishes take the four features as input from user and predict the species giving percentage probability for each.

### ``classifier.py``
This is quite useful inorder to find info about the iris dataset used for training.

It can plot multiple figures comparing the four features of the three species individually in each figure.

The `iris_dataset` is stored in `tests` as `test_iris.xlsx`. 
It has been taken from a trusted and authenticated source [UCI Machine Learning Repository: Iris Data Set](https://archive.ics.uci.edu/ml/datasets/Iris).

![individual-feature](https://github.com/nepython/iris-dataset-classifier/blob/master/iris_dataset_classifier/readme_images/individual_feature.png)

Inorder to find out brief details about the dataset simply run `test_classifier.py` in `tests` directory.

Made with :sparkling_heart: by @nepython
