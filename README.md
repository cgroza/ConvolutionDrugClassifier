# Goal

Using a convolutional neural network designed for image recognition to
classify drugs from SDF structural data. This data species the 3D
position of every atom in a molecule.

The two classes to be distinguished are stimulants and sedatives.

Accumulating a good data set is very difficult. Therefore, fitting will be done with only 123 molecules.

80% of the data will be used for training.
20% will be used to test the accuracy of the neural network.

# Results

Confusion matrix

|-----------|----------|-----------|
|           | Sedative | Stimulant |
|-----------|----------|-----------|
| Sedative  |        8 |         2 |
|-----------|----------|-----------|
| Stimulant |     1    |        14 |

Training Accuracy = 0.908
Testing Accuracy = 0.88
Expected accuracy from guessing = 0.64 (due to imbalanced data set)

However, this varies depending on the distribution of training and testing data.
Accuracy can go as high as 96% when the random number generator seed is set to
1.

# Authors
Cristian Groza
Alexei Nolin-Lapalme
