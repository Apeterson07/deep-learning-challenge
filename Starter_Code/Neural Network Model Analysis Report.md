Performance Analysis of Neural Network Model for Alphabet Soup Fund Selection

Overview

This analysis evaluates a deep learning model designed to predict whether an organization receiving funding from the Alphabet Soup nonprofit foundation will successfully utilize the funds. Using a binary classification approach, the model predicts success (1) or failure (0) based on historical data containing over 34,000 records with various organizational metadata.

The dataset includes attributes such as application type, classification, funding requests, and prior funding success. The goal is to develop and assess a deep neural network model capable of accurately predicting future funding success.

Results

Data Preprocessing

To ensure optimal performance, several preprocessing steps were undertaken before training the neural network model.

Target Variable:

The target variable (y) is IS_SUCCESSFUL, where 1 represents successful fund utilization and 0 indicates failure.

Feature Variables:

The independent variables (X) include application type, affiliation, classification, use case, organization, status, income amount, special considerations, and requested funding amount.

Variables Removed:

EIN and NAME were excluded as they are unique identifiers and do not contribute to predicting funding success.

Data Cleaning:

Categorical variables like APPLICATION_TYPE and CLASSIFICATION were encoded using one-hot encoding, with rare categories grouped under “Other.”

SPECIAL_CONSIDERATIONS was ordinally encoded, mapping “N” to 0 and “Y” to 1.

Feature Scaling:

StandardScaler was applied to normalize the feature values, enhancing neural network performance.

Model Development and Evaluation

Model Architecture

Input Layer:

The model accepts 42 input features (X_train.shape[1]).

Hidden Layers:

First hidden layer: 80 neurons with the ReLU activation function.

Second hidden layer: 30 neurons with the Tanh activation function.

Output Layer:

A single neuron with a Sigmoid activation function for binary classification.

Compilation:

Optimizer: Adam (adaptive gradient-based optimization)

Loss Function: Binary Cross-Entropy (suitable for classification tasks)

Evaluation Metric: Accuracy

Training:

The model was trained for 100 epochs, though additional training may further optimize performance.

Evaluation Results:

Training Loss: 0.5609

Training Accuracy: 72.75%

Test Loss: 0.5609

Test Accuracy: 72.75%

While the model achieved reasonable classification accuracy, further optimization is possible.

Comparison of Optimized Models

A performance comparison was conducted using different neural network architectures.

Model

Hidden Layers & Activation Functions

Loss

Accuracy

Model 1

2 hidden layers (Tanh, Tanh)

0.5604

72.63%

Model 2

4 hidden layers (ReLU, ReLU, Tanh, Sigmoid)

0.5671

73.03%

Model 3

2 hidden layers (ReLU, Sigmoid) with Early Stopping

0.5579

72.01%

Best Performing Model

Highest Accuracy: Model 2 (73.03%) – The additional layers and activation functions enhanced pattern recognition.

Lowest Loss: Model 3 (0.5579) – It minimized errors more effectively but had slightly lower accuracy.

Balanced Choice: Model 1 offered a good trade-off between accuracy (72.63%) and loss (0.5604).

Alternative Model Recommendation

Given the dataset characteristics, an alternative approach using tree-based models such as Random Forest or XGBoost may offer better predictive performance.

Model

Test Accuracy

Overfitting Risk

Random Forest

74%

Moderate

XGBoost

75%

Low-Moderate

Recommended Model: XGBoost

XGBoost outperforms deep learning for structured tabular data, offering:

Highest test accuracy (75%)

Better generalization on smaller datasets

A balance between high recall and low false positives

Model Selection Criteria:

If interpretability is a priority → Random Forest

If maximizing predictive power is key → XGBoost

Conclusion

While the deep neural network achieved reasonable accuracy (~72.75%), tree-based models, particularly XGBoost, provided superior predictive performance. Future improvements could involve hyperparameter tuning, dataset expansion, or feature engineering to enhance model accuracy and robustness.