Neural Network Model for Alphabet Soup Fund Selection: Performance Analysis

1. Introduction

Why This Analysis Matters

Alphabet Soup, a nonprofit organization, provides funding to various groups, but not every recipient successfully utilizes the funds. Our goal is to build a predictive model that helps determine which organizations are most likely to make good use of the funding. Using historical data from over 34,000 funding applications, we trained a deep learning model to predict success (1) or failure (0) based on key organizational attributes.

2. Data Preprocessing

Understanding the Data

Target Variable (y): IS_SUCCESSFUL – Indicates whether the funding was used successfully.

Feature Variables (X): Includes attributes such as:

APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT, etc.

Cleaning and Preparing the Data

Dropped Columns: EIN and NAME were removed because they don’t contribute to prediction.

Categorical Encoding:

One-hot encoding for APPLICATION_TYPE and CLASSIFICATION, grouping rare categories under “Other.”

SPECIAL_CONSIDERATIONS converted into binary values (0 for “N” and 1 for “Y”).

Feature Scaling:

StandardScaler was applied to numerical values to ensure uniformity, helping the neural network train more effectively.

3. Model Development

Neural Network Structure

Input Layer: 42 features.

Hidden Layers:

Layer 1: 80 neurons, ReLU activation.

Layer 2: 30 neurons, Tanh activation.

Output Layer:

A single neuron with Sigmoid activation (for binary classification).

Compilation & Training:

Optimizer: Adam

Loss Function: Binary Cross-Entropy

Trained for 100 epochs.

4. Results and Model Performance

Performance Metrics

Training Accuracy: 72.75%

Test Accuracy: 72.75%

Training & Test Loss: 0.5609

Model Comparisons

Model

Hidden Layers & Activations

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

What These Results Tell Us

Best Accuracy: Model 2 performed slightly better (73.03%), likely due to the additional hidden layers.

Lowest Loss: Model 3 had the lowest error rate (0.5579), though accuracy was slightly lower.

Balanced Choice: Model 1 had a good trade-off between accuracy (72.63%) and loss (0.5604).

Key Takeaways

Which factors are most important?

APPLICATION_TYPE, ASK_AMT, INCOME_AMT, and CLASSIFICATION had the strongest impact on predictions.

How did encoding affect performance?

One-hot encoding improved categorical feature representation but increased model complexity.

Why scale features?

StandardScaler ensured consistent numeric inputs, making training more stable.

How did adding layers impact the model?

More layers helped Model 2 achieve the highest accuracy, though at the cost of increased training time.

Does loss always align with accuracy?

Not necessarily—Model 3 had the lowest loss but wasn’t the most accurate.

How can we improve performance?

Trying different architectures, optimizing hyperparameters, or exploring other machine learning models.

5. Alternative Model Recommendation

Why Consider a Tree-Based Model?

Neural networks work well with large datasets, but for structured tabular data, decision trees often perform better. We compared two popular tree-based models:

Model

Test Accuracy

Risk of Overfitting

Random Forest

74%

Moderate

XGBoost

75%

Low-Moderate

Why XGBoost Wins

Higher Accuracy (75%) compared to our neural network model.

Better suited for tabular data, requiring fewer computational resources.

Lower risk of overfitting due to built-in regularization.

Choosing the Right Model:

Need transparency? → Random Forest (easier to interpret)

Need better accuracy? → XGBoost (more powerful predictions)

6. Conclusion

Our deep learning model achieved 72.75% accuracy, but alternative models like XGBoost performed even better. While deep neural networks can be useful for complex problems, structured data often benefits from decision trees. Future work should explore hyperparameter tuning, increasing dataset size, and feature engineering to improve model performance.



