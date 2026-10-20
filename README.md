# Experiment 11: Multiclass Classification Using an Artificial Neural Network (ANN)

## 1. Aim

To study and implement an Artificial Neural Network (ANN) algorithm to classify data into multiple classes based on predictor variables.

## 2. Objectives

1. Understand the basic architecture of an Artificial Neural Network.
2. Identify input features, weights, biases, and activation functions.
3. Understand forward propagation and backpropagation.
4. Implement multiclass classification using an ANN.
5. Understand the Softmax activation function and categorical cross-entropy loss.
6. Evaluate classification performance using accuracy, precision, recall, F1-score, and a confusion matrix.

## 3. Introduction

An Artificial Neural Network (ANN) is a machine-learning model inspired by the functioning of biological neural networks. It consists of interconnected processing units called neurons, arranged in layers.

ANNs can learn complex relationships between input features and output classes by adjusting their weights and biases during training.

In multiclass classification, the model assigns each observation to one of three or more classes. This experiment demonstrates ANN-based multiclass classification using the Iris dataset.

## 4. Theory

### 4.1 Architecture of an Artificial Neural Network

A basic feed-forward ANN consists of three types of layers:

- **Input Layer:** Receives predictor variables from the dataset.
- **Hidden Layer(s):** Learns patterns and relationships in the input data.
- **Output Layer:** Produces predictions for the target classes.

Each neuron calculates a weighted sum of its inputs and applies an activation function.

### 4.2 Weighted Sum

The weighted sum calculated by a neuron is:

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

Where:

- \(x_i\) = Input feature
- \(w_i\) = Weight associated with the input
- \(b\) = Bias
- \(n\) = Number of input features
- \(z\) = Weighted sum

### 4.3 ReLU Activation Function

The Rectified Linear Unit (ReLU) introduces non-linearity into the network and is commonly used in hidden layers.

$$
\text{ReLU}(z) = \max(0,z)
$$

For negative inputs, ReLU returns zero. For positive inputs, it returns the input value.

### 4.4 Softmax Activation Function

For multiclass classification, the Softmax function converts output scores into probabilities.

$$
p_k = \frac{\exp(z_k)}
{\sum_{j=1}^{K}\exp(z_j)}
$$

Where:

- \(p_k\) = Predicted probability of class \(k\)
- \(z_k\) = Output score for class \(k\)
- \(K\) = Total number of classes

The sum of the probabilities across all classes equals 1. The class with the highest predicted probability is selected as the final prediction.

### 4.5 Categorical Cross-Entropy Loss

Categorical cross-entropy measures the difference between actual class labels and predicted probabilities.

$$
L = -\sum_{k=1}^{K} y_k \log(p_k)
$$

Where:

- \(y_k\) = Actual class label in one-hot encoded form
- \(p_k\) = Predicted probability of class \(k\)
- \(K\) = Total number of classes

The training process aims to minimize this loss.

### 4.6 Forward Propagation

During forward propagation, input features pass through the network layers. Each neuron performs a weighted-sum calculation and applies an activation function to generate the output.

### 4.7 Backpropagation

Backpropagation calculates gradients of the loss function with respect to the network weights and biases. An optimization algorithm uses these gradients to update the parameters and reduce prediction error.

The weights are updated conceptually using gradient descent:

$$
w_{\text{new}} = w_{\text{old}} -
\eta \frac{\partial L}{\partial w}
$$

Where:

- \(w_{\text{new}}\) = Updated weight
- \(w_{\text{old}}\) = Previous weight
- \(\eta\) = Learning rate
- \(L\) = Loss function

## 5. Dataset Description

### Iris Dataset

The Iris dataset, available through `scikit-learn`, is used for this experiment.

| Property | Description |
|---|---|
| Dataset name | Iris |
| Total observations | 150 |
| Number of predictor variables | 4 |
| Number of target classes | 3 |
| Problem type | Multiclass classification |
| Learning type | Supervised learning |

### Predictor Variables

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

### Target Classes

- Iris Setosa
- Iris Versicolor
- Iris Virginica

**Important:** Target labels are used as the supervised learning output, not as input features.

## 6. Software and Libraries Required

- Python 3.x
- Jupyter Notebook
- NumPy
- pandas
- Matplotlib
- scikit-learn

### Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## 7. Algorithm

1. Start the program.
2. Import the required Python libraries.
3. Load the Iris dataset.
4. Separate predictor variables and target labels.
5. Divide the dataset into training and testing sets.
6. Standardize the predictor variables using a scaler fitted on the training data.
7. Initialize the ANN classifier using `MLPClassifier`.
8. Train the model using the training dataset.
9. Predict the classes of the test dataset.
10. Calculate accuracy, precision, recall, and F1-score.
11. Generate a confusion matrix to compare actual and predicted classes.
12. Display the evaluation results and conclude the experiment.

## 8. Implementation Overview

The experiment is implemented in the Jupyter Notebook:

`Experiment_11_Multiclass_ANN_Classification.ipynb`

The implementation uses the `MLPClassifier` class from `sklearn.neural_network`.

The main stages are:

- **Data Preparation:** Load the dataset and separate features from target labels.
- **Data Splitting:** Create training and testing subsets.
- **Feature Scaling:** Standardize the predictor variables using `StandardScaler`.
- **Model Training:** Train the neural network using the training subset.
- **Prediction:** Predict the target classes for the testing subset.
- **Evaluation:** Measure classification performance using standard metrics.

The exact architecture, training parameters, and evaluation values depend on the settings used in the notebook.

## 9. Evaluation Metrics

### 9.1 Accuracy

Accuracy measures the proportion of correctly classified observations.

$$
\text{Accuracy} =
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
$$

### 9.2 Precision

Precision measures the proportion of correctly predicted positive observations among all observations predicted as positive for a class.

$$
\text{Precision} =
\frac{TP}{TP+FP}
$$

### 9.3 Recall

Recall measures the proportion of actual positive observations correctly identified for a class.

$$
\text{Recall} =
\frac{TP}{TP+FN}
$$

### 9.4 F1-Score

The F1-score is the harmonic mean of precision and recall.

$$
F_1 = 2 \times
\frac{\text{Precision}\times\text{Recall}}
{\text{Precision}+\text{Recall}}
$$

Where:

- \(TP\) = True Positives
- \(FP\) = False Positives
- \(FN\) = False Negatives

For multiclass problems, precision, recall, and F1-score can be calculated using averaging methods such as macro, micro, or weighted averaging.

### 9.5 Confusion Matrix

A confusion matrix compares actual class labels with predicted class labels. It helps identify correctly classified observations and misclassifications for each class.

## 10. Expected Output

After executing all cells in the Jupyter Notebook, the experiment should produce the following results, depending on the implementation:

1. Dataset information and predictor variables.
2. Training and testing dataset details.
3. ANN model training results.
4. Predicted class labels.
5. Classification accuracy.
6. Precision, recall, and F1-score.
7. Classification report.
8. Confusion matrix visualization.

**Note:** Actual metric values should be taken from the executed notebook. They are not specified here because they depend on the model configuration, data split, and training settings.

## 11. Applications

ANN-based multiclass classification is useful in:

- Image and object recognition.
- Medical diagnosis and disease classification.
- Document and text categorization.
- Product classification.
- Pattern recognition.
- Species identification.

## 12. Advantages

- Learns complex and non-linear relationships.
- Supports classification into multiple classes.
- Can work with several predictor variables.
- Can generalize to previously unseen observations when trained appropriately.

## 13. Limitations

- Performance depends on the network architecture and hyperparameters.
- Feature scaling is generally important for effective training.
- Training can require more computation than simpler models.
- Neural networks can overfit when the model is not appropriately regularized.
- Model predictions may be difficult to interpret compared with simpler classifiers.

## 14. Result

The Artificial Neural Network algorithm is implemented to classify the Iris dataset into three species using four predictor variables. Its performance is evaluated using classification metrics and a confusion matrix.

The final classification performance is determined by the results obtained from executing the Jupyter Notebook.

## 15. Conclusion

This experiment demonstrates how an Artificial Neural Network performs multiclass classification using predictor variables. It covers the network architecture, weighted-sum calculation, ReLU activation, Softmax probabilities, loss calculation, forward propagation, and backpropagation. The model is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

The experiment provides a practical understanding of ANN-based supervised learning and its applications in multiclass classification.

## 16. References

1. [Scikit-learn Documentation — Neural Network Models](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
2. [Scikit-learn Documentation — Iris Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
3. [Scikit-learn Documentation — MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
4. [Scikit-learn Documentation — Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

**Experiment:** 11  
**Topic:** Multiclass Classification Using an Artificial Neural Network  
**Notebook:** `Experiment_11_Multiclass_ANN_Classification.ipynb`