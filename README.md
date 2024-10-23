# Machine Learning Models From Scratch

This repository contains Jupyter notebooks implementing various machine learning algorithms from scratch, without the use of machine learning libraries like `scikit-learn`. The focus is on understanding the core concepts and mathematics behind these models, and how to implement them using Python.

## Table of Contents

- [K-Nearest Neighbors (Classification & Regression)](#k-nearest-neighbors-classification--regression)
- [Linear and Polynomial Regression (Regression)](#linear-and-polynomial-regression-regression)
- [Logistic Regression (Classification)](#logistic-regression-classification)
- [K-Means Clustering](#k-means-clustering)

## K-Nearest Neighbors (Classification & Regression)

File: `KNN(Classification_&_Regression).ipynb`

This notebook demonstrates the **K-Nearest Neighbors (KNN)** algorithm for both classification and regression tasks. The KNN algorithm works by identifying the 'k' nearest data points to a target point and assigning a class or value based on the majority (for classification) or the average (for regression) of the neighbors.

Key Features:
- Explanation of the KNN algorithm.
- Implementation of KNN from scratch.
- Application on a sample dataset for both classification and regression tasks.

## Linear and Polynomial Regression (Regression)

File: `Linear_Polynomial_Regression(Regression).ipynb`

This notebook covers the implementation of **Linear** and **Polynomial Regression** models from scratch. The linear regression model is used for predicting a continuous dependent variable based on one or more independent variables, while the polynomial regression is a generalization to capture non-linear relationships.

Key Features:
- Step-by-step derivation and explanation of linear regression.
- Extension to polynomial regression.
- Gradient Descent-based optimization for model training.

## Logistic Regression (Classification)

File: `Logistic_Regression(Classification).ipynb`

This notebook introduces the **Logistic Regression** model, which is used for binary classification tasks. Logistic regression predicts the probability of an outcome, typically modeled as a sigmoid function applied to a linear combination of input features.

Key Features:
- Mathematical foundation of logistic regression.
- Implementation from scratch using Python.
- Application to a binary classification dataset.

## K-Means Clustering

File: `k-means(Clustering).ipynb`

This notebook covers **K-Means Clustering**, an unsupervised machine learning algorithm that partitions data into 'k' clusters. The goal is to minimize the variance within each cluster and maximize the variance between clusters.

Key Features:
- Explanation of the K-Means algorithm and its iterative process.
- Implementation of the algorithm from scratch.
- Application to a sample dataset to visualize clustering results.

## Requirements

- Python >3.8
- Jupyter Notebook
- You can install the python packages using

```bash
pip install package_name
```


## How to Use

1. Clone the repository:
    ```bash
    git clone https://github.com/dsrestrepo/Machine_Learning_Models_From_Scratch.git
    ```
2. Navigate to the repository folder:
    ```bash
    cd Machine_Learning_Models_From_Scratch
    ```
3. Open the desired Jupyter notebook:
    ```bash
    jupyter notebook
    ```
4. Select and run the notebook of your choice.

## License

This project is licensed under the MIT License.

---

Feel free to explore the notebooks to understand how different machine learning models work and how to implement them without relying on external machine learning libraries. Contributions and feedback are welcome!
