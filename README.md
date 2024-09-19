# Ensemble-Methods

This repository contains implementations of various ensemble learning techniques from scratch, using only `numpy`, `pandas`, and `matplotlib`. The models are applied to the MNIST dataset, and dimensionality reduction is performed using Principal Component Analysis (PCA). 

The key ensemble methods implemented are Decision Trees, Bagging, AdaBoost, and Gradient Boosting.

## Project Overview

The goal of this project is to demonstrate the power of ensemble learning models in improving predictive performance by combining multiple models. All methods have been built from scratch, and no machine learning libraries have been used to implement these algorithms. By leveraging the MNIST dataset, these models predict labels with high accuracy and handle dimensionality reduction through PCA.

## Files Overview

### 1. **DecisionTrees and Bagging.ipynb**

- **Algorithm**: Decision Tree using Gini Index for splitting, implemented for bagging.
- **Accuracy**: 82.4% overall accuracy, with class-wise accuracy reported.
- **Highlights**: Class-wise accuracy calculation and feature splitting by mid-points.
- **Metrics**: Gini Index, class-wise accuracy metrics.

### 2. **AdaBoost.ipynb**

- **Algorithm**: AdaBoost with Decision Trees as weak learners.
- **Preprocessing**: Applied PCA for dimensionality reduction with `p=5`.
- **Highlights**: Weights are updated based on misclassified points, and decision boundaries are recalculated in each iteration.
- **Metrics**: Boosting results visualized using accuracy trends over iterations.

### 3. **GradientBoosting.ipynb**

- **Algorithm**: Gradient Boosting with SSR (Sum of Squared Residuals) for optimization.
- **Preprocessing**: PCA for dimensionality reduction, focusing on 5 principal components.
- **Highlights**: Tree splitting by minimizing SSR and model performance tracked over several iterations.
- **Metrics**: MSE reduction over boosting rounds plotted.

### 4. **PCA.ipynb**

- **Algorithm**: Principal Component Analysis (PCA) for dimensionality reduction.
- **Highlights**: Eigen decomposition of the covariance matrix and projection onto top principal components.
- **Dataset**: MNIST dataset.

## Dataset

The MNIST dataset has been used to test the models. Dimensionality reduction via PCA is applied to reduce computational complexity while maintaining the key features of the data.

## Key Concepts

- **Bagging**: Reduces variance by averaging predictions from multiple decision trees.
- **AdaBoost**: Focuses on improving weak learners iteratively by reweighting misclassified points.
- **Gradient Boosting**: Optimizes the model by minimizing SSR, adjusting residuals in each boosting round.

## Results

- **Decision Trees and Bagging**: Achieved an overall accuracy of 82.4% on the test data.
- **AdaBoost**: Iteratively improved performance by focusing on misclassified data points.
- **Gradient Boosting**: Reduced Mean Squared Error (MSE) over several boosting rounds.

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`

Install the required libraries by running:
```bash
pip install numpy pandas matplotlib
```
## Usage

To use the models and reproduce the results, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Ensemble-Methods.git
   cd Ensemble-Methods
   pip install numpy pandas matplotlib
   jupyter notebook
```
