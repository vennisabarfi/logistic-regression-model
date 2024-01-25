# Logistic Regression: Breast Cancer Prediction

This GitHub repository contains a simple implementation of logistic regression using PyTorch. The logistic regression model is designed to predict binary outcomes, making it suitable for tasks such as binary classification. I am modeling breast cancer predictions with this model

## Key Components

1. **Model Design:**
   - The model is implemented as a class named `LogisticRegression` in PyTorch.
   - The model's forward pass computes the sigmoid activation of a linear combination of input features using the `nn.Linear` module.

2. **Loss and Optimizer:**
   - Binary Cross Entropy Loss (`nn.BCELoss`) is used as the loss function, suitable for binary classification problems.
   - Stochastic Gradient Descent (SGD) is employed as the optimizer to update the learnable parameters of the model.

3. **Training Loop:**
   - The training loop consists of forward and backward passes, updating model parameters to minimize the loss.
   - The loop runs for a specified number of epochs, printing the loss every 10th epoch.

4. **Data Preparation:**
   - Breast cancer data from the Scikit-learn library is used for training and testing.
   - Features are scaled using `StandardScaler` to have zero mean and unit variance.
   - Data is split into training and testing sets using `train_test_split`.

5. **Evaluation:**
   - Model evaluation is performed on the test set, calculating accuracy by comparing predicted and actual labels.
   - The code also includes an example of extending the model to multiple layers for experimentation.

## Usage
```python
# Clone the repository
git clone https://github.com/yourusername/logistic-regression-pytorch.git

# Navigate to the project directory
cd logistic-regression-pytorch

# Run the script
python logistic_regression.py
```

Feel free to explore the code, experiment with different datasets, and modify the model architecture for your specific use case. Contributions and improvements are welcome!
