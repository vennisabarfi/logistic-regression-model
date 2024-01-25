# Logistic regression algorithm in PyTorch

# Design model(input,output size, forward pass)
# Construct loss and optimizer
# Training loop
# -forward pass: compute prediction and loss
# - backward pass: gradients
# - update weights

# Scikit learn documentation: https://scikit-learn.org/stable/auto_examples/index.html#preprocessing

import torch
import torch.nn as nn # neural network module
import numpy as np
from sklearn import datasets # to load a binary classification dataset
from sklearn.preprocessing import StandardScaler # to scale features
from sklearn.model_selection import train_test_split #separation of training and testing data

# prepare data
bc = datasets.load_breast_cancer() #breast cancer data for binary classification problem to predict cancer based on input features
X,y = bc.data, bc.target #X = independent variables(input features), y= dependent variables(what you want to predict)

n_samples, n_features = X.shape
print(n_samples,n_features) #num of rows = samples, num of columns = features

"""" So, after this line of code is executed, 
 n_samples will be the number of samples in your dataset, 
and n_features will be the number of features (independent variables) in each sample. """

# split data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=1234)

# scale (Normalization Step)
sc = StandardScaler() # make our features have zero mean and unit variance
X_train = sc.fit_transform(X_train) # appropriately scales the training data
X_test= sc.transform(X_test) # no need to fit as scaling parameters for X_train has been learned and you don't want to introduce new parameters

# convert to torch tensors
X_train = torch.from_numpy(X_train.astype(np.float32)) # convert to a tensor of float 32 datatype to avoid errors
X_test = torch.from_numpy(X_test.astype(np.float32)) # convert to a tensor of float 32 datatype to avoid errors
y_train = torch.from_numpy(y_train.astype(np.float32)) # convert to a tensor of float 32 datatype to avoid errors
y_test = torch.from_numpy(y_test.astype(np.float32)) # convert to a tensor of float 32 datatype to avoid errors

#ensure compatibility with the loss function etc. we are going to use
y_train = y_train.view(y_train.shape[0], 1) # make into a column vector
y_test = y_test.view(y_test.shape[0],1) # make into a column vector


# model
# linear combination of weights and biases: f = wx+b, sigmoid at the end
class LogisticRegression(nn.Module):
    
    def __init__(self, n_input_features):
        super(LogisticRegression, self) .__init__()
        self.linear = nn.Linear(n_input_features, 1) # linear network with n input features and 1 output feature since it is a classification model

    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
model = LogisticRegression(n_features)


# loss and optimizer
# loss is calculated using binary cross entropy loss (useful for binary classification)
learning_rate = 0.08
criterion = nn.BCELoss(reduction= 'mean') # specified the reduction here, optional
# parameters are the learnable weights and biases in the model
optimizer = torch.optim.SGD(model.parameters(), lr =learning_rate) # stochastic gradient descent. less computationally expensive than regular GD
# training loop
num_epochs = 500
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train) # forward pass with input data X_train
    loss =  criterion(y_predicted, y_train)

    # backward pass
    loss.backward() # calculate loss
    # updates 
    optimizer.step() # Pytorch will do all the updates for us

    # zero gradients. Empty the gradients
    optimizer.zero_grad()

    # print info for every 10th step
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# Evaluate model
with torch.no_grad(): # disable gradient calculation for evaluation
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round() # rounds predicted value to nearest integer.if greater than 0.5, return (round to) a +1. cls-- classes
    acc = y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0]) # denominator is the number of test samples
    print(f'accuracy = {acc:.4f}') # print accuracy

# code for multiple layers(hidden layers)

# import torch
# import torch.nn as nn

# class CustomLogisticRegression(nn.Module):
    
#     def __init__(self, n_input_features, n_hidden1, n_hidden2):
#         super(CustomLogisticRegression, self).__init__()
        
#         # First hidden layer
#         self.linear1 = nn.Linear(n_input_features, n_hidden1)
#         self.activation1 = nn.ReLU()
        
#         # Second hidden layer
#         self.linear2 = nn.Linear(n_hidden1, n_hidden2)
#         self.activation2 = nn.ReLU()
        
#         # Output layer
#         self.linear_out = nn.Linear(n_hidden2, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Forward pass through the layers
#         x = self.linear1(x)
#         x = self.activation1(x)
        
#         x = self.linear2(x)
#         x = self.activation2(x)
        
#         y_predicted = self.sigmoid(self.linear_out(x))
        
#         return y_predicted
