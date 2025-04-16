import torch 
import numpy as np

class LinearModel:

    def __init__(self):
        self.w = None 
    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))/X.size()[1]

        # your computation here: compute the vector of scores s
        return X@self.w

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is the value of score(Xi). 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions evaluated as score(X). y_hat.size() = (n,)
        """
        scores = self.score(X)
        return scores
    
class MyLinearRegression(LinearModel):
    
    def loss(self, X, y):
        s = self.score(X)
        return torch.mean((y - s)**2)
    
class OverParameterizedLinearRegressionOptimizer:
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        self.model.w = torch.linalg.pinv(X)@y