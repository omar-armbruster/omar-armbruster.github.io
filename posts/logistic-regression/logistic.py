import torch 
import numpy as np

class LinearModel:

    def __init__(self):
        self.w = None 
        self.wprev = None
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
            self.wprev = self.w

        # your computation here: compute the vector of scores s
        return X@self.w

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        scores = self.score(X)
        return (scores >= 0)*1.0

class LogisticRegression(LinearModel):
    
    def loss(self,X,y):
        n = len(X)
        sigmoid = lambda x: 1/(1 + torch.exp(-x))
        s = self.score(X)
        l = (1/n)*torch.sum(-y*torch.log(sigmoid(s)) - (1 - y)*torch.log(1 - sigmoid(s)))
        return l

    def grad(self, X,y):
        sigmoid = lambda x: 1/(1 + torch.exp(-x))
        s = self.score(X)
        grad = torch.mean((sigmoid(s) - y)[:,None]*X, axis = 0)
        return grad
    

class GradientDescentOptimizer:
    def __init__(self, model):
        self.model = model 

    def step(self, X, y, alpha = 1, beta = 0):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """
        if self.model.loss(X, y) > 0:
            # print(self.model.w, self.model.grad(X, y, alpha))
            # self.model.w -= self.model.grad(X, y, alpha)
            new = self.model.w - alpha*self.model.grad(X,y) + beta*(self.model.w - self.model.wprev) 
            self.model.wprev = self.model.w
            self.model.w = new