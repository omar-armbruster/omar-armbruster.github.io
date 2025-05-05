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
    
    def loss(self,X,y, lam = 0):
        """
        Computes the loss over the entire data set X based on known labels y. 

        ARGUMENTS:
        X, torch.Tensor: the feature n by p dimensional feature matrix such that n is 
        the number of observations and p is the number of features. Because we are using a linear model,
        this implementaiton assumes that the last column is a constant column of 1s. 

        y, torch.Tensor: An array of length n where the ith entry corresponds to the label of the ith row of X.

        RETURNS: 
            l, float: The loss of the given feature matrix based on the weights defined by self.w
        """
        n = len(X)
        sigmoid = lambda x: 1/(1 + torch.exp(-x))
        s = self.score(X)
        l = (1/n)*torch.sum(-y*torch.log(sigmoid(s)) - (1 - y)*torch.log(1 - sigmoid(s))) + lam*(self.w**2).sum()
        return l

    def grad(self, X,y, lam = 0):
        """
        Computes the gradient needed to update self.w for gradient descent.

        ARGUMENTS:
        X, torch.Tensor: the feature n by p dimensional feature matrix such that n is 
        the number of observations and p is the number of features. Because we are using a linear model,
        this implementaiton assumes that the last column is a constant column of 1s. 

        y, torch.Tensor: An array of length n where the ith entry corresponds to the label of the ith row of X.

        RETURNS: 
            grad: torch.Tensor: Array of length p representing the change to the self.w array.
        """
        sigmoid = lambda x: 1/(1 + torch.exp(-x))
        s = self.score(X)
        grad = torch.mean((sigmoid(s) - y)[:,None]*X, axis = 0) + lam*2*self.w#torch.cat((lam*(2*self.w[:-1]), torch.tensor([0])), axis = 0)
        return grad
    

class GradientDescentOptimizer:
    def __init__(self, model):
        self.model = model 

    def step(self, X, y, alpha = 1, beta = 0, lam = 0):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 

        ARGUMENTS:
        X, torch.Tensor: the feature n by p dimensional feature matrix such that n is 
        the number of observations and p is the number of features. Because we are using a linear model,
        this implementaiton assumes that the last column is a constant column of 1s. 

        y, torch.Tensor: An array of length n where the ith entry corresponds to the label of the ith row of X.

        alpha, float: The learning rate for vanilla gradient descent. Determines how highly weighted the gradient updates
        should be. alpha > 0 for the algorithm to work effectively.

        beta, float: The learning rate for gradient descent with momentum. Determines how highly weighted the momentum term is,
        which influences how large the adjustments to the weights are.

        RETURNS:
        NA, no return value but weights are updated based on the gradient function.
        """
        if self.model.loss(X, y) > 0:
            new = self.model.w - alpha*self.model.grad(X,y, lam) + beta*(self.model.w - self.model.wprev) 
            self.model.wprev = self.model.w
            self.model.w = new

class NewtonOptimizer:
    def __init__(self, model):
        self.model = model
    
    def hessian(self, X, y):
        sigmoid = lambda x: 1/(1 + torch.exp(-x))
        diag = (torch.eye(X.size(0)) * sigmoid(y)*(1 - sigmoid(y)))
        hess = (X.T@diag@X)/X.size(0)
        return hess
    
    def step(self, X, y, alpha = 1, lam = 0):
        if self.model.loss(X, y) > 0:
            new = self.model.w - alpha*torch.linalg.pinv(self.hessian(X,y))@self.model.grad(X,y, lam)
            self.model.wprev = self.model.w
            self.model.w = new

class AdamOptimizer:
    def __init__(self, model, batch_size, alpha, beta_1, beta_2, w0 = None):
        self.model = model
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.w0 = w0
        self.t = 0
    def step(self, X, y, lam = 0):
        if self.w0 == None:
            self.w0  = torch.rand(X.size(1))
        if self.t == 0:
            self.m = torch.zeros(X.size(1))
            self.v = torch.zeros(X.size(1))
        if self.model.loss(X,y) > 0:
            self.t += 1
            data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y),
            batch_size = self.batch_size,
            shuffle = True)
            for X_batch, y_batch in data_loader:
                g = self.model.grad(X_batch,y_batch, lam)
                self.m = self.beta_1*self.m + (1 - self.beta_1)*g
                self.v = self.beta_2*self.v + (1 - self.beta_2)*g**2
                mhat = self.m/(1 - self.beta_1**self.t)
                vhat = self.v/(1 - self.beta_2**self.t)
                self.model.w = self.model.w - self.alpha*mhat/(vhat**0.5 + 10e-8)

class StochasticGradientDescent:
    def __init__(self, model, batch_size, alpha):
        self.model = model
        self.batch_size = batch_size
        self.alpha = alpha
    def step(self, X, y, lam = 0):
        if self.model.loss(X,y) > 0:
            data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y),
            batch_size = self.batch_size,
            shuffle = True)
            for i, data in enumerate(data_loader):
                X_batch, y_batch = data
                g = self.model.grad(X_batch,y_batch, lam)
                self.model.w = self.model.w - self.alpha*g
    