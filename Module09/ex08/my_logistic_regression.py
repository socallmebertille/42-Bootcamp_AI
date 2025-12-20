import numpy as np
from itertools import combinations_with_replacement

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Module07.ex09.data_spliter import data_spliter
from Module08.ex01.log_pred import logistic_predict_
from Module08.ex02.log_loss import log_loss_elem_, log_loss_
from Module08.ex05.vec_log_gradient import vec_log_gradient
from Module09.ex03.logistic_loss_reg import reg_log_loss_
from Module09.ex05.reg_logistic_grad import vec_reg_logistic_grad

class MyLogisticRegression():
    """
    Description:
        My personnal logistic regression to classify things.
    """
    supported_penalities = ['l2']  # We consider l2 penalties only
    
    def __init__(self, theta, alpha=0.001, max_iter=1000, penality='l2', lambda_=1.0):
        """
        Args:
            theta: initial parameter vector
            alpha: learning rate
            max_iter: maximum number of iterations
            penality: type of regularization ('l2' or None)
            lambda_: regularization parameter (only used if penality='l2')
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.penality = penality
        # Lambda is 0 if penality is not in supported_penalities
        self.lambda_ = lambda_ if penality in self.supported_penalities else 0.0
        
        if isinstance(theta, np.ndarray):
            self.theta = theta.reshape(-1, 1)
        else:
            print("Error\nTheta array is not from the numpy library")
    
    @staticmethod
    def data_spliter_(x, y, proportion):
        return data_spliter(x, y, proportion)
    
    def predict_(self, x):
        """Predict using logistic function (sigmoid)"""
        return logistic_predict_(x, self.theta)
    
    def loss_elem_(self, y, y_hat):
        """Element-wise logistic loss (cross-entropy) - not regularized"""
        return log_loss_elem_(y, y_hat)
    
    def loss_(self, y, y_hat):
        """
        Compute loss with optional L2 regularization
        - If penality='l2': use regularized loss
        - Otherwise: use standard logistic loss
        """
        if self.penality == 'l2' and self.lambda_ > 0:
            return reg_log_loss_(y, y_hat, self.theta, self.lambda_)
        else:
            return log_loss_(y, y_hat)
    
    def gradient_(self, x, y):
        """
        Compute gradient with optional L2 regularization
        - If penality='l2': use regularized gradient
        - Otherwise: use standard logistic gradient
        """
        if self.penality == 'l2' and self.lambda_ > 0:
            return vec_reg_logistic_grad(y, x, self.theta, self.lambda_)
        else:
            return vec_log_gradient(x, y, self.theta)
    
    def fit_(self, x, y):
        """
        Fit the logistic regression model using gradient descent
        - If penality='l2': use regularized gradient descent
        - Otherwise: use standard gradient descent
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        if x.size == 0 or y.size == 0:
            return None
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        new_theta = np.copy(self.theta)
        for _ in range(self.max_iter):
            grad = self.gradient_(x, y)
            if grad is None:
                return None
            new_theta = new_theta - self.alpha * grad
        self.theta = new_theta
        return self.theta
    
    @staticmethod
    def add_polynomial_features(x, power):
        """Add polynomial features to matrix X for all columns and all combinations up to 'power'."""
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            return None
        if not isinstance(power, int) or power < 1:
            return None

        m, n = x.shape
        X_poly = np.ones((m, 0))
        for p in range(1, power + 1):
            for comb in combinations_with_replacement(range(n), p):
                new_col = np.prod(x[:, comb], axis=1).reshape(-1, 1)
                X_poly = np.hstack((X_poly, new_col))
        return X_poly
    
    @staticmethod
    def normalize_features(x):
        """Normalize features in X."""
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            return None
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        X_norm = (x - mean) / std
        return X_norm, mean, std