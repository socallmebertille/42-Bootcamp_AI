import numpy as np
from itertools import combinations_with_replacement

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Module07.ex09.data_spliter import data_spliter
from Module08.ex01.log_pred import logistic_predict_
from Module08.ex02.log_loss import log_loss_elem_, log_loss_
from Module08.ex05.vec_log_gradient import vec_log_gradient

class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        if isinstance(theta, np.ndarray):
            self.thetas = theta.reshape(-1, 1)
        else:
            print("Error\nThetas array isnot from the numpy library")

    @staticmethod
    def data_spliter_(x, y, proportion):
        return data_spliter(x, y, proportion)

    def predict_(self, x):
        return logistic_predict_(x, self.thetas)
    
    def loss_elem_(self, y, y_hat):
        return log_loss_elem_(y, y_hat)
    
    def loss_(self, y, y_hat):
        return log_loss_(y, y_hat)
    
    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        if x.size == 0 or y.size == 0:
            return None
        new_theta = np.copy(self.thetas)
        for _ in range(self.max_iter):
            grad = vec_log_gradient(x, y, new_theta)
            if grad is None:
                return None
            new_theta = new_theta - self.alpha * grad
        self.thetas = np.copy(new_theta)
        return self.thetas
    
    @staticmethod
    def add_polynomial_features(x, power):
        """Add polynomial features to matrix X for all columns and all combinations up to 'power'."""
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            return None
        if not isinstance(power, int) or power < 1:
            return None

        m, n = x.shape
        X_poly = np.ones((m, 0))  # initialisation vide
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
