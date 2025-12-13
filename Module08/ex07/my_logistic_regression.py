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
            print("Error\nThetas array is not from the numpy library")
        
        # Cache pour éviter de recalculer
        self.mean_ = None
        self.std_ = None
        self.x_train_ = None
        self.x_test_ = None
        self.y_train_ = None
        self.y_test_ = None
        self.x_train_poly_ = None
        self.x_test_poly_ = None
        self.x_train_norm_ = None
        self.x_test_norm_ = None

    def prepare_data(self, x, y, proportion=0.8, poly_degree=3, normalize=True):
        """
        Prépare les données : split, polynomial features, normalisation
        Sauvegarde tout en cache pour réutilisation
        """
        # Step 1: Split
        self.x_train_, self.x_test_, self.y_train_, self.y_test_ = data_spliter(x, y, proportion)
        
        # Step 2: Polynomial features
        self.x_train_poly_ = self.add_polynomial_features(self.x_train_, poly_degree)
        self.x_test_poly_ = self.add_polynomial_features(self.x_test_, poly_degree)
        
        # Step 3: Normalisation
        if normalize:
            self.x_train_norm_, self.mean_, self.std_ = self.normalize_features(self.x_train_poly_)
            self.x_test_norm_ = (self.x_test_poly_ - self.mean_) / self.std_
        else:
            self.x_train_norm_ = self.x_train_poly_
            self.x_test_norm_ = self.x_test_poly_
        
        # Réinitialiser thetas avec la bonne dimension
        self.thetas = np.ones((self.x_train_norm_.shape[1] + 1, 1))
        
        return self

    def train(self):
        """Entraîne le modèle sur les données préparées"""
        if self.x_train_norm_ is None or self.y_train_ is None:
            print("Error: Data not prepared. Call prepare_data() first.")
            return None
        return self.fit_(self.x_train_norm_, self.y_train_)

    def evaluate(self):
        """Évalue le modèle et retourne les prédictions et accuracy"""
        if self.x_train_norm_ is None or self.x_test_norm_ is None:
            print("Error: Data not prepared. Call prepare_data() first.")
            return None
        
        y_pred_train = self.predict_(self.x_train_norm_)
        y_pred_test = self.predict_(self.x_test_norm_)
        
        # Convert to binary
        y_pred_binary_train = np.where(y_pred_train >= 0.5, 1, 0)
        y_pred_binary_test = np.where(y_pred_test >= 0.5, 1, 0)
        
        # Calculate accuracy
        accuracy_train = np.mean(y_pred_binary_train == self.y_train_)
        accuracy_test = np.mean(y_pred_binary_test == self.y_test_)
        
        return {
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'y_pred_binary_train': y_pred_binary_train,
            'y_pred_binary_test': y_pred_binary_test,
            'accuracy_train': accuracy_train,
            'accuracy_test': accuracy_test
        }

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