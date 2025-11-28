import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Module07.ex01.prediction import predict_
from Module07.ex02.loss import loss_, loss_elem_
from Module05.ex09.other_losses import mse_
from Module07.ex04.fit import fit_

class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        if isinstance(thetas, list):
            self.thetas = np.array(thetas, dtype=float).reshape(-1, 1)
        elif isinstance(thetas, np.ndarray):
            self.thetas = thetas.reshape(-1, 1)
        else:
            self.thetas = thetas
    
    def fit_(self, x, y):
        new_theta = fit_(x, y, self.thetas, self.alpha, self.max_iter)
        if new_theta is not None:
            self.thetas = new_theta
        return self.thetas
    
    def predict_(self, x):
        if not isinstance(x, np.ndarray) or not isinstance(self.thetas, np.ndarray):
            return None
        if x.shape[1] + 1 != self.thetas.shape[0]:
            return None
        X_ = np.hstack((np.ones((x.shape[0], 1)), x))
        return X_ @ self.thetas
    
    def loss_elem_(self, y, y_hat):
        return loss_elem_(y, y_hat)
    
    def loss_(self, y, y_hat):
        return loss_(y, y_hat)
    
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
        
    @staticmethod
    def mse(y, y_hat):
        return mse_(y, y_hat)
    
    def plot_data(self, x, y, loss: bool, title: str, xlabel: str, ylabel: str):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return
        if x.size == 0 or y.size == 0:
            return
        if x.ndim != 2 or y.ndim != 2:
            return
        y_hat = self.predict_(x)
        plt.figure(figsize=(7,5))
        plt.scatter(x, y, color='lightskyblue', label='True data')
        x_line = np.array([x.min(), x.max()])
        y_line = self.thetas[0] + self.thetas[1] * x_line
        plt.plot(x_line, y_line, color='lightgreen', label='Prediction')
        if loss == True:
            for i, (xi, yi, yhi) in enumerate(zip(x, y, y_hat)): # zip parcourt en parallèle les 3 array
                plt.plot([xi, xi], [yi, yhi], linestyle='--', color='red', linewidth=1, label='Loss' if i == 0 else None) # coordonnées x (2x le même) et y réel et prédit
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_loss(self, x, y, precision: float):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return
        if x.size == 0 or y.size == 0:
            return
        if x.ndim != 2 or y.ndim != 2:
            return
        
        theta0_list = np.linspace(self.thetas[0][0] - precision, self.thetas[0][0] + precision, 6)   # 6 val de theta0 espacées également de -20 à 100
        theta1_range = np.linspace(-14, -4, 200)
        for t0 in theta0_list:
            losses = []
            for t1 in theta1_range:
                tmp_thetas = np.array([[t0], [t1]])
                y_hat = predict_(x, tmp_thetas)
                loss = self.mse(y, y_hat)
                losses.append(loss)
            plt.plot(theta1_range, losses, label=f"J(θ0={t0:.1f}, θ1)")
        plt.xlabel("θ1")
        plt.ylabel("cost function J(θ0, θ1)")
        plt.title("Loss evolution for different values of θ0")
        plt.grid(True)
        plt.legend()
        plt.show()