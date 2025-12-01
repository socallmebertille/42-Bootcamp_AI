import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ex01.log_pred import logistic_predict_
from ex02.log_loss import log_loss_elem_, log_loss_
from ex05.vec_log_gradient import vec_log_gradient

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

def main():
    """Univariate Logistic Regression
    Train three different univariate models to predict spacecraft prices.
    """

    print("============= TEST ===================")

    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    mylr = MyLogisticRegression(thetas)

    print("X array : \n", X)
    print("Y array : \n", Y)
    print("theta : \n", thetas)

    print("============= 1 ===================")
    y_hat = mylr.predict_(X)
    print("predict : \n", y_hat)
    print("Expected : array([[0.99930437],\n\t\t[1. ],\n\t\t[1. ]])")

    print("============= 2 ===================")
    print("loss : \n", mylr.loss_(Y,y_hat))  # error in the subject : mylr.loss_(X,Y)
    print("Expected : 11.513157421577002")

    print("============= 3 ===================")
    mylr.fit_(X, Y)
    print("thetas trained : \n", mylr.thetas)
    print("Expected : array([[ 2.11826435]\n\t\t[ 0.10154334]\n\t\t[ 6.43942899]\n\t\t[-5.10817488]\n\t\t[ 0.6212541 ]])")

    print("============= 4 ===================")
    y_hat = mylr.predict_(X)
    print("predict : \n", y_hat)
    print("Expected : array([[0.57606717]\n\t\t[0.68599807]\n\t\t[0.06562156]])")

    print("============= 5 ===================")
    print("loss : \n", mylr.loss_(Y,y_hat))   # error in the subject : mylr.loss_(X,Y)
    print("Expected : 1.4779126923052268")

    return 0

if __name__ == "__main__":
    main()