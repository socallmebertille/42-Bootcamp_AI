import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Module08.ex05.vec_log_gradient import MyLinearRegression as MyLR

class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

    def predict_(self, x):
        return
    
    def loss_elem_(self, y, y_hat):
        return
    
    def loss_(self, y, y_hat):
        return
    
    def fit_(self, x, y):
        return

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
    print("predict : \n", mylr.predict_(X))
    print("Expected : array([[0.99930437],\n\t\t[1. ],\n\t\t[1. ]])")

    print("============= 2 ===================")
    print("loss : \n", mylr.loss_(X,Y))
    print("Expected : 11.513157421577002")

    print("============= 3 ===================")
    mylr.fit_(X, Y)
    print("thetas trained : \n", mylr.theta)
    print("Expected : array([[ 2.11826435]\n\t\t[ 0.10154334]\n\t\t[ 6.43942899]\n\t\t[-5.10817488]\n\t\t[ 0.6212541 ]])")

    print("============= 4 ===================")
    print("predict : \n", mylr.predict_(X))
    print("Expected : array([[0.57606717]\n\t\t[0.68599807]\n\t\t[0.06562156]])")

    print("============= 5 ===================")
    print("loss : \n", mylr.loss_(X,Y))
    print("Expected : 1.4779126923052268")

    return 0

if __name__ == "__main__":
    main()