import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ex01.l2_reg import iterative_l2, l2
from ex02.linear_loss_reg import reg_loss_
from ex03.logistic_loss_reg import reg_log_loss_
from ex04.reg_linear_grad import vec_reg_linear_grad, reg_linear_grad
from ex05.reg_logistic_grad import vec_reg_logistic_grad, reg_logistic_grad
from ex06.mylinearregression import MyLinearRegression

class MyRidge(MyLinearRegression):
    """
    Description:
        My personnal ridge regression class to fit like a boss
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        self.lambda_ = lambda_

    def get_params_():
        return
    
    def set_params_():
        return

    def predict_(self, x):
        return l2(x, self.thetas)
    
    def loss_elem_(self, y, y_hat):
        return reg_loss_(y, y_hat)
    
    def loss_(self, y, y_hat):
        return reg_log_loss_(y, y_hat)
    
    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        if x.size == 0 or y.size == 0:
            return None
        new_theta = np.copy(self.thetas)
        for _ in range(self.max_iter):
            grad = vec_reg_logistic_grad(x, y, new_theta)
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
    mylr = MyRidge(thetas)

    print("X array : \n", X)
    print("Y array : \n", Y)
    print("theta : \n", thetas)

    print("============= 1 ===================")
    y_hat = mylr.predict_(X)
    print("predict : \n", y_hat)
    print("Expected : array([[0.99930437],\n\t\t[1. ],\n\t\t[1. ]])")

    print("============= 2 ===================")
    print("loss : \n", mylr.loss_(Y,y_hat))
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
    print("loss : \n", mylr.loss_(Y,y_hat))
    print("Expected : 1.4779126923052268")

    return 0

if __name__ == "__main__":
    main()