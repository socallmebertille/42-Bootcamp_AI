import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ex01.l2_reg import iterative_l2, l2
from ex02.linear_loss_reg import reg_loss_
from ex04.reg_linear_grad import vec_reg_linear_grad, reg_linear_grad
from ex06.mylinearregression import MyLinearRegression

class MyRidge(MyLinearRegression):
    """
    Description:
        My personnal ridge regression class to fit like a boss
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        super().__init__(thetas, alpha, max_iter)
        self.lambda_ = lambda_

    def get_params_(self):
        """Returns the parameters of the estimator"""
        return {
            'thetas': self.thetas,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'lambda_': self.lambda_
        }
    
    def set_params_(self, **kwargs):
        """Sets the parameters of the estimator"""
        if 'thetas' in kwargs:
            self.thetas = kwargs['thetas']
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        if 'max_iter' in kwargs:
            self.max_iter = kwargs['max_iter']
        if 'lambda_' in kwargs:
            self.lambda_ = kwargs['lambda_']
        return self

    def predict_(self, x):
        return super().predict_(x)
    
    def loss_elem_(self, y, y_hat):
        """
        Returns element-wise squared differences (loss elements)
        This is NOT the regularized version - just the base loss elements
        """
        return super().loss_elem_(y, y_hat)
    
    def loss_(self, y, y_hat):
        """
        Returns the regularized loss (MSE + L2 regularization)
        """
        return reg_loss_(y, y_hat, self.thetas, self.lambda_)
    
    def gradient_(self, x, y):
        """
        Computes the regularized gradient
        """
        return vec_reg_linear_grad(x, y, self.thetas, self.lambda_)
    
    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        if x.size == 0 or y.size == 0:
            return None
        new_theta = np.copy(self.thetas)
        for _ in range(self.max_iter):
            grad = vec_reg_linear_grad(x, y, new_theta, self.lambda_)
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