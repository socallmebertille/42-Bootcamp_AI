"""
ridge.py - Logistic Ridge Regression
"""
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ex01.l2_reg import iterative_l2, l2
from ex03.logistic_loss_reg import reg_log_loss_
from ex05.reg_logistic_grad import vec_reg_logistic_grad, reg_logistic_grad
from ex08.my_logistic_regression import MyLogisticRegression

class MyRidge(MyLogisticRegression):
    """
    Description:
        My personnal logistic ridge regression class to fit like a boss.
        Inherits from MyLogisticRegression and adds proper L2 regularization support.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000, lambda_=1.0, penalty=None):
        super().__init__(theta, alpha, max_iter, penalty, lambda_)
        if isinstance(theta, np.ndarray):
            self.theta = theta.reshape(-1, 1)
        else:
            print("Error\nThetas array is not from the numpy library")
        self.penality = penalty if penalty == None or penalty == "l2" else None
        self.alpha = alpha
        self.max_iter = max_iter
        self.lambda_ = lambda_

    def get_params_(self):
        """Returns the parameters of the estimator"""
        return {
            'theta': self.theta,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'lambda_': self.lambda_
        }
    
    def set_params_(self, **kwargs):
        """Sets the parameters of the estimator"""
        if 'theta' in kwargs:
            self.theta = kwargs['theta']
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        if 'max_iter' in kwargs:
            self.max_iter = kwargs['max_iter']
        if 'lambda_' in kwargs:
            self.lambda_ = kwargs['lambda_']
        return self

    def predict_(self, x):
        """Inherited from MyLogisticRegression - uses sigmoid"""
        return super().predict_(x)
    
    def loss_elem_(self, y, y_hat):
        """
        Returns element-wise logistic loss (cross-entropy)
        NOT regularized - just the base loss elements
        """
        return super().loss_elem_(y, y_hat)
    
    def loss_(self, y, y_hat):
        """
        Returns the regularized logistic loss (Cross-entropy + L2 regularization)
        """
        return super().loss_(y, y_hat)
    
    def gradient_(self, x, y):
        """
        Computes the regularized logistic gradient
        IMPORTANT: Check the parameter order expected by vec_reg_logistic_grad!
        """
        return super().gradient_(x, y)
    
    def fit_(self, x, y):
        """
        Fits the logistic ridge regression model using gradient descent
        """
        self.theta = super().fit_(x, y)
        return self.theta


def main():
    """Test Logistic Ridge Regression"""

    # valeurs expected encore erronnees mais avec lambda 0.0 on s'y rapproche le +

    print("============= TEST ===================")

    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    
    mylr = MyRidge(thetas, alpha=0.001, max_iter=10000, lambda_=0.0)

    print("X array : \n", X)
    print("Y array : \n", Y)
    print("Initial theta : \n", thetas)

    print("\n============= 1 - Initial Prediction ===================")
    y_hat = mylr.predict_(X)
    print("predict : \n", y_hat)
    print("Expected : array([[0.99930437],\n\t\t[1. ],\n\t\t[1. ]])")

    print("\n============= 2 - Initial Loss ===================")
    loss = mylr.loss_(Y, y_hat)
    print("loss : ", loss)
    print("Expected : 11.513157421577002")

    print("\n============= 3 - Training ===================")
    print("Training with alpha={}, max_iter={}, lambda_={}...".format(
        mylr.alpha, mylr.max_iter, mylr.lambda_))
    mylr.fit_(X, Y)
    print("thetas trained : \n", mylr.theta)
    print("Expected : array([[ 2.11826435]\n\t\t[ 0.10154334]\n\t\t[ 6.43942899]\n\t\t[-5.10817488]\n\t\t[ 0.6212541 ]])")

    print("\n============= 4 - Prediction after training ===================")
    y_hat = mylr.predict_(X)
    print("predict : \n", y_hat)
    print("Expected : array([[0.57606717]\n\t\t[0.68599807]\n\t\t[0.06562156]])")

    print("\n============= 5 - Final Loss ===================")
    loss = mylr.loss_(Y, y_hat)
    print("loss : ", loss)
    print("Expected : 1.4779126923052268")
    
    print("\n============= Test get_params_ / set_params_ ===================")
    params = mylr.get_params_()
    print("Parameters:", params)
    
    mylr.set_params_(alpha=0.01, lambda_=2.0)
    print("After setting alpha=0.01, lambda_=2.0:")
    print("New alpha:", mylr.alpha, "New lambda:", mylr.lambda_)

    return 0

if __name__ == "__main__":
    main()