import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Module05.ex06.loss import predict

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.ndim < 1 or y.ndim < 1 or x.shape != y.shape or theta.shape != (2,1):
        return None
    return 

def main():
    """Tester of my functions of precision indicator"""

    print("============= TEST ===================")

    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))

    print("x array : ", x)
    print("y array : ", y)

    theta= np.array([1, 1]).reshape((-1, 1))
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print("theta1 : \n", theta1)
    print("Expected : array([[1.40709365], [1.1150909 ]])")

    print("predict : \n", predict(x, theta1))
    print("Expected : array([[15.3408728 ],\n\t\t[25.38243697],\n\t\t[36.59126492],\n\t\t[55.95130097],\n\t\t[65.53471499]])")

    return 0

if __name__ == "__main__":
    main()
