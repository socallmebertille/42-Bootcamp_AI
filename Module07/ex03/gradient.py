import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Module05.ex03.tools import add_intercept

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
        The gradient as a numpy.array, a vector of dimensions n * 1,
        containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible dimensions.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.ndim != 2 or y.ndim != 2 or theta.ndim != 2:
        return None
    if theta.shape[1] != 1 or y.shape[1] != 1 or x.shape[1] + 1 != theta.shape[0] or x.shape[0] != y.shape[0]:
        return None
    y = y.reshape(-1, 1)
    X = add_intercept(x)
    return (X.T @ (X @ theta - y)) / y.shape[0]

def main():
    """Tester of my gradient function"""

    print("============= TEST  ===================")

    x = np.array([
                    [ -6, -7, -9],
                    [ 13, -2, 14],
                    [ -7, 14, -1],
                    [ -8, -4, 6],
                    [ -5, -9, 6],
                    [ 1, -5, 11],
                    [ 9, -11, 8]])
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

    print("x array : \n", x)
    print("y array : \n", y)

    print("============= 1 ===================")

    theta1 = np.array([0, 3, 0.5, -6]).reshape((-1, 1))
    print("theta1: \n", theta1)
    print("gradient : \n", gradient(x, y, theta1))
    print("Expected : array([[ -33.71428571], [ -37.35714286], [183.14285714], [-393.]])")
    
    print("============= 2 ===================")

    theta2 = np.array([0, 0, 0, 0]).reshape((-1, 1))
    print("theta2: \n", theta2)
    print("gradient : \n", gradient(x, y, theta2))
    print("Expected : array([[ -0.71428571], [ 0.85714286], [23.28571429], [-26.42857143]])")
    return 0

if __name__ == "__main__":
    main()