import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Module05.ex03.tools import add_intercept

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.arrays, without any for loop.
    The three arrays must have compatible shapes.
    Args:
        x: has to be a numpy.array, a vector of shape m * 1.
        y: has to be a numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        None if x, y, or theta is an empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if (x.ndim != 1 and x.shape[1] > 1) or (y.ndim != 1 and y.shape[1] > 1):
        return None
    if x.shape != y.shape or theta.shape != (2,1):
        return None
    X = add_intercept(x.reshape(-1))  # add_intercept attend 1D
    return (X.T @ (X @ theta - y)) / x.shape[0]

def main():
    """Tester of my gradient vector function optimized"""

    print("============= TEST ===================")

    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))

    print("x array : \n", x)
    print("y array : \n", y)

    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    print("theta1 : \n", theta1)
    print("gradient : \n", gradient(x, y, theta1) )
    print("Expected : array([[-19.0342574], [-586.66875564]])")

    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print("theta2 : \n", theta2)
    print("gradient : \n", gradient(x, y, theta2) )
    print("Expected : array([[-57.86823748], [-2230.12297889]])")

    return 0

if __name__ == "__main__":
    main()
