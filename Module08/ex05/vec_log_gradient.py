import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ex01.log_pred import logistic_predict_

def vec_log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have comp
    Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector (n +1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible shapes.
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
    m = x.shape[0]
    X_prime = np.hstack((np.ones((m, 1)), x))
    y_hat = logistic_predict_(x, theta)
    return (1 / m) * (X_prime.T @ (y_hat - y))

def main():
    """Tester of my gradient function"""

    print("============= TEST ===================")

    print("============= 1 ===================")
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    print("x array : \n", x1)
    print("theta : \n", theta1)
    print("gradient : \n", vec_log_gradient(x1, y1, theta1))
    print("Expected : array([[-0.01798621],\n\t\t[-0.07194484]])")

    print("============= 2 ===================")
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print("x array : \n", x2)
    print("theta : \n", theta2)
    print("gradient : \n", vec_log_gradient(x2, y2, theta2))
    print("Expected : array([[0.3715235],\n\t\t[3.25647547]])")

    print("============= 3 ===================")
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print("x array : \n", x3)
    print("theta : \n", theta3)
    print("gradient : \n", vec_log_gradient(x3, y3, theta3))
    print("Expected : array([[-0.55711039],\n\t\t[-0.90334809],\n\t\t[-2.01756886],\n\t\t[-2.10071291],\n\t\t[-3.27257351]])")

    return 0

if __name__ == "__main__":
    main()