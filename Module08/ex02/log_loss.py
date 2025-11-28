import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ex01.log_pred import logistic_predict_

def log_loss_elem_(y, y_hat, eps=1e-15):
    """Computes the logistic loss element-wise.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
    Returns:
        The logistic loss element-wise as a numpy.ndarray, a vector of shape m * 1.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.ndim != 2 or y_hat.ndim != 2:
        return None
    if y.shape != y_hat.shape:
        return None
    return y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)

def log_loss_(y, y_hat, eps=1e-15):
    """Computes the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.ndim != 2 or y_hat.ndim != 2:
        return None
    if y.shape != y_hat.shape:
        return None
    m = y.shape[0]
    return - (1 / m) * np.sum(log_loss_elem_(y, y_hat, eps))

def main():
    """Tester of my loss function"""

    print("============= TEST ===================")

    print("============= 1 ===================")
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    y_hat1 = logistic_predict_(x1, theta1)
    print("x array : \n", x1)
    print("theta : \n", theta1)
    print("predict : \n", y_hat1)
    print("loss : \n", log_loss_(y1, y_hat1))
    print("Expected : 0.01814992791780973")

    print("============= 2 ===================")
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat2 = logistic_predict_(x2, theta2)
    theta2 = np.array([[2], [0.5]])
    print("x array : \n", x2)
    print("theta : \n", theta2)
    print("predict : \n", y_hat2)
    print("loss : \n", log_loss_(y2, y_hat2))
    print("Expected : 2.4825011602474483")

    print("============= 3 ===================")
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    y_hat3 = logistic_predict_(x3, theta3)
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print("x array : \n", x3)
    print("theta : \n", theta3)
    print("predict : \n", y_hat3)
    print("loss : \n", log_loss_(y3, y_hat3))
    print("Expected : 2.9938533108607053")

    return 0

if __name__ == "__main__":
    main()
