import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ex01.l2_reg import l2

def reg_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a linear regression model from two non-empty numpy.array,
    without any for loop. The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta are empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
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
    return

def main():
    """Tester of my regularized loss function of a linear regression model"""

    print("============= TEST ===================")

    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
    print("y array : \n", y)
    print("predict : \n", y_hat)
    print("theta : \n", theta)

    print("============= 1 ===================")
    print("loss w/ lambda: .5: \n", reg_loss_(y, y_hat, theta, .5))
    print("Expected : 0.8503571428571429")

    print("============= 2 ===================")
    print("loss w/ lambda: .05 : \n", reg_loss_(y, y_hat, theta, .05))
    print("Expected : 0.5511071428571429")

    print("============= 3 ===================")
    print("loss w/ lambda: .9 : \n", reg_loss_(y, y_hat, theta, .9))
    print("Expected : 1.116357142857143")

    return 0

if __name__ == "__main__":
    main()
