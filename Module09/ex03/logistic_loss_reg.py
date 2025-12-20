import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ex01.l2_reg import l2

def reg_log_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray,
    without any for loop. The two arrays must have the same shapes.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta is empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or not isinstance(theta, np.ndarray) or not isinstance(lambda_, float):
        return None
    if y.size == 0 or y_hat.size == 0 or theta.size == 0:
        return None
    if y.ndim != 2 or y_hat.ndim != 2 or theta.ndim != 2:
        return None
    if y.shape != y_hat.shape or y.shape[1] != 1 or theta.shape[1] != 1:
        return None
    m = y.shape[0]
    one_vec = np.ones((m, 1))
    loss_function = - (((y.T @ np.log(y_hat)) + ((one_vec - y).T @ (np.log(one_vec - y_hat)))) / m)
    biais = ((lambda_ * l2(theta)) / (2 * m))
    return (loss_function + biais).item()

def main():
    """Tester of my regularized loss function of a logistic regression model"""

    print("============= TEST ===================")

    y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
    print("y array : \n", y)
    print("predict : \n", y_hat)
    print("theta : \n", theta)

    print("============= 1 ===================")
    print("loss w/ lambda: .5: \n", reg_log_loss_(y, y_hat, theta, .5))
    print("Expected : 0.43377043716475955")

    print("============= 2 ===================")
    print("loss w/ lambda: .05 : \n", reg_log_loss_(y, y_hat, theta, .05))
    print("Expected : 0.13452043716475953")

    print("============= 3 ===================")
    print("loss w/ lambda: .9 : \n", reg_log_loss_(y, y_hat, theta, .9))
    print("Expected : 0.6997704371647596")

    return 0

if __name__ == "__main__":
    main()