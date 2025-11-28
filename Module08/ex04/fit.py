import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Module08.ex01.log_pred import predict_
from Module08.ex03.vec_log_loss import gradient

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a matrix of dimension m * n:
        (number of training examples, number of features).
        y: has to be a numpy.array, a vector of dimension m * 1:
        (number of training examples, 1).
        theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
        (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
        new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if not isinstance(alpha, (int, float)) or not isinstance(max_iter, int):
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.ndim != 2 or y.ndim != 2 or theta.ndim != 2:
        return None
    if theta.shape[1] != 1 or y.shape[1] != 1 or x.shape[1] + 1 != theta.shape[0] or x.shape[0] != y.shape[0]:
        return None
    new_theta = theta.copy()
    for _ in range(max_iter):
        grad = gradient(x, y, new_theta)
        if grad is None:
            return None
        new_theta = new_theta - alpha * grad
    return new_theta.astype(float)

def main():
    """Tester of my fit function"""

    print("============= TEST  ===================")

    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])

    print("x array : \n", x)
    print("y array : \n", y)

    print("============= 1 ===================")

    theta = np.array([[42.], [1.], [1.], [1.]])
    print("theta1: \n", theta)
    
    print("============= 2 ===================")

    theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter=42000)
    print("theta2: \n", theta2)
    print("Expected : array([[41.99..],[0.97..], [0.77..], [-1.20..]])")

    print("predict: \n", predict_(x, theta2))
    print("Expected : array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])")

    return 0

if __name__ == "__main__":
    main()