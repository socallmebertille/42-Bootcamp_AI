import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Module07.ex03.gradient import add_intercept

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a one-dimensional array of size m.
        theta: has to be an numpy.array, a two-dimensional array of shape 2 * 1.
    Returns:
        y_hat as a numpy.array, a two-dimensional array of shape m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x.size == 0 or theta.size == 0:
        return None
    if x.ndim != 1 or theta.shape != (2,1):
        return None
    y_hat = add_intercept(x) @ theta # matrix multiplication operator @ => permet le produit matriciel
    return y_hat.astype(float)

def main():
    """Tester of my predict function"""

    print("============= TEST  ===================")

    x = np.arange(1,6)
    print("x arange : ", x)

    theta1 = np.array([[5], [0]])
    print("theta1 : \n", theta1)
    print("predict : \n", predict_(x, theta1))
    print("Expected : array([[5.], [5.], [5.], [5.], [5.]])")
    # Do you remember why y_hat contains only 5’s here?

    theta2 = np.array([[0], [1]])
    print("theta2 : \n", theta2)
    print("predict : \n", predict_(x, theta2))
    print("Expected : array([[1.], [2.], [3.], [4.], [5.]])")
    # Do you remember why y_hat == x here?

    theta3 = np.array([[5], [3]])
    print("theta3 : \n", theta3)
    print("predict : \n", predict_(x, theta3))
    print("Expected : array([[ 8.], [11.], [14.], [17.], [20.]])")

    theta4 = np.array([[-3], [1]])
    print("theta4 : \n", theta4)
    print("predict : \n", predict_(x, theta4))
    print("Expected : array([[-2.], [-1.], [ 0.], [ 1.], [ 2.]])")

    return 0

if __name__ == "__main__":
    main()