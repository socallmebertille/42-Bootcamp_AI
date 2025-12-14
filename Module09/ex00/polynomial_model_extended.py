import numpy as np
from itertools import combinations_with_replacement

def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising its columns to every power in the range
    of 1 up to the power given in argument.
    Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        power: has to be an int, the power up to which the columns of matrix x are going
        to be raised.
    Returns:
        The matrix of polynomial features as a numpy.ndarray, of shape m * (np),
        containg the polynomial feature values for all
        training examples.
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        return None
    if not isinstance(power, int) or power < 1:
        return None

    m, n = x.shape
    X_poly = np.ones((m, 0), dtype=int)
    for p in range(1, power + 1):
        X_poly = np.hstack((X_poly, x ** p))
    return X_poly

def main():
    """Tester of my function who add polynomial features to x"""

    print("============= TEST ===================")
    x = np.arange(1,11).reshape(5, 2)
    print("x array : \n", x)

    print("============= 1 ===================")
    print("add polynomial features to x : \n", add_polynomial_features(x, 3))
    print("Expected : array([[ 1, 2, 1, 4, 1, 8], \
                    \n\t\t[ 3, 4, 9, 16, 27, 64], \
                    \n\t\t[ 5, 6, 25, 36, 125, 216], \
                    \n\t\t[ 7, 8, 49, 64, 343, 512], \
                    \n\t\t[ 9, 10, 81, 100, 729, 1000]])")

    print("============= 2 ===================")
    print("add polynomial features to x : \n", add_polynomial_features(x, 4))
    print("Expected : array([[ 1, 2, 1, 4, 1, 8, 1, 16], \
                        \n\t\t[ 3, 4, 9, 16, 27, 64, 81, 256], \
                        \n\t\t[ 5, 6, 25, 36, 125, 216, 625, 1296], \
                        \n\t\t[ 7, 8, 49, 64, 343, 512, 2401, 4096], \
                        \n\t\t[ 9, 10, 81, 100, 729, 1000, 6561, 10000]]")

    return 0

if __name__ == "__main__":
    main()
