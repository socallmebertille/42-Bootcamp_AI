import numpy as np

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
    if not isinstance(x, np.ndarray):
        return None
    if x.size == 0:
        return None
    if x.ndim != 2 or x.shape[1] != 1:
        return None
    return 

def main():
    """Tester of my function who add polynomial features to x"""

    print("============= TEST ===================")

    print("============= 1 ===================")
    x = np.arange(1,11).reshape(5, 2)
    print("x array : \n", x)
    print("sigmoid : \n", add_polynomial_features(x, 3))
    print("Expected : array([[ 1, 2, 1, 4, 1, 8], \
                    \n\t\t[ 3, 4, 9, 16, 27, 64], \
                    \n\t\t[ 5, 6, 25, 36, 125, 216], \
                    \n\t\t[ 7, 8, 49, 64, 343, 512], \
                    \n\t\t[ 9, 10, 81, 100, 729, 1000]])")

    print("============= 2 ===================")
    x = np.array([[2]])
    print("x array : \n", x)
    print("sigmoid : \n", add_polynomial_features(x, 4))
    print("Expected : array([[ 1, 2, 1, 4, 1, 8, 1, 16], \
                        \n\t\t[ 3, 4, 9, 16, 27, 64, 81, 256], \
                        \n\t\t[ 5, 6, 25, 36, 125, 216, 625, 1296], \
                        \n\t\t[ 7, 8, 49, 64, 343, 512, 2401, 4096], \
                        \n\t\t[ 9, 10, 81, 100, 729, 1000, 6561, 10000]]")

    return 0

if __name__ == "__main__":
    main()
