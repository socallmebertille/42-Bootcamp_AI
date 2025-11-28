import numpy as np

def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
        The matrix of polynomial features as a numpy.array, of dimension m * n,
        containing the polynomial feature values for all training examples.
        None if x is an empty numpy.array.
        None if x or power is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(power, int):
        return None
    if x.size == 0 or y <= 0:
        return None
    if x.ndim != 1:
        return None
    
    return 

def main():
    """Tester of my loss function"""

    print("============= TEST ===================")

    x = np.arange(1,6).reshape(-1, 1)
    print("x array : \n", x)

    print("add_polynomial_features : \n", add_polynomial_features(x, 3))
    print("Expected : array([[ 1, 1, 1],\n\t\t[ 2, 4, 8],\n\t\t[ 3, 9, 27],\n\t\t[ 4, 16, 64],\n\t\t[ 5, 25, 125]])")

    print("add_polynomial_features : \n", add_polynomial_features(x, 6))
    print("Expected : array([[ 1, 1, 1, 1, 1, 1],\n\t\t[ 2, 4, 8, 16, 32, 64],\n\t\t[ 3, 9, 27, 81, 243, 729],\n\t\t[ 4, 16, 64, 256, 1024, 4096],\n\t\t[ 5, 25, 125, 625, 3125, 15625]])")

    return 0

if __name__ == "__main__":
    main()
