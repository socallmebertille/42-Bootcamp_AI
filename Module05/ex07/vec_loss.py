import numpy as np

def loss_(y, y_hat):
    """Computes the half mean-squared-error of two non-empty numpy.arrays, without any for loop.
    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a one-dimensional array of size m.
        y_hat: has to be an numpy.array, a one-dimensional array of size m.
    Returns:
        The half mean-squared-error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.ndim != 1 or y_hat.ndim != 1 or y.shape != y_hat.shape:
        return None
    
    return np.sum((y_hat - y) ** 2) / (2 * y.shape[0])

def main():
    """Tester of my loss function"""

    print("============= TEST ===================")

    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    print("X array : ", X)
    print("Y array : ", Y)

    print("loss : ", loss_(X, Y))
    print("Expected : 2.142857142857143")

    print("loss : ", loss_(X, X))
    print("Expected : 0.0")

    return 0

if __name__ == "__main__":
    main()
