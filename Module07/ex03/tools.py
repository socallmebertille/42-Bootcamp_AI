import numpy as np

def add_intercept(x):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array. x can be a one-dimensional (m * 1) or two-dimensional (m * n) array.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
        x = x.reshape(-1, 1) # dim(m,n) -> m = -1 => NumPy will calculate this number for you
    ones = np.ones((x.shape[0], 1)) # mat_of_1((m,n)) -> coloumn of 1s of x_nb_line
    return np.concatenate((ones, x), axis=1) # axis = 1 => concat horizontalement (0 => vert. / None => tout sur une ligne)

def main():
    """Tester of my intercept function"""

    print("============= TEST  ===================")

    x = np.arange(1,6)
    print("x arange :\n", x)
    print("intercept : \n", add_intercept(x))
    print("Expected : array([[1., 1.],\n\t\t[1., 2.],\n\t\t[1., 3.],\n\t\t[1., 4.],\n\t\t[1., 5.]])")

    y = np.arange(1,10).reshape((3,3))
    print("y arange : \n", y)
    print("intercept : \n", add_intercept(y))
    print("Expected : array([[1., 1., 2., 3.],\n\t\t[1., 4., 5., 6.],\n\t\t[1., 7., 8., 9.]])")

    return 0

if __name__ == "__main__":
    main()