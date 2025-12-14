import numpy as np

def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(theta, np.ndarray):
        return None
    if theta.size == 0:
        return None
    if theta.ndim != 2:
        return None
    if theta.shape[1] != 1 or theta.shape[0]:
        return None
    return 

def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(theta, np.ndarray):
        return None
    if theta.size == 0:
        return None
    if theta.ndim != 2:
        return None
    if theta.shape[1] != 1 or theta.shape[0]:
        return None
    return 

def main():
    """Tester of my l2 reg functions"""

    print("============= TEST ===================")

    print("============= 1 ===================")
    x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print("x array : \n", x)
    print("iterative_l2: \n", iterative_l2(x))
    print("Expected : 911.0")

    print("============= 2 ===================")
    print("l2 : \n", l2(x))
    print("Expected : 911.0")

    print("============= 3 ===================")
    y = np.array([3,0.5,-6]).reshape((-1, 1))
    print("y array : \n", y)
    print("iterative_l2 : \n", iterative_l2(y))
    print("Expected : 36.25")

    print("============= 4 ===================")
    print("l2 : \n", l2(y))
    print("Expected : 36.25")

    return 0

if __name__ == "__main__":
    main()