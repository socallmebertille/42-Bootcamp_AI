import numpy as np

def sigmoid_(x):
    """Compute the sigmoid of a vector.
    Args:
        x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x.size == 0:
        return None
    if x.ndim != 2:
        return None
    return 

def main():
    """Tester of my sigmoid function"""

    print("============= TEST ===================")

    print("============= 1 ===================")
    x = np.array([[-4]])
    print("x array : \n", x)
    print("sigmoid : \n", sigmoid_(x))
    print("Expected : array([[0.01798620996209156]])")

    print("============= 2 ===================")
    x = np.array([[2]])
    print("x array : \n", x)
    print("sigmoid : \n", sigmoid_(x))
    print("Expected : array([[0.8807970779778823]])")

    print("============= 3 ===================")
    x = np.array([[-4], [2], [0]])
    print("x array : \n", x)
    print("sigmoid : \n", sigmoid_(x))
    print("Expected : array([[0.01798620996209156], [0.8807970779778823], [0.5]])")

    return 0

if __name__ == "__main__":
    main()