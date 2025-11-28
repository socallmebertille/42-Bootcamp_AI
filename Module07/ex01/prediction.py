import numpy as np

def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimensions m * n.
        theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
    Return:
        y_hat as a numpy.array, a vector of dimensions m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
        None if x or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x.size == 0 or theta.size == 0:
        return None
    if x.ndim != 2 or theta.ndim != 2:
        return None
    if theta.shape[1] != 1 or x.shape[1] + 1 != theta.shape[0]:
        return None
    m = x.shape[0]
    X_prime = np.hstack([np.ones((m, 1)), x]) # concatenation de colonnes
    y_hat = X_prime @ theta
    return y_hat

def main():
    """Tester of my predict function"""

    print("============= TEST ===================")

    x = np.arange(1,13).reshape((4,-1))
    print("x array : \n", x)

    print("============= 1 ===================")
    theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
    print("theta1 : \n", theta1)

    print("predict : \n", predict_(x, theta1))
    print("Expected : array([[5.], [5.], [5.], [5.]])")

    print("============= 2 ===================")
    theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
    print("theta2 : \n", theta2)

    print("predict : \n", predict_(x, theta2))
    print("Expected : array([[ 1.], [ 4.], [ 7.], [10.]])")

    print("============= 3 ===================")
    theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
    print("theta3 : \n", theta3)

    print("predict : \n", predict_(x, theta3))
    print("Expected : array([[ 9.64], [24.28], [38.92], [53.56]])")

    print("============= 4 ===================")
    theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
    print("theta4 : \n", theta4)

    print("predict : \n", predict_(x, theta4))
    print("Expected : array([[12.5], [32. ], [51.5], [71. ]])")

    return 0

if __name__ == "__main__":
    main()