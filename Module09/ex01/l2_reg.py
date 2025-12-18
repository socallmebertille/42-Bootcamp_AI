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
    l2_sum = 0.0
    for j in range(1, theta.shape[0]):  # on saute theta[0]
        l2_sum += theta[j, 0] ** 2
    return float(l2_sum)

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
    theta_prime = theta.copy()
    theta_prime[0, 0] = 0
    return float((theta_prime.T @ theta_prime).item()) # item() : recupere la valeur scalaire de la mutiplication de 2 vecteur place au sein d'un tableau => equivaent a np[0, 0]

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