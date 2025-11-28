import numpy as np

def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
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
    return 

def main():
    """Tester of my predict function"""

    print("============= TEST ===================")

    print("============= 1 ===================")
    x = np.array([4]).reshape((-1, 1))
    print("x array : \n", x)
    theta = np.array([[2], [0.5]])
    print("theta : \n", theta)
    print("predict : \n", logistic_predict_(x, theta))
    print("Expected : array([[0.01798620996209156]])")

    print("============= 2 ===================")
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    print("x array : \n", x2)
    theta2 = np.array([[2], [0.5]])
    print("theta : \n", theta2)
    print("predict : \n", logistic_predict_(x2, theta2))
    print("Expected : array([[0.8807970779778823]])")

    print("============= 3 ===================")
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    print("x array : \n", x3)
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print("theta : \n", theta3)
    print("predict : \n", logistic_predict_(x3, theta3))
    print("Expected : array([[0.01798620996209156], [0.8807970779778823], [0.5]])")

    return 0

if __name__ == "__main__":
    main()