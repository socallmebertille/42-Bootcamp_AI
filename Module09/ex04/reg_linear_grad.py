import numpy as np

def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
    with two for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """

def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.ndim != 2 or y.ndim != 2 or theta.ndim != 2:
        return None
    if theta.shape[1] != 1 or y.shape[1] != 1 or x.shape[1] + 1 != theta.shape[0] or x.shape[0] != y.shape[0]:
        return None

    return 

def main():
    """Tester of my gradient function of a linear regression model"""

    print("============= TEST ===================")
    x = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[ -8, -4, 6],
[ -5, -9, 6],
[ 1, -5, 11],
[ 9, -11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])
    print("x array : \n", x)
    print("y array : \n", y)
    print("theta : \n", theta)

    print("============= 1.1 ===================")
    print("gradient w/ lambda 1 : \n", reg_linear_grad(y, x, theta, 1))
    print("Expected : array([[ -60.99 ], \n\t\t[-195.64714286],\n\t\t[ 863.46571429],\n\t\t[-644.52142857]])")

    print("============= 1.2 ===================")
    print("vector gradient w/ lambda 1 : \n", vec_reg_linear_grad(y, x, theta, 1))
    print("Expected : array([[ -60.99 ],\n\t\t[-195.64714286],\n\t\t[ 863.46571429],\n\t\t[-644.52142857]])")

    print("============= 2.1 ===================")
    print("gradient w/ lambda 0.5 : \n", reg_linear_grad(y, x, theta, 0.5))
    print("Expected : array([[ -60.99 ],\n\t\t[-195.86142857],\n\t\t[ 862.71571429],\n\t\t[-644.09285714]])")

    print("============= 2.2 ===================")
    print("vector gradient w/ lambda 0.5 : \n", vec_reg_linear_grad(y, x, theta, 0.5))
    print("Expected : array([[ -60.99 ],\n\t\t[-195.86142857],\n\t\t[ 862.71571429],\n\t\t[-644.09285714]])")

    print("============= 3.1 ===================")
    print("gradient w/ lambda 0.0 : \n", reg_linear_grad(y, x, theta, 0.0))
    print("Expected : array([[ -60.99 ],\n\t\t[-196.07571429],\n\t\t[ 861.96571429],\n\t\t[-643.66428571]])")

    print("============= 3.2 ===================")
    print("vector gradient w/ lambda  0.0: \n", vec_reg_linear_grad(y, x, theta, 0.0))
    print("Expected : array([[ -60.99 ],\n\t\t[-196.07571429],\n\t\t[ 861.96571429],\n\t\t[-643.66428571]]")

    return 0

if __name__ == "__main__":
    main()