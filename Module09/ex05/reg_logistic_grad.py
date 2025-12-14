import numpy as np

def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, with two for-loops.
    The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """

def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, without
    any for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of shape m * n.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
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
    """Tester of my gradient function of a logistic regression model"""

    print("============= TEST ===================")
    x = np.array([[0, 2, 3, 4],
[2, 4, 5, 5],
[1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print("x array : \n", x)
    print("y array : \n", y)
    print("theta : \n", theta)

    print("============= 1.1 ===================")
    print("gradient w/ lambda 1 : \n", reg_logistic_grad(y, x, theta, 1))
    print("Expected : array([[-0.55711039],\n\t\t[-1.40334809],\n\t\t[-1.91756886],\n\t\t[-2.56737958],\n\t\t[-3.03924017]])")

    print("============= 1.2 ===================")
    print("vector gradient w/ lambda 1 : \n", vec_reg_logistic_grad(y, x, theta, 1))
    print("Expected : array([[-0.55711039],\n\t\t[-1.40334809],\n\t\t[-1.91756886],\n\t\t[-2.56737958],\n\t\t[-3.03924017]])")

    print("============= 2.1 ===================")
    print("gradient w/ lambda 0.5 : \n", reg_logistic_grad(y, x, theta, 0.5))
    print("Expected : array([[-0.55711039],\n\t\t[-1.15334809],\n\t\t[-1.96756886],\n\t\t[-2.33404624],\n\t\t[-3.15590684]])")

    print("============= 2.2 ===================")
    print("vector gradient w/ lambda 0.5 : \n", vec_reg_logistic_grad(y, x, theta, 0.5))
    print("Expected : array([[-0.55711039],\n\t\t[-1.15334809],\n\t\t[-1.96756886],\n\t\t[-2.33404624],\n\t\t[-3.15590684]])")

    print("============= 3.1 ===================")
    print("gradient w/ lambda 0.0 : \n", reg_logistic_grad(y, x, theta, 0.0))
    print("Expected : array([[-0.55711039],\n\t\t[-0.90334809],\n\t\t[-2.01756886],\n\t\t[-2.10071291],\n\t\t[-3.27257351]])")

    print("============= 3.2 ===================")
    print("vector gradient w/ lambda  0.0: \n", vec_reg_logistic_grad(y, x, theta, 0.0))
    print("Expected : array([[-0.55711039],\n\t\t[-0.90334809],\n\t\t[-2.01756886],\n\t\t[-2.10071291],\n\t\t[-3.27257351]])")

    return 0

if __name__ == "__main__":
    main()