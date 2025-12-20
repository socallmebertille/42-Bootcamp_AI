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
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray) or not isinstance(lambda_, float):
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.ndim != 2 or y.ndim != 2 or theta.ndim != 2:
        return None
    if x.shape[0] != y.shape[0] or y.shape[1] != 1 or theta.shape != (x.shape[1] + 1 , 1):
        return None
    x_prime = np.zeros((x.shape[0], x.shape[1] + 1))
    for l in range(x_prime.shape[0]):
        for c in range(x_prime.shape[1]):
            x_prime[l, c] = 1 if c == 0 else x[l, c - 1]
    y_hat = np.zeros((x_prime.shape[0], 1))
    for i in range(x_prime.shape[0]):
        sum_val = 0
        for j in range(x_prime.shape[1]):
            sum_val += x_prime[i, j] * theta[j, 0]
        y_hat[i, 0] = sum_val
    gradient = np.zeros(theta.shape)
    for j in range(theta.shape[0]):
        sum_grad = 0
        for i in range(x_prime.shape[0]):
            sum_grad += (y_hat[i, 0] - y[i, 0]) * x_prime[i, j]
        if j == 0:
            gradient[j, 0] = sum_grad / y.shape[0]
        else:
            gradient[j, 0] = (sum_grad + lambda_ * theta[j, 0]) / y.shape[0]
    return gradient

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
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray) or not isinstance(lambda_, float):
        return None
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    if x.ndim != 2 or y.ndim != 2 or theta.ndim != 2:
        return None
    if x.shape[0] != y.shape[0] or y.shape[1] != 1 or theta.shape != (x.shape[1] + 1 , 1):
        return None
    x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
    y_hat = x_prime @ theta
    return (x_prime.T @ (y_hat - y) + (lambda_ * theta)) / y.shape[0]

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