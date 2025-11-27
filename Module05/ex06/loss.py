import numpy as np

def loss_elem_(y, y_hat):
    """
    Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
        y: has to be an numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
    Returns:
        J_elem: numpy.array, a array of dimension (number of the training examples, 1).
        None if there is a dimension matching problem.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.ndim != 2 or y_hat.ndim != 2 or y.shape != y_hat.shape:
        return None
    return ((y_hat - y) ** 2).astype(float)

def loss_(y, y_hat):
    """
    Description:
        Calculates the value of loss function.
    Args:
        y: has to be an numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
    Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.ndim != 2 or y_hat.ndim != 2 or y.shape != y_hat.shape:
        return None
    return float(np.sum(loss_elem_(y, y_hat)) / (2 * y.shape[0]))

def predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a one-dimensional array of size m.
        theta: has to be an numpy.array, a two-dimensional array of shape 2 * 1.
    Returns:
        y_hat as a numpy.array, a two-dimensional array of shape m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x.size == 0 or theta.size == 0:
        return None
    if (x.ndim > 1 and x.shape[1] != 1) or theta.shape != (2,1):
        return None
    y_hat = theta[0] + theta[1] * x.reshape(-1, 1)  # reshape x pour avoir m * 1
    return y_hat.astype(float)

def main():
    """Tester of my loss function"""

    print("============= TEST 1 ===================")

    x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
    print("x1 array : \n", x1)
    print("y1 array : \n", y1)

    theta1 = np.array([[2.], [4.]])
    print("theta1 : \n", theta1)
    y_hat1 = predict(x1, theta1)
    print("y_hat1 : \n", y_hat1)

    print("loss array : \n", loss_elem_(y1, y_hat1))
    print("Expected : array([[0.], [1], [4], [9], [16]])")

    print("loss : ", loss_(y1, y_hat1))
    print("Expected : 3.0")
    
    print("============= TEST 2 ===================")

    x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
    y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)
    print("x2 array : \n", x2)
    print("y2 array : \n", y2)

    theta2 = np.array(np.array([[0.], [1.]]))
    print("theta2 : \n", theta2)
    y_hat2 = predict(x2, theta2)
    print("y_hat2 : \n", y_hat2)

    print("loss : ", loss_(y2, y_hat2))
    print("Expected : 2.142857142857143")

    print("loss : ", loss_(y2, y2))
    print("Expected : 0.0")

    return 0

if __name__ == "__main__":
    main()
