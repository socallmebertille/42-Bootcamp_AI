import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

def mse_(y, y_hat):
    """
    Description:
        Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be a numpy.array, a two-dimensional vector of shape m * 1.
    Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.ndim != 2 or y_hat.ndim != 2 or y.shape != y_hat.shape:
        return None
    return float(np.sum((y_hat - y) ** 2) / y.shape[0])

def rmse_(y, y_hat):
    """
    Description:
        Calculate the RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.ndim != 2 or y_hat.ndim != 2 or y.shape != y_hat.shape:
        return None
    mse = mse_(y, y_hat)
    if mse is None:
        return None
    return float(mse ** 0.5)

def mae_(y, y_hat):
    """
    Description:
        Calculate the MAE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
    Returns:
        mae: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.ndim != 2 or y_hat.ndim != 2 or y.shape != y_hat.shape:
        return None
    return float(np.sum(np.abs(y_hat - y)) / y.shape[0])

def r2score_(y, y_hat):
    """
    Description:
        Calculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
    Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.ndim != 2 or y_hat.ndim != 2 or y.shape != y_hat.shape:
        return None
    return float(1 - (np.sum((y_hat - y) ** 2) / np.sum((y - np.mean(y)) ** 2) ))

def main():
    """Tester of my functions of precision indicator"""

    print("============= TEST ===================")

    x = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    print("x array : ", x)
    print("y array : ", y)

    print("mse : ", mse_(x,y))
    print("Expected : ", mean_squared_error(x,y))

    print("rmse : ", rmse_(x,y))
    print("Expected : ", sqrt(mean_squared_error(x,y)))

    print("mae : ", mae_(x,y))
    print("Expected : ", mean_absolute_error(x,y))

    print("R^2 : ", r2score_(x,y))
    print("Expected : ", r2_score(x,y))

    return 0

if __name__ == "__main__":
    main()
