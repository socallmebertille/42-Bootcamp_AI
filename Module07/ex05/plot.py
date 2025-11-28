import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a one-dimensional array of size m.
        y: has to be an numpy.array, a one-dimensional array of size m.
        theta: has to be an numpy.array, a two-dimensional array of shape 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return
    if x.ndim != 1 or y.ndim != 1 or theta.shape != (2, 1):
        return
    plt.scatter(x, y, color='blue', label='Data points')
    x_line = np.array([x.min(), x.max()])
    y_line = theta[0][0] + theta[1][0] * x_line
    plt.plot(x_line, y_line, color="red", label="Prediction line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def main():
    """Tester of my plot function"""

    print("============= TEST  ===================")

    x = np.arange(1,6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
    print("x arange : ", x)
    print("y array : ", y)

    theta1 = np.array([[4.5],[-0.2]])
    print("theta1 : \n", theta1)
    plot(x, y, theta1)
    
    theta2 = np.array([[-1.5],[2]])
    print("theta2 : \n", theta2)
    plot(x, y, theta2)

    theta3 = np.array([[3],[0.3]])
    print("theta3 : \n", theta3)
    plot(x, y, theta3)
    return 0

if __name__ == "__main__":
    main()
