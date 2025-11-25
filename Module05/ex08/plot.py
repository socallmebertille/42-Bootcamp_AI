import numpy as np
import matplotlib.pyplot as plt

def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, one-dimensional array of size m.
        y: has to be an numpy.ndarray, one-dimensional array of size m.
        theta: has to be an numpy.ndarray, one-dimensional array of size 2.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return
    if x.ndim != 1 or y.ndim != 1 or theta.size != 2:
        return
    y_hat = theta[0] + theta[1] * x
    loss = np.sum((y_hat - y) * (y_hat - y)) / (y.shape[0])
    plt.scatter(x, y, color='blue', label='Data points')
    x_line = np.array([x.min(), x.max()])
    y_line = theta[0] + theta[1] * x_line
    plt.plot(x_line, y_line, color='red', label='Prediction line')
    for i, (xi, yi, yhi) in enumerate(zip(x, y, y_hat)): # zip parcourt en parallèle les 3 array
        plt.plot([xi, xi], [yi, yhi], linestyle='--', color='red', linewidth=1, label='Loss' if i == 0 else None) # coordonnées x (2x le même) et y réel et prédit
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Cost: {round(loss, 6)}")
    plt.show()

def main():
    """Tester of my plot function"""

    print("============= TEST  ===================")

    x = np.arange(1,6)
    y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
    print("x arange : ", x)
    print("y array : ", y)

    theta1= np.array([18,-1])
    print("theta1 : \n", theta1)
    plot_with_loss(x, y, theta1)
    
    theta2 = np.array([14, 0])
    print("theta2 : \n", theta2)
    plot_with_loss(x, y, theta2)

    theta3 = np.array([12, 0.8])
    print("theta3 : \n", theta3)
    plot_with_loss(x, y, theta3)

    return 0

if __name__ == "__main__":
    main()
