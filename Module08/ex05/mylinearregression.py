import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Module07.ex01.prediction import predict_
from Module07.ex02.loss import loss_, loss_elem_
from Module05.ex09.other_losses import mse_
from Module07.ex04.fit import fit_

class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        if isinstance(thetas, list):
            self.thetas = np.array(thetas, dtype=float).reshape(-1, 1)
        elif isinstance(thetas, np.ndarray):
            self.thetas = thetas.reshape(-1, 1)
        else:
            self.thetas = thetas
    
    def fit_(self, x, y):
        new_theta = fit_(x, y, self.thetas, self.alpha, self.max_iter)
        if new_theta is not None:
            self.thetas = new_theta
        return self.thetas
    
    def predict_(self, x):
        return predict_(x, self.thetas)
    
    def loss_elem_(self, y, y_hat):
        return loss_elem_(y, y_hat)
    
    def loss_(self, y, y_hat):
        return loss_(y, y_hat)
    
    @staticmethod
    def mse(y, y_hat):
        return mse_(y, y_hat)
    
    def plot_data(self, x, y, loss: bool, title: str, xlabel: str, ylabel: str):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return
        if x.size == 0 or y.size == 0:
            return
        if x.ndim != 2 or y.ndim != 2:
            return
        y_hat = self.predict_(x)
        plt.figure(figsize=(7,5))
        plt.scatter(x, y, color='lightskyblue', label='True data')
        x_line = np.array([x.min(), x.max()])
        y_line = self.thetas[0] + self.thetas[1] * x_line
        plt.plot(x_line, y_line, color='lightgreen', label='Prediction')
        if loss == True:
            for i, (xi, yi, yhi) in enumerate(zip(x, y, y_hat)): # zip parcourt en parallèle les 3 array
                plt.plot([xi, xi], [yi, yhi], linestyle='--', color='red', linewidth=1, label='Loss' if i == 0 else None) # coordonnées x (2x le même) et y réel et prédit
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_loss(self, x, y, precision: float):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return
        if x.size == 0 or y.size == 0:
            return
        if x.ndim != 2 or y.ndim != 2:
            return
        
        theta0_list = np.linspace(self.thetas[0][0] - precision, self.thetas[0][0] + precision, 6)   # 6 val de theta0 espacées également de -20 à 100
        theta1_range = np.linspace(-14, -4, 200)
        for t0 in theta0_list:
            losses = []
            for t1 in theta1_range:
                tmp_thetas = np.array([[t0], [t1]])
                y_hat = predict_(x, tmp_thetas)
                loss = self.mse(y, y_hat)
                losses.append(loss)
            plt.plot(theta1_range, losses, label=f"J(θ0={t0:.1f}, θ1)")
        plt.xlabel("θ1")
        plt.ylabel("cost function J(θ0, θ1)")
        plt.title("Loss evolution for different values of θ0")
        plt.grid(True)
        plt.legend()
        plt.show()

def main():
    """Tester of my linear regression class"""

    print("============= TEST ===================")

    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])

    print("X array : \n", X)
    print("Y array : \n", Y)
    mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])

    print("============= 1 ===================")

    y_hat = mylr.predict_(X)
    print("y_hat predict : \n", y_hat)
    print("Expected : array([[8.], [48.], [323.]])")

    print("loss array :\n", mylr.loss_elem_(Y, y_hat))
    print("Expected : array([[225.], [0.], [11025.]])")

    print("loss :\n", mylr.loss_(Y, y_hat))
    print("Expected : 1875.0")

    print("============= 2 ===================")

    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.fit_(X, Y)
    print("thetas : \n", mylr.thetas)
    print("Expected : array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])")

    y_hat = mylr.predict_(X)
    print("y_hat predict : \n", y_hat)
    print("Expected : array([[23.417..], [47.489..], [218.065...]])")

    print("loss array :\n", mylr.loss_elem_(Y, y_hat))
    print("Expected : array([[0.174..], [0.260..], [0.004..]])")

    print("loss :\n", mylr.loss_(Y, y_hat))
    print("Expected : 0.0732..")

    return 0

if __name__ == "__main__":
    main()