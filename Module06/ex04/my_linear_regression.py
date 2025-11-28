import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Module05.ex06.loss import predict, loss_, loss_elem_
from Module05.ex09.other_losses import mse_
from Module06.ex02.fit import fit_

class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
    
    def fit_(self, x, y):
        new_theta = fit_(x, y, self.thetas, self.alpha, self.max_iter)
        if new_theta is not None:
            self.thetas = new_theta
        return self.thetas
    
    def predict_(self, x):
        return predict(x, self.thetas)
    
    def loss_elem_(self, y, y_hat):
        return loss_elem_(y, y_hat)
    
    def loss_(self, y, y_hat):
        return loss_(y, y_hat)
    
    @staticmethod # car méthode n'a pas accès à l'intsance self oou la class cls
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
                y_hat = predict(x, tmp_thetas)
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
    """Tester of my functions of precision indicator"""

    print("============= TEST ===================")

    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])

    print("x array : \n", x)
    print("y array : \n", y)

    print("============= 1 ===================")

    lr1 = MyLinearRegression(np.array([[2], [0.7]]))
    y_hat = lr1.predict_(x)
    print("y_hat predict : \n", y_hat)
    print("Expected : array([[10.74695094],\n\t\t[17.05055804],\n\t\t[24.08691674],\n\t\t[36.24020866],\n\t\t[42.25621131]])")

    print("loss array :\n", lr1.loss_elem_(y, y_hat))
    print("Expected : array([[710.45867381],\n\t\t[364.68645485],\n\t\t[469.96221651],\n\t\t[108.97553412],\n\t\t[299.37111101]])")

    print("loss :\n", lr1.loss_(y, y_hat))
    print("Expected : 195.34539903032385")

    print("============= 2 ===================")

    lr2 = MyLinearRegression(np.array([[1], [1]]), 5e-8, 1500000)
    lr2.fit_(x, y)
    print("thetas : \n", lr2.thetas)
    print("Expected : array([[1.40709365],\n\t\t[1.1150909 ]])")

    y_hat = lr2.predict_(x)
    print("y_hat predict : \n", y_hat)
    print("Expected : array([[15.3408728 ],\n\t\t[25.38243697],\n\t\t[36.59126492],\n\t\t[55.95130097],\n\t\t[65.53471499]])")

    print("loss array :\n", lr2.loss_elem_(y, y_hat))
    print("Expected : array([[486.66604863],\n\t\t[115.88278416],\n\t\t[ 84.16711596],\n\t\t[ 85.96919719],\n\t\t[ 35.71448348]])")

    print("loss :\n", lr2.loss_(y, y_hat))
    print("Expected : 80.83996294128525")

    return 0

if __name__ == "__main__":
    main()