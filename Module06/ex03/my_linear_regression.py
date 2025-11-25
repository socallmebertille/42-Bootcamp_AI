import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Module05.ex06.loss import predict, loss_, loss_elem_
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