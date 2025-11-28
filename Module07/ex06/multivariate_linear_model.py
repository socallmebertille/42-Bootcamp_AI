import pandas as pd
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ex05.mylinearregression import MyLinearRegression as MyLR

def main():
    """Tester of my predict function"""

    print("============= TEST  ===================")

    data = pd.read_csv("~/42AI-Bootcamp/Module07/ex06/spacecraft_data.csv")
    X = np.array(data[['Age']])
    Y = np.array(data[['Sell_price']])
    myLR_age = MyLR(thetas = [[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 100000)
    myLR_age.fit_(X, Y)
    y_hat = myLR_age.predict_(X)
    print("MSE : ", myLR_age.mse(Y, y_hat))
    print("Expected : 55736.867198...")

    return 0

if __name__ == "__main__":
    main()