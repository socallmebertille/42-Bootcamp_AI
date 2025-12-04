import numpy as np
import pandas as pd
import os, sys
from my_logistic_regression import MyLogisticRegression as MyLR

def check_flag(flag):
    if not flag.startswith("-zipcode="):
        print("Error: flag must be -zipcode=")
        return 1
    try:
        x = int(flag.split("=")[1])
    except ValueError:
        print("Error: x must be an integer (0, 1, 2 or 3).")
        return 1
    if x not in [0, 1, 2, 3]:
        print("Error: x must be 0, 1, 2 or 3.")
        return 1
    return 0

def main():
    """Tester of my polynomial function"""

    if len(sys.argv) != 2 or check_flag(sys.argv[1]) == 1:
        print("Usage: python mono_log.py -zipcode=x (x = 0, 1, 2 ou 3)")
        return 1
    zipcode = int((sys.argv[1]).split("=")[1])
    print("Zipcode: ", zipcode)

    try:
        csv_path = os.path.join(os.path.dirname(__file__), "solar_system_census.csv")
        data_x = pd.read_csv(csv_path)
        csv_path = os.path.join(os.path.dirname(__file__), "solar_system_census_planets.csv")
        data_y = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: solar_system_census.csv not found!")
        print("Make sure the file is in the same directory as this script.")
        return 1
    categories = ['weight', 'height', 'bone_density']
    x = np.array(data_x[categories[zipcode]]).reshape(1, -1)
    y = np.array(data_y['Origin']).reshape(-1, 1)
    thetas = np.zeros((x.shape[0], 1))
    myLR = MyLR(thetas, alpha=1e-3, max_iter=100000)
    x_train, x_test, y_train, y_test = myLR.data_spliter_(x, y, 0.8)
    X_poly = myLR.add_polynomial_features(x_train, 3)

    print("============= TEST ===================")

    print(" : \n", )
    print("Expected : ")

    print(" : \n", )
    print("Expected : ")

    return 0

if __name__ == "__main__":
    main()
