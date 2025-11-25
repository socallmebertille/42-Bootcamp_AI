import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

import sys, os#, csv
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ex03.my_linear_regression import MyLinearRegression as MyLR

# def read_csv(file):
#     """
#     Description:
#         Read a CSV file and return its content as a list of lists.
#     Returns:
#         data: numpy.ndarray, a matrix of dimension m * n.
#         None if file is empty.
#     """

#     data = []
#     try:
#         with open(file, "r", encoding="utf-8") as f:
#             lecteur = csv.reader(f, delimiter=", ")  # ou ";" selon ton fichier
#             for ligne in lecteur:
#                 data.append(ligne)
#         return np.array(data)
#     except FileNotFoundError:
#         print(f"Error : the file {file} doesnot exist.")
#         return None

def main():
    """Tester of my functions of precision indicator"""

    print("============= TEST ===================")

    data = pd.read_csv("are_blue_pills_magic.csv")
    Xpill = np.array(data['Micrograms']).reshape(-1,1)
    Yscore = np.array(data['Score']).reshape(-1,1)
    linear_model1 = MyLR(np.array([[89.0], [-8]]))
    linear_model2 = MyLR(np.array([[89.0], [-6]]))
    Y_model1 = linear_model1.predict_(Xpill)
    Y_model2 = linear_model2.predict_(Xpill)

    print(MyLR.mse_(Yscore, Y_model1))
    print("Expected : 57.60304285714282")

    print(mean_squared_error(Yscore, Y_model1))
    print("Expected : 57.603042857142825")

    print(MyLR.mse_(Yscore, Y_model2))
    print("Expected : 232.16344285714285")

    print(mean_squared_error(Yscore, Y_model2))
    print("Expected : 232.16344285714285")

    return 0

if __name__ == "__main__":
    main()

# Patient: number of the patient.
# Micrograms: quantity of blue pills patient has taken (in micrograms).
# Score: Standardized score at the spacecraft driving test.