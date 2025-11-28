import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ex10.mylinearregression import MyLinearRegression as MyLR
from ex09.data_spliter import data_spliter

def main():
    """Tester of my Machine Learning function"""

    try:
        csv_path = os.path.join(os.path.dirname(__file__), "space_avocado.csv")
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: space_avocado.csv not found!")
        print("Make sure the file is in the same directory as this script.")
        return 1

    X = data[["weight", "prod_distance", "time_delivery"]].to_numpy()
    Y = data[["target"]].to_numpy()

    Xtrain, Xtest, Ytrain, Ytest = data_spliter(X, Y, 0.8)

    # Charger tous les modèles
    with open(os.path.join(os.path.dirname(__file__), "models.pkl"), "rb") as f:
        models = pickle.load(f)

    # Choisir le meilleur modèle (ici degree 4)
    best_degree = 4
    model = models[best_degree]
    theta = model['theta']
    mean = model['mean']
    std = model['std']

    # Polynomial features et normalisation
    X_poly_train = MyLR.add_polynomial_features(Xtrain, best_degree)
    X_poly_train_norm = (X_poly_train - mean) / std

    X_poly_test = MyLR.add_polynomial_features(Xtest, best_degree)
    X_poly_test_norm = (X_poly_test - mean) / std

    # Créer le LR avec theta sauvegardé
    lr = MyLR(theta)
    y_pred_train = lr.predict_(X_poly_train_norm).reshape(-1, 1)
    y_pred_test = lr.predict_(X_poly_test_norm).reshape(-1, 1)

    # Calcul MSE
    mse_train = lr.mse(y_pred_train, Ytrain)
    mse_test = lr.mse(y_pred_test, Ytest)
    print(f"\nBest model (degree {best_degree}) → Train MSE = {mse_train:.2f}, Test MSE = {mse_test:.2f}")

    # Plot evaluation curve (MSE train/test pour tous les degrés)
    degrees = list(models.keys())
    mse_train_list = []
    mse_test_list = []
    for d in degrees:
        m = models[d]
        lr_tmp = MyLR(m['theta'])
        X_poly_tmp = MyLR.add_polynomial_features(Xtrain, d)
        X_poly_tmp_norm = (X_poly_tmp - m['mean']) / m['std']
        y_pred_tmp = lr_tmp.predict_(X_poly_tmp_norm).reshape(-1,1)
        mse_train_list.append(lr_tmp.mse(y_pred_tmp, Ytrain))

        X_poly_test_tmp = MyLR.add_polynomial_features(Xtest, d)
        X_poly_test_norm_tmp = (X_poly_test_tmp - m['mean']) / m['std']
        y_pred_test_tmp = lr_tmp.predict_(X_poly_test_norm_tmp).reshape(-1,1)
        mse_test_list.append(lr_tmp.mse(y_pred_test_tmp, Ytest))

    plt.plot(degrees, mse_train_list, marker='o', label="Train MSE")
    plt.plot(degrees, mse_test_list, marker='o', label="Test MSE")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("MSE")
    plt.title("Evaluation Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Scatter 3D: vrai prix vs prédiction
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xtest[:,0], Xtest[:,1], Ytest, label="True price", color='blue')
    ax.scatter(Xtest[:,0], Xtest[:,1], y_pred_test, label="Predicted price", color='red')
    ax.set_xlabel("Weight")
    ax.set_ylabel("Prod Distance")
    ax.set_zlabel("Target Price")
    ax.set_title(f"Predictions vs True Price (Degree {best_degree})")
    ax.legend()
    plt.show()

    return 0


if __name__ == "__main__":
    main()