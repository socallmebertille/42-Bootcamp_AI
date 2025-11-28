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
    try:
        models_path = os.path.join(os.path.dirname(__file__), "models.pkl")
        with open(models_path, "rb") as f:
            models = pickle.load(f)
    except FileNotFoundError:
        print(" Error: models.pkl not found!")
        print("Please run benchmark_train.py first to train the models.")
        return 1

    X = data[["weight", "prod_distance", "time_delivery"]].to_numpy()            # recup data de l'echantillon
    Y = data[["target"]].to_numpy()
    Xtrain, Xtest, Ytrain, Ytest = data_spliter(X, Y, 0.8)                       # step 1 : split data 80% train / 20% test

    # 1. Evaluate all models and select the best one based on test MSE
    best_degree = min(models.keys(), key=lambda d: models[d]['mse_test'])
    model = models[best_degree]
    theta = model['theta']
    mean = model['mean']
    std = model['std']
    X_poly_train = MyLR.add_polynomial_features(Xtrain, best_degree)              # step 2 : polynomial features
    X_poly_test = MyLR.add_polynomial_features(Xtest, best_degree)
    X_poly_train_norm = (X_poly_train - mean) / std                               # step 3 : normalisation  
    X_poly_test_norm = (X_poly_test - mean) / std
    lr = MyLR(theta)                                                              # step 4 : init model avec thetas sauvegardes 
    y_pred_train = lr.predict_(X_poly_train_norm).reshape(-1, 1)                  # step 5 : evaluation
    y_pred_test = lr.predict_(X_poly_test_norm).reshape(-1, 1)
    mse_train = lr.mse(y_pred_train, Ytrain)                                      # step 6 : test performance
    mse_test = lr.mse(y_pred_test, Ytest)
    print(f"\nBest model (degree {best_degree}) → Train MSE = {mse_train:.2f}, Test MSE = {mse_test:.2f}")

    # 2. Plot evaluation curve (MSE train/test pour tous les degrés)
    degrees = list(models.keys())
    mse_train_list = []
    mse_test_list = []
    for d in degrees:
        m = models[d]
        lr_tmp = MyLR(m['theta'])
        X_poly_tmp = MyLR.add_polynomial_features(Xtrain, d)
        X_poly_test_tmp = MyLR.add_polynomial_features(Xtest, d)
        X_poly_tmp_norm = (X_poly_tmp - m['mean']) / m['std']
        X_poly_test_norm_tmp = (X_poly_test_tmp - m['mean']) / m['std']
        y_pred_tmp = lr_tmp.predict_(X_poly_tmp_norm).reshape(-1, 1)
        y_pred_test_tmp = lr_tmp.predict_(X_poly_test_norm_tmp).reshape(-1, 1)
        mse_train_list.append(lr_tmp.mse(y_pred_tmp, Ytrain))
        mse_test_list.append(lr_tmp.mse(y_pred_test_tmp, Ytest))
    plt.plot(degrees, mse_train_list, marker='o', label="Train MSE")
    plt.plot(degrees, mse_test_list, marker='o', label="Test MSE")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("MSE")
    plt.title("Evaluation Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. Plot 2D of 3 scatterplots (= pour chaque feature)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    feature_names = ["Weight (tons)", "Prod Distance (Mkm)", "Time Delivery (days)"]
    for i, ax in enumerate(axes):
        ax.scatter(Xtest[:, i], Ytest, c='blue', alpha=0.6, s=30, label='True price')
        ax.scatter(Xtest[:, i], y_pred_test, c='red', alpha=0.6, s=30, marker='^', label='Predicted price')
        ax.set_xlabel(feature_names[i], fontsize=11)
        ax.set_ylabel("Target Price", fontsize=11)
        ax.set_title(f"Price vs {feature_names[i]}", fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Model Performance (Degree {best_degree})", fontsize=14, fontweight='bold')
    plt.tight_layout() # ajuste les sous-graphes pour éviter le chevauchement
    plt.show()

    return 0


if __name__ == "__main__":
    main()