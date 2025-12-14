import numpy as np
import pandas as pd
import pickle
import os
from Module09.ex06.mylinearregression import MyLinearRegression as MyLR

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
    theta = np.ones((X.shape[1] + 2, 1))                                    # init de ma regression (theta_nb 1 pour biais)
    lr = MyLR(theta, alpha=1e-3, max_iter=100000)
    Xtrain, Xtest, Ytrain, Ytest = lr.data_spliter_(X, Y, 0.8)              # step 1 : split data 80% train / 20% test

    models = {}
    for degree in range(1, 5):
        print(f"\n=== Training degree {degree} ===")
        X_poly = MyLR.add_polynomial_features(Xtrain, degree)               # step 2 : polynomial features
        X_poly_test = MyLR.add_polynomial_features(Xtest, degree)
        X_poly_norm, mean, std = MyLR.normalize_features(X_poly)            # step 3 : normalisation
        X_poly_test_norm = (X_poly_test - mean) / std
        lr.fit_(X_poly_norm, Ytrain)                                        # step 4 : entrainement
        y_pred_train = lr.predict_(X_poly_norm)                             # step 5 : evaluation
        y_pred_test = lr.predict_(X_poly_test_norm)
        mse_train = lr.mse(y_pred_train, Ytrain)                            # step 6 : test performance
        mse_test = lr.mse(y_pred_test, Ytest)
        models[degree] = {                                                  # step 7 : save model
            'theta': lr.thetas,
            'mean': mean,
            'std': std,
            'mse_train': float(mse_train),
            'mse_test': float(mse_test)
        }
        print(f"Degree {degree} → Train MSE = {mse_train:.2f}, Test MSE = {mse_test:.2f}")

    with open(os.path.join(os.path.dirname(__file__), "models.pkl"), "wb") as f: # Sauvegarde des modèles dans fichier pickle
        pickle.dump(models, f)
        print("\nAll models saved to 'models.pkl'.")

    return 0

if __name__ == "__main__":
    main()