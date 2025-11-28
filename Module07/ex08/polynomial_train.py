import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ex05.mylinearregression import MyLinearRegression as MyLR
from ex07.polynomial_model import add_polynomial_features


def main():
    """Plotting Curves With Matplotlib with the manipulation of polynomial hypothesis"""

    try:
        csv_path = os.path.join(os.path.dirname(__file__), "are_blue_pills_magics.csv")
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: are_blue_pills_magics.csv not found!")
        print("Make sure the file is in the same directory as this script.")
        return 1

    Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
    Yscore = np.array(data["Score"]).reshape(-1, 1)

    theta4 = np.array([[-20],[160],[-80],[10],[-1]], dtype=float)
    theta5 = np.array([[1140],[-1850],[1110],[-305],[40],[-2]], dtype=float)
    theta6 = np.array([[9110],[-18015],[13400],[-4935],[966],[-96.4],[3.86]], dtype=float)

    initial_thetas = {
        1: np.ones((2,1)),
        2: np.ones((3,1)),
        3: np.ones((4,1)),
        4: theta4,
        5: theta5,
        6: theta6
    }

    alphas = {
        1: 5e-4,
        2: 1e-4,
        3: 5e-6,
        4: 1e-6,
        5: 5e-8,
        6: 1e-9
    }

    max_iters = {
        1: 100000,
        2: 150000,
        3: 200000,
        4: 300000,
        5: 400000,
        6: 500000
    }

    models = {}
    mse_scores = {}
    for degree in range(1, 7):
        print(f"\n=== Training degree {degree} ===")
        X_poly = add_polynomial_features(Xpill, degree)
        theta = initial_thetas[degree]
        
        lr = MyLR(theta, alpha=alphas[degree], max_iter=max_iters[degree])
        lr.fit_(X_poly, Yscore)
        
        models[degree] = lr
        y_pred = lr.predict_(X_poly)
        mse = lr.mse(y_pred, Yscore)
        mse_scores[degree] = mse
        print(f"Degree {degree} → MSE = {mse:.2f}")

    # Bar plot of MSE
    plt.figure(figsize=(10, 6))
    plt.bar(list(mse_scores.keys()), list(mse_scores.values()), color='skyblue')
    plt.xlabel("Polynomial Degree")
    plt.ylabel("MSE")
    plt.title("MSE of polynomial models of different degrees")
    plt.grid(True, linestyle="--", alpha=0.4)
    for degree, mse in mse_scores.items():
        plt.text(degree, mse, f'{mse:.2f}', ha='center', va='bottom')
    plt.show()

    # Plot the 6 models on top of the data
    plt.figure(figsize=(10, 6))
    plt.scatter(Xpill, Yscore, color='black', label="Data")
    continuous_x = np.arange(Xpill.min(), Xpill.max(), 0.01).reshape(-1, 1)
    for degree in range(1, 7):
        X_poly_cont = add_polynomial_features(continuous_x, degree)
        y_hat = models[degree].predict_(X_poly_cont)
        plt.plot(continuous_x, y_hat, label=f"Degree {degree}")
    plt.xlabel("Micrograms")
    plt.ylabel("Score")
    plt.title("Polynomial models vs data")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()

    return 0


if __name__ == "__main__":
    main()

# trouver les bons alpha pour chaque degré
# si MSE = nan ou tres grand, diminuer alpha de /10 ou + chaque degre
# sinon on normalise les X avant d'entrainer le modèle

