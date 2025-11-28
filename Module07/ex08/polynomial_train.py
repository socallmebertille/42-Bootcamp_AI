import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Module07.ex05.mylinearregression import MyLinearRegression as MyLR
from Module07.ex07.polynomial_model import add_polynomial_features
from Module06.ex05.z_score import zscore  # ou from minmax import minmax


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

    models = {}
    mse_scores = {}
    alpha = 1e-3
    max_iter = 100000
    for degree in range(1, 7):
        print(f"\n=== Training degree {degree} ===")
        X_poly = add_polynomial_features(Xpill, degree)
        X_poly_norm = zscore(X_poly) # normalisation des donnees
        theta = np.ones((degree + 1, 1))
        lr = MyLR(theta, alpha=alpha, max_iter=max_iter)
        lr.fit_(X_poly_norm, Yscore)
        y_pred = lr.predict_(X_poly_norm)
        mse = lr.mse(y_pred, Yscore)
        models[degree] = lr
        mse_scores[degree] = mse
        print(f"Degree {degree} → MSE = {mse:.2f}")

    # Bar plot of MSE
    plt.figure(figsize=(10, 6))
    plt.bar(list(mse_scores.keys()), list(mse_scores.values()), 
            color='skyblue', edgecolor='black')
    plt.xlabel("Polynomial Degree")
    plt.ylabel("MSE")
    plt.title("MSE of polynomial models (with normalization)")
    plt.grid(True, linestyle="--", alpha=0.4, axis='y')
    for degree, mse in mse_scores.items():
        plt.text(degree, mse, f'{mse:.1f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()

    # Plot the 6 models on top of the data
    plt.figure(figsize=(12, 7))
    plt.scatter(Xpill, Yscore, color='black', label="Data", s=50, alpha=0.7, zorder=5)
    
    continuous_x = np.arange(Xpill.min(), Xpill.max(), 0.01).reshape(-1, 1)
    
    for degree in range(1, 7):
        # Créer et normaliser les features continues
        X_poly_cont = add_polynomial_features(continuous_x, degree)
        X_poly_cont_norm = zscore(X_poly_cont)  # ou minmax(X_poly_cont)
        
        # Prédire
        y_hat = models[degree].predict_(X_poly_cont_norm)
        
        plt.plot(continuous_x, y_hat, label=f"Degree {degree}", linewidth=2)
    
    plt.xlabel("Micrograms", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Polynomial models vs data (with normalization)", fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    main()