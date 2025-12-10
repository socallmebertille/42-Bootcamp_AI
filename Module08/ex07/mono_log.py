import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
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
    """Train a logistic regression model for one zipcode vs all others"""

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
        print("Error: CSV files not found!")
        return 1
    
    x = np.array(data_x[['weight', 'height', 'bone_density']])
    y = np.array(data_y['Origin']).reshape(-1, 1)
    # Convert y to binary for the specified zipcode
    y_binary = np.where(y == zipcode, 1, 0).reshape(-1, 1)

    # step 1 : split data 80% train / 20% test
    x_train, x_test, y_train, y_test = MyLR.data_spliter_(x, y_binary, 0.8)
    # step 2 : polynomial features
    x_poly = MyLR.add_polynomial_features(x_train, 3)
    x_poly_test = MyLR.add_polynomial_features(x_test, 3)
    # step 3 : normalisation
    x_poly_norm, mean, std = MyLR.normalize_features(x_poly)
    x_poly_norm_test = (x_poly_test - mean) / std
    thetas = np.ones((x_poly_norm.shape[1] + 1, 1))
    myLR = MyLR(thetas, alpha=1e-3, max_iter=100000)
    # step 4 : entrainement
    myLR.fit_(x_poly_norm, y_train)
    # step 5 : evaluation
    y_pred_train = myLR.predict_(x_poly_norm)
    y_pred_test = myLR.predict_(x_poly_norm_test)
    # Convert predictions to binary (0 or 1)
    y_pred_binary_train = np.where(y_pred_train >= 0.5, 1, 0)
    y_pred_binary_test = np.where(y_pred_test >= 0.5, 1, 0)
    # step 6 : test performance
    accuracy_train = np.mean(y_pred_binary_train == y_train)
    accuracy_test = np.mean(y_pred_binary_test == y_test)
    print(f"Train Accuracy = {accuracy_train:.4f}, Test Accuracy = {accuracy_test:.4f}")
    print(f"Fraction of correct predictions on test set: {accuracy_test:.4f}")

    # step 7 : Plot 3 scatter plots
    feature_names = ['weight', 'height', 'bone_density']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # Chaque tuple (i, j) représente une paire de features
    for idx, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        ax = axes[idx]
        # Correct classification = prediction == ground truth
        correct = y_pred_binary_test.flatten() == y_test.flatten()
        # (1) Correctement classés comme "autre planète" (label=0)
        mask = (y_pred_binary_test.flatten() == 0) & correct
        ax.scatter(x_test[mask, i], x_test[mask, j],
                c='blue', marker='o', alpha=0.6, label='Other (correct)')
        # (2) Correctement classés comme "notre zipcode" (label=1)
        mask = (y_pred_binary_test.flatten() == 1) & correct
        ax.scatter(x_test[mask, i], x_test[mask, j],
                c='green', marker='o', alpha=0.6, label=f'Zipcode {zipcode} (correct)')
        # (3) Mauvaises prédictions
        incorrect = ~correct
        ax.scatter(x_test[incorrect, i], x_test[incorrect, j],
                color='red', s=100, linewidths=2, marker='x', label='Incorrect')
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'Logistic Regression: Zipcode {zipcode} vs All (Accuracy: {accuracy_test:.4f})')
    plt.tight_layout()
    plt.show()

    return 0

if __name__ == "__main__":
    main()
