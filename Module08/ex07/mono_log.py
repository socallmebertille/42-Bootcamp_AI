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
    # STEP 1 : Get zipcode flag
    zipcode = int((sys.argv[1]).split("=")[1])
    print(f"Training model for Zipcode {zipcode}...")

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
    
    # STEP 3 : label binarization
    y_binary = np.where(y == zipcode, 1, 0).reshape(-1, 1)
    myLR = MyLR(theta=np.ones((1, 1)), alpha=1e-3, max_iter=100000)
    # STEP 2 : prepare data (split, polynomial features, normalization)
    myLR.prepare_data(x, y_binary, proportion=0.8, poly_degree=3, normalize=True)
    # STEP 4 : train model
    myLR.train()
    # STEP 5 : calculate and display accuracy
    results = myLR.evaluate()
    print(f"Train Accuracy = {results['accuracy_train']:.4f}, Test Accuracy = {results['accuracy_test']:.4f}")
    print(f"Fraction of correct predictions on test set: {results['accuracy_test']:.4f}")

    # STEP 6 : Plot 3 scatter plots
    feature_names = ['weight', 'height', 'bone_density']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        ax = axes[idx]
        correct = results['y_pred_binary_test'].flatten() == myLR.y_test_.flatten()
        
        mask = (results['y_pred_binary_test'].flatten() == 0) & correct
        ax.scatter(myLR.x_test_[mask, i], myLR.x_test_[mask, j],
                c='blue', marker='o', alpha=0.6, label='Other (correct)')
        
        mask = (results['y_pred_binary_test'].flatten() == 1) & correct
        ax.scatter(myLR.x_test_[mask, i], myLR.x_test_[mask, j],
                c='green', marker='o', alpha=0.6, label=f'Zipcode {zipcode} (correct)')
        
        incorrect = ~correct
        ax.scatter(myLR.x_test_[incorrect, i], myLR.x_test_[incorrect, j],
                color='red', s=100, linewidths=2, marker='x', label='Incorrect')
        
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Logistic Regression: Zipcode {zipcode} vs All (Accuracy: {results["accuracy_test"]:.4f})')
    plt.tight_layout()
    plt.show()

    return 0

if __name__ == "__main__":
    main()