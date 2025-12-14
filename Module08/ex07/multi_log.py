import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from my_logistic_regression import MyLogisticRegression as MyLR

def main():
    """Train 4 logistic regression classifiers to discriminate each class from the others"""

    print("=== Multiclass Logistic Regression (One-vs-All) ===\n")

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

    # STEP 1 : Split data
    x_train, x_test, y_train, y_test = MyLR.data_spliter_(x, y, 0.8)

    # STEP 2 : Polynomial features & normalization
    x_poly = MyLR.add_polynomial_features(x_train, 3)
    x_poly_test = MyLR.add_polynomial_features(x_test, 3)
    x_poly_norm, mean, std = MyLR.normalize_features(x_poly)
    x_poly_norm_test = (x_poly_test - mean) / std

    # STEP 3 : Train 4 logistic regression classifiers (one for each planet)
    models = []
    planet_names = ['Venus (0)', 'Earth (1)', 'Mars (2)', 'Asteroids (3)']
    probabilities = np.zeros((x_test.shape[0], 4))
    for zipcode in range(4):
        print(f"Training classifier for {planet_names[zipcode]}...")
        y_binary_train = np.where(y_train == zipcode, 1, 0)        # Create binary labels for this zipcode
        thetas = np.ones((x_poly_norm.shape[1] + 1, 1))            # Init first thetas
        myLR = MyLR(thetas, alpha=1e-3, max_iter=100000)           # Init model
        myLR.fit_(x_poly_norm, y_binary_train)                     # Train model
        models.append(myLR)                                        # Save model
        y_pred_train = myLR.predict_(x_poly_norm)                  # Get prediction
        y_pred_binary_train = np.where(y_pred_train >= 0.5, 1, 0)  # Pred into binary labels
        accuracy = np.mean(y_pred_binary_train == y_binary_train)  # Calculate training accuracy for this classifier
        print(f"  → Training accuracy: {accuracy:.4f}\n")          # Display accuracy on training set
        probabilities[:, zipcode] = myLR.predict_(x_poly_norm_test).flatten() # Get prediction on test set

    # STEP 4 : Select the class with highest probability for each sample
    y_pred_multiclass = np.argmax(probabilities, axis=1).reshape(-1, 1)

    # STEP 6 : Calculate accuracy on test set
    accuracy_test = np.mean(y_pred_multiclass == y_test)
    print(f"\n{'='*50}")
    print(f"Fraction of correct predictions on test set: {accuracy_test:.4f}")
    print(f"{'='*50}\n")

    # Display some example predictions
    print("     First 10 test samples :")
    print(f"{'True':<8} {'Predicted':<12} {'Probabilities (0, 1, 2, 3)'}")
    print("-" * 50)
    for i in range(5):
        true_label = int(y_test[i, 0])
        pred_label = int(y_pred_multiclass[i, 0])
        probs = probabilities[i]
        print(f"{true_label:<8} {pred_label:<12} [{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}, {probs[3]:.3f}]")

    # STEP 7 : Plot 3 scatter plots (one for each pair of features)
    feature_names = ['weight', 'height', 'bone_density']
    colors = ['cyan', 'green', 'orange', 'purple']  # One color per planet
    planet_labels = ['Venus', 'Earth', 'Mars', 'Asteroids']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        ax = axes[idx]
        for zipcode in range(4):
            corr_mask = (y_test.flatten() == zipcode) & (y_pred_multiclass.flatten() == zipcode)
            ax.scatter(x_test[corr_mask, i], x_test[corr_mask, j], c=colors[zipcode], s=50,
                       marker='o', alpha=0.6, label=f'{planet_labels[zipcode]} (correct)')
        incor_mask = y_pred_multiclass.flatten() != y_test.flatten()
        ax.scatter(x_test[incor_mask, i], x_test[incor_mask, j], color='red', s=150, 
                   linewidths=2, marker='x', label='Incorrect')
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'Multiclass Logistic Regression (One-vs-All) - Accuracy: {accuracy_test:.4f}')
    plt.tight_layout()
    plt.show()

    return 0

if __name__ == "__main__":
    main()