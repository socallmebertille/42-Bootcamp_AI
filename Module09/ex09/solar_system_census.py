import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Module09.ex08.my_logistic_regression import MyLogisticRegression as MyLR
from Module08.ex08.other_metrics import f1_score_


def one_vs_all_predict(classifiers, X):
    """Predict class labels using one-vs-all classifiers."""
    probs = [clf.predict_(X) for clf in classifiers]
    probs = np.hstack(probs)
    return np.argmax(probs, axis=1).reshape(-1, 1)


def load_models(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_scores(models):
    lambdas = [float(l) for l in models.keys()]
    scores = [models[l]["f1_test"] for l in models.keys()]
    plt.figure(figsize=(8, 5))
    plt.bar(lambdas, scores, width=0.1, color="#4f8bff", edgecolor="#123c9b")
    plt.xlabel("λ value")
    plt.ylabel("F1 score (test)")
    plt.title("Regularized logistic models – test F1 per λ")
    plt.ylim(0, 1.05)
    for l, s in zip(lambdas, scores):
        plt.text(l, s + 0.01, f"{s:.2f}", ha="center", va="bottom", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_predictions(X, y_true, y_pred):
    # Use weight vs height for a readable 2D view
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    labels = ["Venus", "Earth", "Mars", "Asteroids"]
    plt.figure(figsize=(9, 6))
    for cls, color, label in zip(range(4), colors, labels):
        mask_true = (y_true.ravel() == cls)
        mask_pred = (y_pred.ravel() == cls)
        plt.scatter(
            X[mask_true, 0], X[mask_true, 1],
            c=color, marker="o", edgecolors="k", alpha=0.6,
            label=f"Actual {label}"
        )
        plt.scatter(
            X[mask_pred, 0], X[mask_pred, 1],
            color=color, marker="x", linewidths=1.5, alpha=0.9,
            label=f"Pred {label}"
        )
    plt.xlabel("Weight")
    plt.ylabel("Height")
    plt.title("Best model – actual vs predicted classes")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()


def main():
    base = os.path.dirname(__file__)
    models_path = os.path.join(base, "models.pkl")

    models = load_models(models_path)

    # Display stored scores
    print("Stored model scores (validation → test):")
    for lambda_ in sorted(models.keys()):
        print(
            f"λ = {lambda_:.1f} | val F1 = {models[lambda_]['f1_val']:.4f} | "
            f"test F1 = {models[lambda_]['f1_test']:.4f}"
        )

    # Choose best lambda based on validation F1
    best_lambda = max(models.items(), key=lambda kv: kv[1]['f1_val'])[0]
    degree = int(models[best_lambda]["degree"])
    print(f"\nBest λ based on validation: {best_lambda:.1f} (degree={degree})")

    # Load data and drop CSV index column
    X = pd.read_csv(os.path.join(base, "solar_system_census.csv"), index_col=0).to_numpy()
    Y = pd.read_csv(os.path.join(base, "solar_system_census_planets.csv"), index_col=0).to_numpy()

    # Train / test split
    X_train, X_test, Y_train, Y_test = MyLR.data_spliter_(X, Y, 0.8)

    # Polynomial features and normalization
    X_train_poly = MyLR.add_polynomial_features(X_train, degree)
    X_test_poly = MyLR.add_polynomial_features(X_test, degree)

    X_train_norm, mean, std = MyLR.normalize_features(X_train_poly)
    X_test_norm = (X_test_poly - mean) / std

    # Train best model from scratch (one-vs-all)
    classifiers = []
    for cls in range(4):
        y_bin = (Y_train == cls).astype(int)
        theta = np.zeros((X_train_norm.shape[1] + 1, 1))
        lr = MyLR(theta, alpha=0.01, max_iter=3000, penality='l2', lambda_=best_lambda)
        lr.fit_(X_train_norm, y_bin)
        classifiers.append(lr)

    # Evaluate on test set
    y_test_pred = one_vs_all_predict(classifiers, X_test_norm)
    f1_test = f1_score_(Y_test, y_test_pred)
    print(f"Best model retrained → Test F1 = {f1_test:.4f}\n")

    # Plots
    plot_scores(models)
    plot_predictions(X_test, Y_test, y_test_pred)
    return 0


if __name__ == "__main__":
    main()
