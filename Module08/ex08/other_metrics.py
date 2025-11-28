import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    Returns:
    The accuracy score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    return

def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
    The precision score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    return

def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
    The recall score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    return

def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
    The f1 score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    return

def main():
    """Tester of my functions of precision indicators"""

    print("============= TEST 1 ===================")
    y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))
    print("y_hat:\n", y_hat)
    print("y    :\n", y)

    print("============= 1 ===================")
    print("Accuracy :", accuracy_score_(y, y_hat))
    print("Expected : 0.5")
    print("sklearn Accuracy :", accuracy_score(y, y_hat))

    print("============= 2 ===================")
    print("Precision :", precision_score_(y, y_hat))
    print("Expected : 0.4")
    print("sklearn Precision :", precision_score(y, y_hat))

    print("============= 3 ===================")
    print("Recall :", recall_score_(y, y_hat))
    print("Expected : 0.6666666666666666")
    print("sklearn Recall :", recall_score(y, y_hat))

    print("============= 4 ===================")
    print("F1 score :", f1_score_(y, y_hat))
    print("Expected : 0.5")
    print("sklearn F1 score :", f1_score(y, y_hat))

    print("============= TEST 2 ===================")
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
    print("y_hat:\n", y_hat)
    print("y    :\n", y)

    print("============= 1 ===================")
    print("Accuracy :", accuracy_score_(y, y_hat))
    print("Expected : 0.625")
    print("sklearn Accuracy :", accuracy_score(y, y_hat))

    print("============= 2 ===================")
    print("Precision :", precision_score_(y, y_hat, pos_label='dog'))
    print("Expected : 0.6")
    print("sklearn Precision :", precision_score(y, y_hat, pos_label='dog'))

    print("============= 3 ===================")
    print("Recall :", recall_score_(y, y_hat, pos_label='dog'))
    print("Expected : 0.75")
    print("sklearn Recall :", recall_score(y, y_hat, pos_label='dog'))

    print("============= 4 ===================")
    print("F1 score :", f1_score_(y, y_hat, pos_label='dog'))
    print("Expected : 0.6666666666666665")
    print("sklearn F1 score :", f1_score(y, y_hat, pos_label='dog'))


    print("================ TEST 3 ==================")
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
    print("y_hat:\n", y_hat)
    print("y    :\n", y)

    print("============= 1 ===================")
    print("Precision :", precision_score_(y, y_hat, pos_label='norminet'))
    print("Expected : 0.6666666666666666")
    print("sklearn Precision :", precision_score(y, y_hat, pos_label='norminet'))

    print("============= 2 ===================")
    print("Recall :", recall_score_(y, y_hat, pos_label='norminet'))
    print("Expected : 0.5")
    print("sklearn Recall :", recall_score(y, y_hat, pos_label='norminet'))

    print("============= 3 ===================")
    print("F1 score :", f1_score_(y, y_hat, pos_label='norminet'))
    print("Expected : 0.5714285714285715")
    print("sklearn F1 score :", f1_score(y, y_hat, pos_label='norminet'))
    return 0


if __name__ == "__main__":
    main()