import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true: a numpy.ndarray for the correct labels
        y_hat: a numpy.ndarray for the predicted labels
        labels: optional, a list of labels to index the matrix.
                This may be used to reorder or select a subset of labels. (default=None)
        df_option: optional, if set to True the function will return
                a pandas DataFrame instead of a numpy array. (default=False)
    Returns:
        Confusion matrix as a numpy ndarray or a pandas DataFrame according to df_option value.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y_true.shape != y_hat.shape:
        return None
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_hat)))
    else:
        labels = np.array(labels)
    nb_label = len(labels)
    mat = np.zeros((nb_label, nb_label))
    for i in range(nb_label):
        for j in range(nb_label):
            mat[i,j] = np.sum((y_true == labels[i]) & (y_hat == labels[j]))
    if df_option:
        return pd.DataFrame(mat, index=labels, columns=labels)
    return mat

def main():
    """Tester of my confusion_matrix_ function"""

    print("============= TEST 1 ===================")

    y_hat = np.array([['norminet'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['bird']])
    y = np.array([['dog'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['norminet']])
    print("y_hat array : \n", y_hat.reshape(1, -1))
    print("y array : \n", y.reshape(1, -1))

    print("============= 1 ===================")
    print("confusion matrix :\n", confusion_matrix_(y, y_hat))
    print("Expected : array([[0 0 0]\n\t\t[0 2 1]\n\t\t[1 0 2]])")
    print("sklearn confusion matrix :\n", confusion_matrix(y, y_hat))

    print("============= 2 ===================")
    print("confusion matrix :\n", confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))
    print("Expected : array([[2 1]\n\t\t[0 2]])")
    print("sklearn confusion matrix :\n", confusion_matrix(y, y_hat, labels=['dog', 'norminet']))

    print("============= TEST 2 ===================")

    print("============= 1 ===================")
    print("confusion matrix :\n", confusion_matrix_(y, y_hat, df_option=True))
    print("Expected :\n      bird  dog  norminet\n"
          "bird       0    0         0\n"
          "dog        0    2         1\n"
          "norminet   1    0         2")
    
    print("============= 2 ===================")
    print("confusion matrix :\n", confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True))
    print("Expected :\n      bird  dog\n"
          "bird       0    0\n"
          "dog        0    2")

    return 0

if __name__ == "__main__":
    main()
