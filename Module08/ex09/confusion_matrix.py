import numpy as np
from sklearn.metrics import confusion_matrix

def confusion_matrix_(y_true, y_hat, labels=None):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true: numpy.ndarray for the correct labels
        y_hat: numpy.ndarray for the predicted labels
        labels: Optional, a list of labels to index the matrix.
        This may be used to reorder or select a subset of labels. (default=None)
    Returns:
        The confusion matrix as a numpy ndarray.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """


    return 

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
    return

def main():
    """Tester of my confusion_matrix_ function"""

    print("============= TEST 1 ===================")

    y_hat = np.array([['norminet'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['bird']])
    y = np.array([['dog'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['norminet']])
    print("y_hat array : \n", y_hat)
    print("y array : \n", y)

    print("============= 1 ===================")
    print("confusion matrix :", confusion_matrix_(y, y_hat))
    print("Expected : array([[0 0 0]\n\t\t[0 2 1]\n\t\t[1 0 2]])")
    print("sklearn confusion matrix :", confusion_matrix(y, y_hat))

    print("============= 2 ===================")
    print("confusion matrix :", confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))
    print("Expected : array([[2 1]\n\t\t[0 2]])")
    print("sklearn confusion matrix :", confusion_matrix(y, y_hat, labels=['dog', 'norminet']))

    print("============= TEST 2 ===================")

    print("============= 1 ===================")
    print("confusion matrix :", confusion_matrix_(y, y_hat, df_option=True))
    print("Expected :\n      bird  dog  norminet\n"
          "bird       0    0         0\n"
          "dog        0    2         1\n"
          "norminet   1    0         2")
    
    print("============= 2 ===================")
    print("confusion matrix :", confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True))
    print("Expected :\n      bird  dog\n"
          "bird       0    0\n"
          "dog        0    2")

    return 0

if __name__ == "__main__":
    main()
