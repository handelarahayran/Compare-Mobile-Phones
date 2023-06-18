from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def model_results(data_train_predictions, data_test_predictions, augmented_train_predictions,
                  augmented_test_predictions,
                  reduced_train_predictions, reduced_test_predictions, label_train, label_test):
    """
    Calculates accuracy statistics for all data sets and also plots the confusion matrices
    :param data_train_predictions: predictions on normal train set
    :param data_test_predictions:  predictions on normal test set
    :param augmented_train_predictions: predictions on augmented train set
    :param augmented_test_predictions: predictions on augmented test set
    :param reduced_train_predictions: predictions on reduced train set
    :param reduced_test_predictions: predictions on reduced test set
    :param label_train: the correct labels of the training set
    :param label_test: the correct labels of the test set
    :return: training and test set accuracies of all 3 data sets
    """

    # Calculate accuracy on the normal test and training set
    normal_train_accuracy = accuracy_score(label_train, data_train_predictions)*100
    print("Accuracy on the normal training set:", normal_train_accuracy)
    normal_test_accuracy = accuracy_score(label_test, data_test_predictions)*100
    print("Accuracy on the normal test set:", normal_test_accuracy)
    # Calculate accuracy on the augmented test and training set
    augm_train_accuracy = accuracy_score(label_train, augmented_train_predictions)*100
    print("Accuracy on the augmented training set:", augm_train_accuracy)
    augm_test_accuracy = accuracy_score(label_test, augmented_test_predictions)*100
    print("Accuracy on the augmented test set:", augm_test_accuracy)
    # Calculate accuracy on the reduced test and training set
    reduced_train_accuracy = accuracy_score(label_train, reduced_train_predictions)*100
    print("Accuracy on the reduced training set:", reduced_train_accuracy)
    reduced_test_accuracy = accuracy_score(label_test, reduced_test_predictions)*100
    print("Accuracy on the reduced test set:", reduced_test_accuracy)

    # Confusion Matrices
    normal_train_confusion = confusion_matrix(label_train, data_train_predictions)
    normal_test_confusion = confusion_matrix(label_test, data_test_predictions)
    augm_train_confusion = confusion_matrix(label_train, augmented_train_predictions)
    augm_test_confusion = confusion_matrix(label_test, augmented_test_predictions)
    reduced_train_confusion = confusion_matrix(label_train, reduced_train_predictions)
    reduced_test_confusion = confusion_matrix(label_test, reduced_test_predictions)

    f, ax = plt.subplots(2, 3)
    sns.heatmap(normal_train_confusion, fmt='d', annot=True, linewidths=.5, ax=ax[0, 0])
    ax[0, 0].set_title('Normal - Train')
    sns.heatmap(normal_test_confusion, fmt='d', annot=True, linewidths=.5, ax=ax[1, 0])
    ax[1, 0].set_title('Normal - Test')
    sns.heatmap(augm_train_confusion, fmt='d', annot=True, linewidths=.5, ax=ax[0, 1])
    ax[0, 1].set_title('Augmented - Train')
    sns.heatmap(augm_test_confusion, fmt='d', annot=True, linewidths=.5, ax=ax[1, 1])
    ax[1, 1].set_title('Augmented - Test')
    sns.heatmap(reduced_train_confusion, fmt='d', annot=True, linewidths=.5, ax=ax[0, 2])
    ax[0, 2].set_title('Reduced - Train')
    sns.heatmap(reduced_test_confusion, fmt='d', annot=True, linewidths=.5, ax=ax[1, 2])
    ax[1, 2].set_title('Reduced - Test')
    plt.show()

    return normal_train_accuracy, normal_test_accuracy, augm_train_accuracy,\
        augm_test_accuracy, reduced_train_accuracy, reduced_test_accuracy
