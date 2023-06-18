from sklearn.ensemble import RandomForestClassifier
import numpy as np

from model_results import model_results


def random_forest_train(data_train, data_test, augm_train, augm_test, reduced_train, reduced_test, label_train,
                        label_test):
    """
    Trains a random forest model with all 3 data sets and returns the final accuracies on training and test set
    :param data_train: normal data training set
    :param data_test: normal data test set
    :param augm_train: augmented data training set
    :param augm_test: augmented data test set
    :param reduced_train: PCA data training set
    :param reduced_test: PCA data test set
    :param label_train: training set labels
    :param label_test: test set labels (to be passed in model_results)
    :return: training and test set accuracies of all 3 data sets
    """
    # Import random forest model
    clf = RandomForestClassifier()

    # Random Forest on normal data set
    clf.fit(data_train, label_train)
    data_train_predictions = np.clip(np.round(
        clf.predict(data_train)).astype(int), 0, 3)
    data_test_predictions = np.clip(np.round(
        clf.predict(data_test)).astype(int), 0, 3)
    # Random Forest on augmented data set
    clf.fit(augm_train, label_train)
    augmented_train_predictions = np.clip(np.round(
        clf.predict(augm_train)).astype(int), 0, 3)
    augmented_test_predictions = np.clip(np.round(
        clf.predict(augm_test)).astype(int), 0, 3)
    # Random Forest on reduced data set
    clf.fit(reduced_train, label_train)
    reduced_train_predictions = np.clip(np.round(
        clf.predict(reduced_train)).astype(int), 0, 3)
    reduced_test_predictions = np.clip(np.round(
        clf.predict(reduced_test)).astype(int), 0, 3)

    # Print and plot all the results, then return the accuracies
    return model_results(data_train_predictions, data_test_predictions, augmented_train_predictions,
                         augmented_test_predictions,
                         reduced_train_predictions, reduced_test_predictions, label_train, label_test)
