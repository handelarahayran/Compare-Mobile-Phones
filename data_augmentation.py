import numpy as np


def data_augmentation(data, names):
    """
    Takes a data set and augments its number of features by getting all the possible
    pair-multiplications. Does the same for the features names
    :param data: The samples to be augmented
    :param names: The features names to be augmented
    :return: new_data: the augmented data set
    :return: names: the now-augmented name list
    """
    features = np.shape(data)[1]
    new_data = np.concatenate((data, np.zeros(((np.shape(data)[0]), int(features/2 + (features**2)/2)))), axis=1)

    pointer = 0
    for i in range(features):
        for j in range(i, features):
            new_data[:, features + pointer] = data[:, i] * data[:, j]
            names.append(names[i]+'*'+names[j])
            pointer += 1

    return new_data, names
