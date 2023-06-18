import matplotlib.pyplot as plt
import numpy as np


def data_visualization(data, labels, names, original_names):
    """
    Presents the data set content and sizes in the command line and
    with graphs, in order for the user to get more familiar with the data set
    :param data: the data set features
    :param labels: the data labels
    :param names: the feature names
    :param original_names: the feature original names (altered to be more descriptive)
    """
    # Various useful information plotted in the command line
    print('Original feature names: ' + ', '.join(original_names))
    print('Enhanced feature names: ' + ', '.join(names))
    print('Number of features: ' + str(np.shape(data)[1]))
    print('Number of samples: ' + str(np.shape(data)[0]))
    print('Unique sample labels: ' + np.array2string(np.int_(np.unique(labels))))

    for i in range(np.shape(data)[1]):
        # Plot all the features-label histograms separately
        unique_values = len(np.unique(data[:, i]))
        plt.hist2d(labels, data[:, i], [4, min(10, unique_values)], cmap='copper_r')
        plt.title(names[i])
        plt.colorbar()
        plt.show()

        if unique_values > 2:
            # if the feature has more than 2 unique values, plot a boxplot as well
            plt.boxplot(data[:, i])
            plt.title(names[i] + ', mean = ' + str(np.mean(data[:, i])))
            plt.show()
