import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from data_visualization import data_visualization
from data_augmentation import data_augmentation
from random_forest import random_forest_train
from regression import regression_train
from NN_trainer import neural_net_train

# Simulation settings
k_fold_splits = 5           # number of bins for the k fold cross validation
visualize_data = False;     # whether to use the data_visualization function or not (takes time)
# Neural Network Settings
n_epochs = 750              # Number of epochs
batch_size = 3              # Size of batch
hid_lay_norm = [60, 20]     # Normal set hidden neurons and layers
hid_lay_augm = [150, 80, 25]    # Augmented set hidden neurons and layers
hid_lay_redu = [20, 20]     # Reduced set hidden neurons and layers
lea_rate_norm = 0.0001      # Normal set learning rate
lea_rate_augm = 0.000001    # Augmented set learning rate
lea_rate_redu = 0.00001     # Reduced set learning rate

# Import and prepare data set
data = pd.read_csv('C:\luigiLite\Sxolh_lite\EARIN\Project\\train.csv')
original_names = data.columns.values.tolist()  # (original names are unrecognizably short)
names = ['battery_power', 'blue_tooth', 'clock_speed', 'dual_sim', 'front_cam',
         'four_g', 'int_memory', 'mobile_depth', 'mobile_width', 'n_cores', 'primary_cam',
         'pixel_height', 'pixel_width', 'ram', 'screen_height', 'screen_width', 'talk_time',
         'three_g', 'touch_screen', 'wifi']
data = data.to_numpy()
labels = data[:, 20]
data = data[:, 0:20]

# Visualize the data set
if visualize_data:
    data_visualization(data, labels, names, original_names)

# Prepare augmented data set (multiply each feature with each other - 230 features)
[augmented_data, augmented_names] = data_augmentation(data, names)
print("Augmented data features: " + str(np.shape(augmented_data)[1]))

# Prepare reduced data set using PCA
pca = PCA(n_components=3, svd_solver='full')
pca.fit(data)
reduced_data = pca.transform(data)
print("Reduced data features: " + str(np.shape(reduced_data))[1])

# Separate data set to test and training set using stratified K-Fold Validation
skf = StratifiedKFold(n_splits=k_fold_splits)
# Separate data set to test and training set
data_train, data_test, augm_train, augm_test, reduced_train, reduced_test, label_train, label_test \
    = train_test_split(data, augmented_data, reduced_data, labels, test_size=0.33, random_state=40)
print("Training set size: " + str(np.shape(data_train)[0]))
print("Test set size: " + str(np.shape(data_test)[0]))

# Train the 3 data sets with 3 models,
# print the accuracies and confusion matrices
# and return the accuracies for performance checking
accuracy_results = np.zeros((k_fold_splits, 3, 6))
for i, (train_index, test_index) in enumerate(skf.split(np.zeros(np.shape(data)[0]), labels)):
    # Train the  Regression model and return the data sets accuracies
    print('-----------------')
    print('Regression Training: Fold ' + str(i + 1))
    print('-----------------')
    accuracy_results[i, 0, :] = regression_train(data[train_index, :], data[test_index, :],
                                                 augmented_data[train_index, :],
                                                 augmented_data[test_index, :], reduced_data[train_index, :],
                                                 reduced_data[test_index, :],
                                                 labels[train_index], labels[test_index])

    print('-----------------')
    print('Neural Network Training: Fold ' + str(i + 1))
    print('-----------------')
    accuracy_results[i, 1, :] = neural_net_train(data[train_index, :], data[test_index, :],
                                                 augmented_data[train_index, :],
                                                 augmented_data[test_index, :], reduced_data[train_index, :],
                                                 reduced_data[test_index, :],
                                                 labels[train_index], labels[test_index], n_epochs, batch_size,
                                                 hid_lay_norm, hid_lay_augm, hid_lay_redu, lea_rate_norm, lea_rate_augm,
                                                 lea_rate_redu)
    print('-----------------')
    print('Random Forest Training: Fold ' + str(i + 1))
    print('-----------------')
    accuracy_results[i, 2, :] = random_forest_train(data[train_index, :], data[test_index, :],
                                                    augmented_data[train_index, :],
                                                    augmented_data[test_index, :], reduced_data[train_index, :],
                                                    reduced_data[test_index, :],
                                                    labels[train_index], labels[test_index])

print(accuracy_results)
