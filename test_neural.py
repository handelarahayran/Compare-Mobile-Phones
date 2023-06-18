import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from data_visualization import data_visualization
from data_augmentation import data_augmentation
from model_results import model_results
from regression import regression_train
from NN_trainer import neural_net_train
from neural_network import neural_network

# Simulation settings
k_fold_splits = 5
visualize_data = False

n_epochs = 1000
batch_size = 3
hid_lay_norm = [60, 20]
hid_lay_augm = [150, 80, 25]
hid_lay_redu = [20, 20]
learning_rate = 0.000001

# Import and prepare data set
data = pd.read_csv('train.csv')
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

data_train, data_test, augm_train, augm_test, reduced_train, reduced_test, label_train, label_test \
    = train_test_split(data, augmented_data, reduced_data, labels, test_size=0.2, random_state=39)

data_train = torch.tensor(data_train, dtype=torch.float32)
data_test = torch.tensor(data_test, dtype=torch.float32)
augm_train = torch.tensor(augm_train, dtype=torch.float32)
augm_test = torch.tensor(augm_test, dtype=torch.float32)
reduced_train = torch.tensor(reduced_train, dtype=torch.float32)
reduced_test = torch.tensor(reduced_test, dtype=torch.float32)
label_train = torch.squeeze(torch.tensor(label_train, dtype=torch.long).reshape(-1, 1))
label_test = torch.squeeze(torch.tensor(label_test, dtype=torch.long).reshape(-1, 1))

[data_train_predictions, data_test_predictions] = neural_network(augm_train, augm_test, label_train, label_test,
                                                                 n_epochs, batch_size, hid_lay_augm, learning_rate)

model_results(data_train_predictions, data_test_predictions, data_train_predictions,
              data_test_predictions,
              data_train_predictions, data_test_predictions, label_train, label_test)
