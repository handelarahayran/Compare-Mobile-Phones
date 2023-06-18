import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from data_visualization import data_visualization
from data_augmentation import data_augmentation
from model_results import model_results

# Simulation settings
visualize_data = False


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

data_train, data_test, augm_train, augm_test, reduced_train, reduced_test, label_train, label_test \
    = train_test_split(data, augmented_data, reduced_data, labels, test_size=0.33, random_state=40)


clf = RandomForestClassifier()
clf.fit(data_train, label_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)

print(model_results(pred_train, pred_test, pred_train,
              pred_test,
              pred_train, pred_test, label_train, label_test))
