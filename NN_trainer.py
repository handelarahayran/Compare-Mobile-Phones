import math
from time import perf_counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mnist import MNIST
import matplotlib.pyplot as plt
import random

from neural_network import neural_network
from model_results import model_results


def neural_net_train(data_train, data_test, augm_train, augm_test, reduced_train, reduced_test, label_train,
                     label_test, n_epochs, batch_size, hid_lay_norm, hid_lay_augm, hid_lay_redu, lea_rate_norm,
                     lea_rate_augm, lea_rate_redu):
    """
    The middle function that handles the neural_network function
    Controls all the training and results, then returns them
    :param data_train: normal data training set
    :param data_test: normal data test set
    :param augm_train: augmented data training set
    :param augm_test: augmented data test set
    :param reduced_train: PCA data training set
    :param reduced_test: PCA data test set
    :param label_train: training set labels
    :param label_test: test set labels (to be passed in model_results)
    :param n_epochs: number of epochs
    :param batch_size: batch size
    :param hid_lay_norm: number of neurons for each layer of the normal data set
    :param hid_lay_augm: number of neurons for each layer of the augmented data set
    :param hid_lay_redu: number of neurons for each layer of the reduced data set
    :param lea_rate_norm: the learning rate for the normal data set
    :param lea_rate_redu: the learning rate for the augmented data set
    :param lea_rate_augm: the learning rate for the reduced data set
    :return: training and test set accuracies of all 3 data sets
    """
    data_train = torch.tensor(data_train, dtype=torch.float32)
    data_test = torch.tensor(data_test, dtype=torch.float32)
    augm_train = torch.tensor(augm_train, dtype=torch.float32)
    augm_test = torch.tensor(augm_test, dtype=torch.float32)
    reduced_train = torch.tensor(reduced_train, dtype=torch.float32)
    reduced_test = torch.tensor(reduced_test, dtype=torch.float32)

    label_train = torch.squeeze(torch.tensor(label_train, dtype=torch.long).reshape(-1, 1))
    label_test = torch.squeeze(torch.tensor(label_test, dtype=torch.long).reshape(-1, 1))
    print("Start training normal data set")
    [data_train_predictions, data_test_predictions] = neural_network(data_train, data_test, label_train, label_test,
                                                                     n_epochs, batch_size, hid_lay_norm, lea_rate_norm)
    print("Start training augmented data set")
    [augmented_train_predictions, augmented_test_predictions] = neural_network(augm_train, augm_test, label_train,
                                                                               label_test,
                                                                               n_epochs, batch_size, hid_lay_augm,
                                                                               lea_rate_augm)
    print("Start training reduced data set")
    [reduced_train_predictions, reduced_test_predictions] = neural_network(reduced_train, reduced_test, label_train,
                                                                           label_test,
                                                                           n_epochs, batch_size, hid_lay_redu,
                                                                           lea_rate_redu)

    return model_results(data_train_predictions.numpy(), data_test_predictions.numpy(),
                         augmented_train_predictions.numpy(),
                         augmented_test_predictions.numpy(),
                         reduced_train_predictions.numpy(), reduced_test_predictions.numpy(), label_train.numpy(),
                         label_test.numpy())
