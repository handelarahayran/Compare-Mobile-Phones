import math
from time import perf_counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mnist import MNIST
import matplotlib.pyplot as plt
import random


def neural_network(train_data, test_data, train_lab, test_lab, n_epochs, batch_size, hidden_layers, learning_rate):
    """
    Trains a neural network model and returns the predicted labels
    :param train_data: training data set
    :param test_data: test data set
    :param train_lab: training labels
    :param test_lab: test labels
    :param n_epochs: number of epochs
    :param batch_size: batch size for training
    :param hidden_layers: number of neurons for every hidden layer
    :param learning_rate: the initial learning rate (lowered by 2% every epoch)
    :return: predicted labels of training and test set
    """

    loss_fn = nn.CrossEntropyLoss()
    features = train_data.size(dim=1)

    test_accu_list = np.zeros(n_epochs)
    train_accu_list = np.zeros(n_epochs)

    temporary = features
    modules = []
    for i in range(len(hidden_layers)):
        modules.append(nn.Linear(temporary, hidden_layers[i]))
        modules.append(nn.LeakyReLU(negative_slope=0.1))
        temporary = hidden_layers[i]
    modules.append(nn.Linear(temporary, 4))
    modules.append(nn.LeakyReLU(negative_slope=0.1))
    model = nn.Sequential(*modules)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.3)

    # Begin training
    stopwatch = perf_counter()
    for epoch in range(n_epochs):
        for i in range(0, len(train_data), batch_size):
            # Get batch data
            data_batch = train_data[i:i + batch_size]
            prediction = model(data_batch)
            lab_batch = train_lab[i:i + batch_size]
            loss = loss_fn(prediction, lab_batch)
            # Apply backward propagation for batch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate epoch statistics
        test_pred = model(train_data)
        lab_pred = torch.argmax(test_pred, dim=1)
        train_accu_list[epoch] = 100 * sum(lab_pred == train_lab) / len(lab_pred)

        test_pred = model(test_data)
        lab_pred = torch.argmax(test_pred, dim=1)
        test_accu_list[epoch] = 100 * sum(lab_pred == test_lab) / len(lab_pred)

        print(f'Finished epoch {epoch + 1}: train set accuracy {train_accu_list[epoch]:.4f},'
              f' test set accuracy {test_accu_list[epoch]:.4f}')

    # Plot final results
    timer = perf_counter() - stopwatch
    print("Total training time (seconds): " + str(timer))
    fig2, (ax2, ax3) = plt.subplots(2)
    ax2.set_title("Training set accuracy per epoch")
    ax2.plot(train_accu_list, '*-', linewidth=2.0, )
    ax3.set_title("Test set accuracy per epoch")
    ax3.plot(test_accu_list, '*-', linewidth=2.0)
    plt.show()
    return torch.argmax(model(train_data), dim=1), torch.argmax(model(test_data), dim=1)

