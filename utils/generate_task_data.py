import os
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import tensorflow_datasets as tfds

DATA_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'raw_data')
TASK_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'task_data')
np.random.seed(1)

if not os.path.exists(DATA_ROOT):
    os.mkdir(DATA_ROOT)

if not os.path.exists(TASK_ROOT):
    os.mkdir(TASK_ROOT)


# Save task data to specific directory
def save_task_data(parent_dir, task_dir, X_l, y_l, X_u, y_u, X_test, y_test):
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    task_dir = os.path.join(parent_dir, task_dir)
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)
    np.save(os.path.join(task_dir, 'X_l.npy'), X_l)
    np.save(os.path.join(task_dir, 'y_l.npy'), y_l)
    np.save(os.path.join(task_dir, 'X_u.npy'), X_u)
    np.save(os.path.join(task_dir, 'y_u.npy'), y_u)
    np.save(os.path.join(task_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(task_dir, 'y_test.npy'), y_test)
    return


# Extract data for a specific binary classification task for MNIST and Fashion
def split_for_binary_task(train_data, test_data, c0=0, c1=1, num_labeled=50):
    # Prepare bianrized training data; within the training data, a given number of data is labeled
    train_data_bin_0 = []
    train_data_bin_1 = []
    for i in range(len(train_data)):
        if train_data[i][1] == c0:
            train_data_bin_0.append([train_data[i][0].numpy(), 0])
        elif train_data[i][1] == c1:
            train_data_bin_1.append([train_data[i][0].numpy(), 1])
    train_data_bin_labeled = train_data_bin_0[0: num_labeled // 2] + train_data_bin_1[0: num_labeled // 2]
    train_data_bin_unlabeled = train_data_bin_0[num_labeled // 2:] + train_data_bin_1[num_labeled // 2:]
    X_l = np.array([x[0] for x in train_data_bin_labeled])
    y_l = np.array([x[1] for x in train_data_bin_labeled])
    X_u = np.array([x[0] for x in train_data_bin_unlabeled])
    y_u = np.array([x[1] for x in train_data_bin_unlabeled])

    # Prepare binarized testing data
    test_data_bin = []
    for i in range(len(test_data)):
        if test_data[i][1] == c0:
            test_data_bin.append([test_data[i][0].numpy(), 0])
        elif test_data[i][1] == c1:
            test_data_bin.append([test_data[i][0].numpy(), 1])
    X_test = np.array([x[0] for x in test_data_bin])
    y_test = np.array([x[1] for x in test_data_bin])

    return X_l, y_l, X_u, y_u, X_test, y_test


# Generate task data for binarized MNIST, Fashion and CIFAR10
# Each task is a binary classification on two classes, 45 tasks in total for each dataset
def generate_binary_task_data(dataset='mnist'):

    task_parent_dir = os.path.join(TASK_ROOT, dataset + '_bin')
    if dataset == 'mnist':
        train_data = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=ToTensor())
        test_data = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=ToTensor())
        num_labeled_data = 120
    elif dataset == 'fashion':
        train_data = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=ToTensor())
        test_data = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=ToTensor())
        num_labeled_data = 120
    elif dataset == 'cifar10':
        train_data = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=ToTensor())
        test_data = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=ToTensor())
        num_labeled_data = 400
    else:
        raise NotImplementedError

    if not os.path.exists(task_parent_dir):
        os.mkdir(task_parent_dir)

    # Take two different classes of data to form a distinct binary classification task
    # No swapping of labels allowed, i.e. if there is a task with classes 4 and 5, no future task can be 5 and 4
    for c0 in range(9):
        for c1 in range(c0 + 1, 10):
            X_l, y_l, X_u, y_u, X_test, y_test = split_for_binary_task(train_data, test_data, c0=c0, c1=c1, num_labeled=num_labeled_data)
            save_task_data(task_parent_dir, str(c0) + '_' + str(c1), X_l, y_l, X_u, y_u, X_test, y_test)
            print(dataset + ': task saved for ' + str(c0) + '_' + str(c1))

    return


# Generate pseudo labels by taking part of y_u and flipping at an error rate
# MNIST and Fashion: |X_l| == 120, |X_u| == 5880, do |X_u'| == 120, 2940, 5880, error == 0, 0.2, 0.5
# CIFAR-10: |X_l| == 400, |X_u| == 4600, do |X_u'| == 400, 2300, 4600, error == 0, 0.2, 0.5
def generate_pseudo_labels(dataset='mnist'):

    task_parent_dir = os.path.join(TASK_ROOT, dataset + '_bin')
    if not os.path.exists(task_parent_dir):
        raise NotImplementedError

    if dataset == 'mnist' or dataset == 'fashion':
        n_u_space = [30, 60, 120]
        error_space = [0, 0.1, 0.2]
    elif dataset == 'cifar10':
        n_u_space = [400, 800, 1600]
        error_space = [0, 0.1, 0.2]
    else:
        raise NotImplementedError

    for c0 in range(9):
        for c1 in range(c0 + 1, 10):
            task_dir = os.path.join(task_parent_dir, str(c0) + '_' + str(c1))
            if not os.path.exists(task_dir):
                raise NotImplementedError
            X_u = np.load(os.path.join(task_dir, 'X_u.npy'))
            y_u = np.load(os.path.join(task_dir, 'y_u.npy'))

            for n_u in n_u_space:
                X_u_prime = X_u[:n_u]
                y_u_prime = y_u[:n_u]
                for error in error_space:
                    rand_index = np.random.choice(y_u_prime.shape[0], int(y_u_prime.shape[0] * error))
                    y_u_prime_flipped = y_u_prime.copy()
                    y_u_prime_flipped[rand_index] = 1 - y_u_prime_flipped[rand_index]
                    np.save(os.path.join(task_dir, 'y_u_' + str(n_u) + '_' + str(error) + '.npy'), y_u_prime)
                np.save(os.path.join(task_dir, 'X_u_' + str(n_u) + '.npy'), X_u_prime)

    return


if __name__ == '__main__':
    for dataset in ['mnist', 'fashion', 'cifar10']:
        generate_binary_task_data(dataset)
        generate_pseudo_labels(dataset)

