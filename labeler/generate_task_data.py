"""
Step 0, generating task data for experiments
Each task contains (X_L, Y_L, X_U, Y_U, X_T, Y_T), where Y_U is unused during experiments
"""

import os
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor

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


# Extract data for a specific binary classification task for MNIST and Fashion
def split_for_multi_way(train_data, test_data, c0=0, ck=4, num_labeled=150, num_unlabeled=None):
    n = ck - c0 + 1 # number of classes per task
    # Prepare bianrized training data; within the training data, a given number of data is labeled
    train_data_bins = [[] for _ in range(n)]
    for i in range(len(train_data)):
        c = train_data[i][1]
        if c in range(c0, ck+1):
            train_data_bins[c % n].append([train_data[i][0].numpy(), c % n])
    train_data_bin_labeled = [x[0: num_labeled // n] for x in train_data_bins]
    if num_unlabeled is None:
        train_data_bin_unlabeled = [x[num_labeled // n:] for x in train_data_bins]
    else:
        train_data_bin_unlabeled = [x[num_labeled // n: (num_labeled + num_unlabeled) // n] for x in train_data_bins]
    train_data_bin_labeled = sum(train_data_bin_labeled, [])
    train_data_bin_unlabeled = sum(train_data_bin_unlabeled, [])

    X_l = np.array([x[0] for x in train_data_bin_labeled])
    y_l = np.array([x[1] for x in train_data_bin_labeled])
    X_u = np.array([x[0] for x in train_data_bin_unlabeled])
    y_u = np.array([x[1] for x in train_data_bin_unlabeled])

    # Prepare binarized testing data
    test_data_bin = []
    for i in range(len(test_data)):
        c = test_data[i][1]
        if c in range(c0, ck+1):
            test_data_bin.append([test_data[i][0].numpy(), c % n])
    X_test = np.array([x[0] for x in test_data_bin])
    y_test = np.array([x[1] for x in test_data_bin])
    return X_l, y_l, X_u, y_u, X_test, y_test


# generate 2 5-way tasks for MNIST
def generate_5_way_mnist():
    task_parent_dir = os.path.join(TASK_ROOT, 'mnist_5_way')
    if not os.path.exists(task_parent_dir):
        os.mkdir(task_parent_dir)
    train_data = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=ToTensor())
    num_labeled = 150
    for c0 in [0, 5]:
        X_l, y_l, X_u, y_u, X_test, y_test = split_for_multi_way(train_data, test_data,
                                                                   c0=c0, ck=c0+4, num_labeled=num_labeled)
        save_task_data(task_parent_dir, X_l, y_l, X_u, y_u, X_test, y_test, str(c0) + '-' + str(c0 + 4))
    return


# generate 1 10-way task for CIFAR-10
def generate_10_way_cifar10():
    task_parent_dir = os.path.join(TASK_ROOT, 'cifar10_10_way')
    if not os.path.exists(task_parent_dir):
        os.mkdir(task_parent_dir)
    train_data = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=ToTensor())
    test_data = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=ToTensor())
    num_labeled = 2000
    num_unlabeled = 30000
    X_l, y_l, X_u, y_u, X_test, y_test = \
        split_for_multi_way(train_data, test_data, c0=0, ck=9, num_labeled=num_labeled, num_unlabeled=num_unlabeled)
    save_task_data(task_parent_dir, X_l, y_l, X_u, y_u, X_test, y_test, str(0) + '-' + str(9))
    return


# generate 1 100-way task for CIFAR-100
def generate_100_way_cifar100():
    task_parent_dir = os.path.join(TASK_ROOT, 'cifar100_100_way')
    if not os.path.exists(task_parent_dir):
        os.mkdir(task_parent_dir)
    train_data = datasets.CIFAR100(root=DATA_ROOT, train=True, download=True, transform=ToTensor())
    test_data = datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=ToTensor())
    num_labeled = 5000
    num_unlabeled = 30000
    X_l, y_l, X_u, y_u, X_test, y_test = \
        split_for_multi_way(train_data, test_data, c0=0, ck=99, num_labeled=num_labeled, num_unlabeled=num_unlabeled)
    save_task_data(task_parent_dir, X_l, y_l, X_u, y_u, X_test, y_test, str(0) + '-' + str(99))
    return


def split_by_cifar100_superclasses(train_data, test_data, superclass=0, num_labeled=500, num_unlabeled=2000):
    dict_superclasses = {0: [72, 4, 95, 30, 55], 1: [73, 32, 67, 91, 1], 2: [92, 70, 82, 54, 62], 3: [16, 61, 9, 10, 28],
                         4: [51, 0, 53, 57, 83], 5: [40, 39, 22, 87, 86], 6: [20, 25, 94, 84, 5], 7: [14, 24, 6, 7, 18],
                         8: [43, 97, 42, 3, 88], 9: [37, 17, 76, 12, 68], 10: [49, 33, 71, 23, 60], 11: [15, 21, 19, 31, 38],
                         12: [75, 63, 66, 64, 34], 13: [77, 26, 45, 99, 79], 14: [11, 2, 35, 46, 98], 15: [29, 93, 27, 78, 44],
                         16: [65, 50, 74, 36, 80], 17: [56, 52, 47, 59, 96], 18: [8, 58, 90, 13, 48], 19: [81, 69, 41, 89, 85]}

    train_data_bins = [[] for _ in range(5)]
    for i in range(len(train_data)):
        c = train_data[i][1]
        if c in dict_superclasses[superclass]:
            new_c = dict_superclasses[superclass].index(c)
            train_data_bins[new_c].append([train_data[i][0].numpy(), new_c])
    train_data_bin_labeled = [x[0: num_labeled // 5] for x in train_data_bins]
    if num_unlabeled is None:
        train_data_bin_unlabeled = [x[num_labeled // 5:] for x in train_data_bins]
    else:
        train_data_bin_unlabeled = [x[num_labeled // 5: (num_labeled + num_unlabeled) // 5] for x in train_data_bins]
    train_data_bin_labeled = sum(train_data_bin_labeled, [])
    train_data_bin_unlabeled = sum(train_data_bin_unlabeled, [])

    X_l = np.array([x[0] for x in train_data_bin_labeled])
    y_l = np.array([x[1] for x in train_data_bin_labeled])
    X_u = np.array([x[0] for x in train_data_bin_unlabeled])
    y_u = np.array([x[1] for x in train_data_bin_unlabeled])

    # Prepare binarized testing data
    test_data_bin = []
    for i in range(len(test_data)):
        c = test_data[i][1]
        if c in dict_superclasses[superclass]:
            new_c = dict_superclasses[superclass].index(c)
            test_data_bin.append([test_data[i][0].numpy(), new_c])
    X_test = np.array([x[0] for x in test_data_bin])
    y_test = np.array([x[1] for x in test_data_bin])
    return X_l, y_l, X_u, y_u, X_test, y_test


# generate 20 5-way tasks for CIFAR-100, split by superclasses
# Thanks https://discuss.pytorch.org/t/cifar-100-targets-labels-doubt/81323/2 for providing superclasses dict
def generate_5_way_cifar100():
    task_parent_dir = os.path.join(TASK_ROOT, 'cifar100_5_way')
    if not os.path.exists(task_parent_dir):
        os.mkdir(task_parent_dir)
    train_data = datasets.CIFAR100(root=DATA_ROOT, train=True, download=True, transform=ToTensor())
    test_data = datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=ToTensor())
    for superclass in range(20):
        X_l, y_l, X_u, y_u, X_test, y_test = split_by_cifar100_superclasses(train_data, test_data, superclass=superclass)
        save_task_data(task_parent_dir, X_l, y_l, X_u, y_u, X_test, y_test, str(superclass))
    return


def generate_task_data_main():
    # generate lml tasks for semi-heterogeneous experiments
    generate_binary_task_data('mnist')
    generate_binary_task_data('fashion')
    generate_binary_task_data('cifar10')
    # generate lml tasks for instance-incremental experiments
    generate_5_way_mnist()
    generate_10_way_cifar10()
    # generate lml tasks for class-incremental experiments
    generate_5_way_cifar100()


if __name__ == '__main__':
    generate_task_data_main()


