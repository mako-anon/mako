"""
Helper functions to inspect labeling accuracy for the strong labeler by data programming
See main() for usage
"""

import os
import numpy as np
from matplotlib import pyplot as plt

TASK_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'task_data')
FIG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'figs')


def listdirs(rootdir):
    l = []
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            l.append(d)
    return l


def task_labeling_accuracy(y_u: np.array, y_u_prime: np.array):
    num_correct = np.sum(y_u == y_u_prime)
    accuracy = float(num_correct) / y_u.shape[0]
    return accuracy


def evaluate_labeling_accuracy(task='mnist'):
    if task == 'mnist':
        task_parent_dir = os.path.join(TASK_ROOT, 'mnist_bin')
    elif task == 'fashion':
        task_parent_dir = os.path.join(TASK_ROOT, 'fashion_bin')
    elif task == 'cifar10':
        task_parent_dir = os.path.join(TASK_ROOT, 'cifar10_bin')
    else:
        raise NotImplementedError

    accuracies = []
    task_dirs = listdirs(task_parent_dir)

    for d in task_dirs:
        task_dir = os.path.join(task_parent_dir, d)
        y_u = np.load(os.path.join(task_dir, 'y_u.npy'))
        y_u_prime = np.load(os.path.join(task_dir, 'y_u_prime.npy'))
        accuracy = task_labeling_accuracy(y_u, y_u_prime)
        accuracies.append(accuracy)
    return accuracies


if __name__ == '__main__':

    # Some example uses

    # print labeling accuracies
    datasets = ['mnist', 'cifar10', 'cifar10_10_way', 'cifar100_5_way', 'mnist_5_way']
    overall_accs = []
    for i in range(len(datasets)):
        dataset = datasets[i]
        accuracies = evaluate_labeling_accuracy(dataset)
        overall_accs.append(accuracies)
        print("Labeling accuracy of " + dataset + ": ")
        print(accuracies)

    # plot labeling accuracies
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig = plt.figure(figsize=(6, 4), dpi=200)
    bplot = plt.boxplot(overall_accs, patch_artist=True)
    plt.xticks([1, 2, 3, 4, 5], ['Bin MNIST', 'Bin CIFAR-10', '10-way CIFAR-10',
                                       '5-way CIFAR-100', '5-way MNIST'])
    plt.xticks(rotation=15)
    plt.ylabel('Accuracy')
    colors = ['pink', 'lightblue', 'wheat', 'antiquewhite', 'plum']
    for i in range(len(bplot['boxes'])):
        bplot['boxes'][i].set_facecolor(colors[i])

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_ROOT, 'labeling_accs.png'))
