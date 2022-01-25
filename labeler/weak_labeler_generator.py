"""
Step 1, generating weak labelers by a modified version of Snuba while maintaining guarantee
"""

import os
import numpy as np
import sys
import torch
from labeler.label_generator import LabelGenerator
if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

TASK_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'task_data')


# Open the directory of one task, generate weak labelers for that directory
def generate_labeler_for_one_task(task_dir, task):

    X_l = np.load(os.path.join(task_dir, 'X_l.npy'))
    y_l = np.load(os.path.join(task_dir, 'y_l.npy'))
    X_u = np.load(os.path.join(task_dir, 'X_u.npy'))
    y_u = np.load(os.path.join(task_dir, 'y_u.npy'))

    # tuned hyperparams, don't change!
    if task in ['mnist', 'fashion']:
        init_acc_threshold = 0.85
        bootstrap_size_per_class = 30
    elif task in ['cifar10']:
        init_acc_threshold = 0.8
        bootstrap_size_per_class = 350
    elif task in ['mnist_5_way']:
        init_acc_threshold = 0.8
        bootstrap_size_per_class = 50
    elif task in ['cifar10_10_way']:
        init_acc_threshold = 0.6
        bootstrap_size_per_class = 750
    elif task in ['cifar100_5_way']:
        init_acc_threshold = 0.6
        bootstrap_size_per_class = 200
    else:
        raise NotImplementedError

    # guess 100 times for each weak labeler, initially keep 20, then add keep 1 each time
    # max num weak labelers = 25, min = 5
    lg = LabelGenerator(X_u=X_u, y_u=y_u, X_l=X_l, y_l=y_l, task=task, num_guesses=100, keep=20,
                        max_labelers=25, min_labelers=5, init_acc_threshold=init_acc_threshold,
                        bootstrap_size_per_class=bootstrap_size_per_class)
    task_parent_dir = os.path.dirname(task_dir)
    hfs = lg.generate_snuba_lfs(log_file=os.path.join(task_parent_dir, 'log.txt'))

    for i in range(len(hfs)):
        torch.save(hfs[i].state_dict(), os.path.join(task_dir, 'lf' + str(i) + '.pt'))
    print(task_dir + ': ' + str(len(hfs)) + ' labelers saved')
    return


# Generate and save weak labelers by modified Snuba
def generate_weak_labelers(task):
    if task in ['mnist', 'fashion', 'cifar10']:
        task_parent_dir = os.path.join(TASK_ROOT, task + '_bin')
    elif task in ['mnist_5_way', 'cifar10_10_way', 'cifar100_5_way']:
        task_parent_dir = os.path.join(TASK_ROOT, task)
    else:
        raise NotImplementedError

    for d in os.listdir(task_parent_dir):
        task_dir = os.path.join(task_parent_dir, d)
        generate_labeler_for_one_task(task_dir=task_dir, task=task)
    return


def generate_weak_labelers_main(args):
    if len(args) < 2:
        dataset = 'mnist'
    else:
        dataset = args[1]
    if dataset in ['mnist', 'fashion', 'cifar10', 'mnist_5_way', 'cifar10_10_way', 'cifar100_5_way']:
        generate_weak_labelers(dataset)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    generate_weak_labelers_main(sys.argv)

