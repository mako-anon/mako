"""
Step 1, generating weak labelers by a modified version of Snuba while maintaining guarantee
"""

import os
import numpy as np
import sys
import torch
from mako.labeler.label_generator import LabelGenerator
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

    if task == 'mnist' or task == 'fashion':
        init_acc_threshold = 0.85
        bootstrap_size_per_class = 30
    elif task == 'cifar10':
        init_acc_threshold = 0.8
        bootstrap_size_per_class = 350
    else:
        raise NotImplementedError
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
def generate_labeler_for_binary_classification(task):
    if task == 'mnist':
        task_parent_dir = os.path.join(TASK_ROOT, 'mnist_bin')
    elif task == 'fashion':
        task_parent_dir = os.path.join(TASK_ROOT, 'fashion_bin')
    elif task == 'cifar10':
        task_parent_dir = os.path.join(TASK_ROOT, 'cifar10_bin')
    else:
        raise NotImplementedError

    for c0 in range(0, 9):
        for c1 in range(c0 + 1, 10):
            task_dir = os.path.join(task_parent_dir, str(c0) + '_' + str(c1))
            generate_labeler_for_one_task(task_dir=task_dir, task=task)
    return


if __name__ == '__main__':
    if len(sys.argv) < 2:
        dataset = 'mnist'
    else:
        dataset = sys.argv[1]
    if dataset in ['mnist', 'cifar10']:
        generate_labeler_for_binary_classification(dataset)
    else:
        raise NotImplementedError

