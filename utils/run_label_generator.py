import os
import numpy as np
import sys
from labeler.label_generator import LabelGenerator

TASK_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'task_data')


# Open the directory of one task, generate y_u_prime and logit_u_prime and save to the save directory
def generate_label_for_one_task(task_dir, task, bootstrap_size_per_class=50):

    X_l = np.load(os.path.join(task_dir, 'X_l.npy'))
    y_l = np.load(os.path.join(task_dir, 'y_l.npy'))
    X_u = np.load(os.path.join(task_dir, 'X_u.npy'))
    y_u = np.load(os.path.join(task_dir, 'y_u.npy'))

    lg = LabelGenerator(X_u=X_u, y_u=y_u, X_l=X_l, y_l=y_l, task=task, num_guesses=100, keep=20,
                        max_labelers=25, min_labelers=10, init_acc_threshold=0.9,
                        bootstrap_size_per_class=bootstrap_size_per_class)
    task_parent_dir = os.path.dirname(task_dir)
    logit_u, y_u_prime, logit_l, y_l_prime = lg.generate_labels(log_file=os.path.join(task_parent_dir, 'log.txt'))

    np.save(os.path.join(task_dir, 'logit_u_prime.npy'), logit_u)
    np.save(os.path.join(task_dir, 'y_u_prime.npy'), y_u_prime)
    np.save(os.path.join(task_dir, 'logit_l_prime.npy'), logit_l)
    np.save(os.path.join(task_dir, 'y_l_prime.npy'), y_l_prime)
    print(task_dir + ': label generated')
    return


# Generate y_u_prime and logit_u_prime for binary classification tasks
def generate_label_for_binary_classification(task):
    if task == 'mnist':
        task_parent_dir = os.path.join(TASK_ROOT, 'mnist_bin')
        bootstrap_size_per_class = 15
    elif task == 'cifar10':
        task_parent_dir = os.path.join(TASK_ROOT, 'cifar10_bin')
        bootstrap_size_per_class = 175
    else:
        raise NotImplementedError
    for c0 in range(0, 9):
        for c1 in range(c0 + 1, 10):
            task_dir = os.path.join(task_parent_dir, str(c0) + '_' + str(c1))
            generate_label_for_one_task(task_dir=task_dir, task=task, bootstrap_size_per_class=bootstrap_size_per_class)
    return


def generate_mnist_5_way_labels():
    task_parent_dir = os.path.join(TASK_ROOT, 'mnist_5_way')
    for task in ['0-4', '5-9']:
        task_dir = os.path.join(task_parent_dir, task)
        generate_label_for_one_task(task_dir, 'mnist_5_way', bootstrap_size_per_class=50)
    return


def generate_cifar100_5_way_labels():
    task_parent_dir = os.path.join(TASK_ROOT, 'cifar100_5_way')
    for task in range(20):
        task_dir = os.path.join(task_parent_dir, str(task))
        generate_label_for_one_task(task_dir, 'cifar100_5_way', bootstrap_size_per_class=200)
    return


def generate_cifar10_10_way_labels():
    task_parent_dir = os.path.join(TASK_ROOT, 'cifar10_10_way')
    task_dir = os.path.join(task_parent_dir, '0-9')
    generate_label_for_one_task(task_dir, 'cifar10_10_way', bootstrap_size_per_class=100)
    return


# Wrapper function for all
def generate_all_labels():
    generate_label_for_binary_classification('mnist')
    generate_label_for_binary_classification('cifar10')
    generate_mnist_5_way_labels()
    generate_cifar100_5_way_labels()
    generate_cifar10_10_way_labels()


if __name__ == '__main__':

    # Example usage
    if len(sys.argv) < 2:
        dataset = 'mnist'
    else:
        dataset = sys.argv[1]

    if dataset in ['mnist', 'cifar10']:
        generate_label_for_binary_classification(dataset)
    elif dataset == 'mnist_5_way':
        generate_mnist_5_way_labels()
    elif dataset == 'cifar100_5_way':
        generate_cifar100_5_way_labels()
    elif dataset == 'cifar10_10_way':
        generate_cifar10_10_way_labels()
    else:
        raise NotImplementedError
