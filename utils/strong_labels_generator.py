import os
import sys
import numpy as np
import torch
from scipy.stats import mode
from snorkel.labeling.model.label_model import LabelModel
from mako.labeler.label_generator import LabelGenerator
from mako.labeler.lenet_weak_labeler import LeNetWeakLabeler
from mako.labeler.cifar_weak_labeler import CifarWeakLabeler
from mako.labeler.temp_scaling import tstorch_calibrate
if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

TASK_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'task_data')


# Obtain labelers
def get_labelers(task, task_dir):
    # Read input lfs
    lfs = []
    for i in range(0, 100):
        # print('loading lf ' + str(i))
        if task == 'mnist' or task == 'fashion':
            lf = LeNetWeakLabeler()
        elif task == 'cifar10':
            lf = CifarWeakLabeler()
        else:
            raise NotImplementedError
        if not os.path.exists(os.path.join(task_dir, 'lf' + str(i) + '.pt')):
            break
        else:
            lf.load_state_dict(torch.load(os.path.join(task_dir, 'lf' + str(i) + '.pt')))
            lf.eval()
            lf.to(DEVICE)
            lfs.append(lf)
    return lfs


# Obtain label matrix
def get_label_matrix(task, task_dir, lfs, n_u=None):
    if not os.path.exists(task_dir):
        raise FileNotFoundError

    # Read input data and labels
    X_l = np.load(os.path.join(task_dir, 'X_l.npy'))
    y_l = np.load(os.path.join(task_dir, 'y_l.npy'))
    X_u = np.load(os.path.join(task_dir, 'X_u.npy'))
    y_u = np.load(os.path.join(task_dir, 'y_u.npy'))
    if n_u is not None:
        X_u = X_u[:n_u]
        y_u = y_u[:n_u]

    lg = LabelGenerator(X_l=X_l, y_l=y_l, X_u=X_u, y_u=y_u, task=task)

    # Generate label matrix
    L_u, L_l = lg.generate_label_matrices(lfs)
    return L_u, L_l, y_l


# Generate labels by majority voting
def majority_vote_labeling(L_u):
    y_u_prime_mv = mode(L_u, axis=1)[0].transpose().flatten()
    return y_u_prime_mv


# Generate labels by repeated labeling: find the 10% least confident labels and pass to next lf
def repeated_labeling(task_dir, lfs, n_u=None):
    if not os.path.exists(task_dir):
        raise FileNotFoundError

    X_u = np.load(os.path.join(task_dir, 'X_u.npy'))
    if n_u is not None:
        X_u = X_u[:n_u]
    y_u_prime_rl = None
    confidence = None
    idx = None
    k = int(X_u.shape[0] * 0.1)

    for lf in lfs:
        if idx is None:  # first lf, label every data
            prob_u = lf.prob_matrix(X_u)
            y_u_prime_rl = np.argmax(prob_u, axis=1)
            confidence = np.amax(prob_u, axis=1)
            idx = np.argpartition(confidence, k)
        else:  # after first lf, label data with low confidence
            prob_u = lf.prob_matrix(X_u[idx])
            y_u_prime_rl_lf = np.argmax(prob_u, axis=1)
            confidence_lf = np.amax(prob_u, axis=1)
            # update labels and confidence
            y_u_prime_rl[idx] = y_u_prime_rl_lf
            confidence[idx] = confidence_lf
            # obtain new idx
            idx = np.argpartition(confidence, k)

    return y_u_prime_rl


# Generate labels by Snorkel and calibrate by temperature scaling
def snorkel_labeling(L_u, L_l, y_l):
    L = np.concatenate((L_l, L_u), axis=0)
    snorkel_model = LabelModel(cardinality=2, verbose=False)
    snorkel_model.fit(L)
    y_snorkel_u, logit_u = snorkel_model.predict(L_u, return_probs=True)
    y_snorkel_l, logit_l = snorkel_model.predict(L_l, return_probs=True)
    logit_u_calibrated = tstorch_calibrate(val_logits=logit_l, val_ys=y_l.astype('int64'), logits=logit_u)
    y_u_prime_snorkel = np.argmax(logit_u_calibrated, axis=1)
    return y_u_prime_snorkel, logit_u_calibrated


# Generate labels, methods can be majority voting (mv), repeated labeling (rl) or snorkel
# The labels are generated for different sizes of X_U
def generate_label_for_binary_classification(task, method='mv'):

    task_parent_dir = os.path.join(TASK_ROOT, task)
    if task == 'mnist_bin' or task == 'fashion_bin':
        n_u_space = [30, 60, 120, None]  # None means all data
    elif task == 'cifar10_bin':
        n_u_space = [400, 800, 1600, None]
    else:
        raise NotImplementedError

    for c0 in range(0, 9):
        for c1 in range(c0 + 1, 10):
            task_dir = os.path.join(task_parent_dir, str(c0) + '_' + str(c1))
            lfs = get_labelers(task, task_dir)
            for n_u in n_u_space:
                if method == 'mv':
                    L_u, L_l, y_l = get_label_matrix(task, task_dir, lfs, n_u=n_u)
                    y_u_prime = majority_vote_labeling(L_u)
                elif method == 'rl':
                    y_u_prime = repeated_labeling(task_dir, lfs, n_u=n_u)
                elif method == 'snorkel':
                    L_u, L_l, y_l = get_label_matrix(task, task_dir, lfs, n_u=n_u)
                    y_u_prime, logit_u_prime = snorkel_labeling(L_u, L_l, y_l)
                else:
                    raise NotImplementedError
                if n_u is None:
                    np.save(os.path.join(task_dir, 'y_u_prime_' + method + '.npy'), y_u_prime)
                    print('Label generated for task ' + str(c0) + '_' + str(c1) + ' by ' + method)
                else:
                    np.save(os.path.join(task_dir, 'y_u_prime_' + method + '_' + str(n_u) + '.npy'), y_u_prime)
                    print('Label generated for task ' + str(c0) + '_' + str(c1) + ' by ' + method + ' n_u=' + str(n_u))
    return


# Corrupt pseudo labels by taking part of y_u and flipping at an error rate, for baseline comparison
# MNIST and Fashion: |X_l| == 120, |X_u| == 5880, do |X_u'| == 120, 2940, 5880, error == 0, 0.2, 0.5
# CIFAR-10: |X_l| == 400, |X_u| == 4600, do |X_u'| == 400, 2300, 4600, error == 0, 0.2, 0.5
def corrupt_pseudo_labels(dataset='mnist'):

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
            X_u_prime = np.load(os.path.join(task_dir, 'X_u_prime.npy'))  # edit file name here
            y_u_prime = np.load(os.path.join(task_dir, 'y_u_prime.npy'))  # edit file name here

            for n_u in n_u_space:
                X_u_prime = X_u_prime[:n_u]
                y_u_prime = y_u_prime[:n_u]
                for error in error_space:
                    rand_index = np.random.choice(y_u_prime.shape[0], int(y_u_prime.shape[0] * error))
                    y_u_prime_flipped = y_u_prime.copy()
                    y_u_prime_flipped[rand_index] = 1 - y_u_prime_flipped[rand_index]
                    np.save(os.path.join(task_dir, 'y_u_' + str(n_u) + '_' + str(error) + '.npy'), y_u_prime_flipped)
                # np.save(os.path.join(task_dir, 'X_u_' + str(n_u) + '.npy'), X_u_prime)

    return


if __name__ == '__main__':

    # Generate strong pseudo-labels on different data sets and
    if len(sys.argv) < 2:
        dataset = 'mnist_bin'
    else:
        dataset = sys.argv[1]
    if dataset in ['mnist_bin', 'cifar10_bin']:
        generate_label_for_binary_classification(dataset)
        corrupt_pseudo_labels(dataset)
    else:
        raise NotImplementedError
