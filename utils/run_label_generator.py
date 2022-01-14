import os
import numpy as np
import sys
import torch
from scipy.stats import mode
from snorkel.labeling.model.label_model import LabelModel
from labeler.label_generator import LabelGenerator
from labeler.lenet_weak_labeler import LeNetWeakLabeler
from labeler.cifar_weak_labeler import CifarWeakLabeler
from labeler.temp_scaling import tstorch_calibrate
if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

TASK_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'task_data')


# Open the directory of one task, generate Snuba labeler for that directory
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


# Generate Snuba labeler
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
    if n_u is None:
        X_u = np.load(os.path.join(task_dir, 'X_u.npy'))
        y_u = np.load(os.path.join(task_dir, 'y_u.npy'))
    else:
        X_u = np.load(os.path.join(task_dir, 'X_u_' + str(n_u) + '.npy'))
        y_u = np.load(os.path.join(task_dir, 'y_u_' + str(n_u) + '_0.npy'))

    lg = LabelGenerator(X_l=X_l, y_l=y_l, X_u=X_u, y_u=y_u, task=task)

    # Generate label matrix
    L_u, L_l = lg.generate_label_matrices(lfs)
    return L_u, L_l, y_l


# Generate labels by majority voting
def majority_vote_labeling(L_u):
    y_u_prime_mv = mode(L_u, axis=1)[0].transpose().flatten()
    return y_u_prime_mv


# Generate labels by repeated labeling: find the 10% least confident labels and pass to next lf
def repeated_labeling(task_dir, lfs):
    if not os.path.exists(task_dir):
        raise FileNotFoundError

    X_u = np.load(os.path.join(task_dir, 'X_u.npy'))
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


# Generate labels by Snorkel
def snorkel_labeling(L_u, L_l, y_l):
    L = np.concatenate((L_l, L_u), axis=0)
    snorkel_model = LabelModel(cardinality=2, verbose=False)
    snorkel_model.fit(L)
    y_snorkel_u, logit_u = snorkel_model.predict(L_u, return_probs=True)
    y_snorkel_l, logit_l = snorkel_model.predict(L_l, return_probs=True)
    logit_u_calibrated = tstorch_calibrate(val_logits=logit_l, val_ys=y_l.astype('int64'), logits=logit_u)
    y_u_prime_snorkel = np.argmax(logit_u_calibrated, axis=1)
    return y_u_prime_snorkel, logit_u_calibrated


# Generate labels
def generate_label_for_binary_classification(task):
    if task == 'mnist':
        task_parent_dir = os.path.join(TASK_ROOT, 'mnist_bin')
        n_u_set = [30, 60, 120]
    elif task == 'fashion':
        task_parent_dir = os.path.join(TASK_ROOT, 'fashion_bin')
        n_u_set = [30, 60, 120]
    elif task == 'cifar10':
        task_parent_dir = os.path.join(TASK_ROOT, 'cifar10_bin')
        n_u_set = [400, 800, 1600]
    else:
        raise NotImplementedError
    for c0 in range(0, 9):
        for c1 in range(c0 + 1, 10):
            task_dir = os.path.join(task_parent_dir, str(c0) + '_' + str(c1))
            lfs = get_labelers(task, task_dir)
            y_u_prime_rl = repeated_labeling(task_dir, lfs)
            # L_u, L_l, y_l = get_label_matrix(task, task_dir, lfs)
            # y_u_prime_mv = majority_vote_labeling(L_u)
            # y_u_prime_snorkel, logit_u_prime_snorkel = snorkel_labeling(L_u, L_l, y_l)
            # np.save(os.path.join(task_dir, 'y_u_prime_mv.npy'), y_u_prime_mv)
            # np.save(os.path.join(task_dir, 'y_u_prime_snorkel_.npy'), y_u_prime_snorkel)
            np.save(os.path.join(task_dir, 'y_u_prime_rl.npy'), y_u_prime_rl)
            print('Label generated for task ' + str(c0) + '_' + str(c1))
    return


if __name__ == '__main__':
    if len(sys.argv) < 2:
        dataset = 'mnist'
    else:
        dataset = sys.argv[1]
    if dataset in ['mnist', 'fashion', 'cifar10']:
        generate_label_for_binary_classification(dataset)
    else:
        raise NotImplementedError

