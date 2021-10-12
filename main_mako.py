from os import getcwd, listdir, mkdir
from utils.generate_task_data import generate_all_data
from utils.run_label_generator import generate_all_labels
from lml.utils.utils_exp import model_setup
from lml.utils.utils_data import data_loader_multiclass_instance_incremental_rep, data_loader_binarytask_rep, data_loader_multitask_rep, mnist_data_print_info, cifar_data_print_info
from lml.classification.train_wrapper import train_run_for_each_model

_randomly_generated_task_orders_45t = [
    [41, 36, 22, 2, 21, 32, 30, 26, 4, 0, 10, 3, 29, 33, 39, 44, 5, 15, 23, 18, 31, 9, 35, 34, 37, 17, 25, 14, 20, 19, 40, 7, 43, 11, 42, 38, 1, 24, 28, 12, 16, 8, 6, 13, 27],
    [18, 29, 4, 5, 17, 3, 35, 7, 33, 11, 14, 43, 44, 2, 6, 42, 20, 30, 0, 26, 39, 24, 41, 1, 22, 21, 37, 40, 25, 28, 12, 36, 32, 34, 23, 15, 27, 16, 13, 9, 19, 31, 38, 8, 10],
    [8, 3, 10, 20, 0, 2, 42, 15, 36, 13, 19, 27, 38, 28, 21, 1, 32, 39, 17, 12, 26, 5, 24, 23, 43, 4, 25, 41, 7, 11, 18, 35, 40, 29, 16, 31, 9, 6, 34, 14, 33, 44, 37, 30, 22],
    [38, 43, 12, 0, 14, 2, 8, 30, 42, 40, 25, 5, 27, 33, 13, 6, 29, 9, 15, 37, 22, 4, 18, 35, 44, 23, 17, 41, 39, 20, 24, 21, 10, 19, 16, 31, 1, 26, 3, 28, 11, 7, 36, 32, 34],
    [32, 24, 31, 28, 4, 2, 30, 33, 35, 6, 18, 23, 26, 11, 5, 38, 36, 29, 15, 43, 42, 7, 12, 27, 19, 14, 20, 13, 39, 41, 1, 17, 10, 0, 40, 37, 44, 25, 21, 34, 9, 22, 8, 3, 16],
    [7, 31, 26, 8, 16, 37, 28, 6, 0, 17, 23, 35, 12, 30, 18, 10, 39, 44, 42, 1, 38, 22, 25, 41, 34, 32, 5, 19, 43, 2, 20, 29, 4, 33, 9, 3, 36, 14, 15, 24, 11, 40, 21, 13, 27],
    [4, 25, 21, 44, 10, 33, 18, 27, 28, 15, 13, 23, 29, 2, 24, 5, 38, 40, 3, 36, 17, 7, 19, 31, 34, 8, 42, 20, 22, 32, 37, 11, 16, 43, 0, 6, 26, 39, 41, 12, 1, 9, 30, 14, 35],
    [0, 30, 31, 1, 44, 19, 11, 42, 17, 21, 41, 9, 35, 25, 10, 43, 40, 24, 14, 4, 16, 3, 13, 20, 28, 32, 22, 27, 12, 8, 5, 36, 34, 15, 23, 6, 38, 18, 37, 39, 2, 26, 33, 7, 29],
    [40, 30, 17, 41, 23, 39, 9, 22, 28, 0, 5, 12, 1, 14, 38, 18, 34, 20, 10, 43, 25, 44, 35, 29, 15, 4, 32, 36, 31, 6, 3, 33, 11, 13, 24, 42, 26, 16, 37, 27, 21, 19, 2, 7, 8],
    [43, 41, 21, 22, 13, 14, 9, 30, 25, 2, 0, 34, 10, 38, 18, 29, 8, 23, 6, 15, 20, 7, 5, 17, 1, 4, 28, 36, 31, 24, 26, 11, 19, 35, 3, 39, 16, 42, 27, 40, 33, 32, 12, 44, 37],
    [23, 24, 11, 3, 35, 20, 28, 27, 36, 29, 43, 13, 38, 5, 15, 2, 19, 10, 39, 21, 12, 33, 16, 32, 30, 6, 0, 18, 1, 14, 25, 17, 22, 41, 8, 34, 37, 44, 4, 40, 7, 42, 9, 31, 26],
    [1, 29, 2, 11, 36, 16, 20, 34, 44, 26, 40, 37, 27, 13, 6, 5, 12, 15, 41, 39, 4, 24, 22, 23, 19, 38, 43, 3, 21, 10, 28, 32, 17, 42, 14, 31, 7, 25, 0, 9, 30, 8, 33, 18, 35],
    [14, 16, 40, 12, 3, 36, 38, 25, 30, 0, 13, 17, 34, 10, 43, 29, 44, 7, 37, 5, 11, 21, 35, 19, 6, 20, 22, 1, 31, 24, 28, 23, 32, 8, 9, 15, 26, 18, 2, 41, 42, 39, 33, 4, 27],
    [21, 40, 4, 41, 25, 30, 29, 12, 10, 38, 2, 13, 18, 27, 17, 1, 36, 19, 9, 15, 35, 24, 11, 33, 44, 16, 22, 5, 37, 31, 20, 23, 28, 7, 3, 0, 39, 34, 43, 6, 8, 26, 42, 32, 14],
    [42, 4, 5, 35, 19, 44, 7, 27, 24, 36, 40, 31, 20, 12, 34, 0, 33, 1, 17, 10, 21, 9, 38, 18, 30, 28, 29, 3, 11, 43, 13, 39, 14, 32, 25, 2, 37, 23, 41, 15, 6, 16, 8, 26, 22],
    [43, 29, 37, 1, 16, 40, 4, 14, 36, 30, 39, 13, 42, 25, 23, 12, 5, 10, 28, 32, 0, 20, 9, 15, 38, 26, 34, 8, 6, 24, 44, 17, 18, 22, 11, 33, 21, 2, 31, 7, 27, 41, 19, 35, 3],
    [37, 13, 32, 40, 35, 34, 16, 38, 43, 31, 18, 7, 12, 2, 8, 17, 5, 22, 23, 42, 3, 6, 0, 36, 1, 30, 9, 33, 14, 26, 29, 10, 39, 24, 20, 28, 15, 21, 11, 4, 19, 44, 25, 27, 41],
    [0, 6, 24, 34, 37, 20, 14, 12, 27, 9, 35, 38, 1, 31, 2, 13, 4, 15, 21, 32, 8, 11, 29, 7, 18, 41, 43, 3, 17, 16, 33, 44, 5, 22, 25, 36, 10, 40, 28, 30, 23, 26, 42, 19, 39],
    [43, 5, 17, 40, 18, 4, 24, 16, 25, 31, 23, 9, 34, 41, 8, 39, 36, 14, 19, 13, 12, 28, 44, 3, 21, 22, 42, 33, 11, 32, 7, 37, 29, 15, 20, 35, 27, 0, 38, 30, 1, 26, 6, 10, 2],
    [7, 11, 14, 2, 39, 42, 31, 22, 24, 9, 10, 21, 3, 1, 17, 12, 27, 29, 33, 15, 41, 36, 28, 5, 23, 34, 13, 43, 40, 8, 6, 25, 35, 38, 19, 32, 0, 30, 18, 37, 16, 26, 4, 44, 20],
    [2, 16, 43, 30, 12, 22, 25, 35, 7, 37, 21, 33, 32, 3, 14, 6, 44, 0, 39, 28, 9, 23, 34, 31, 36, 4, 27, 11, 29, 40, 24, 10, 26, 8, 17, 19, 18, 41, 13, 20, 1, 38, 42, 15, 5],
    [13, 36, 2, 32, 28, 42, 21, 35, 7, 37, 31, 44, 23, 11, 26, 8, 20, 19, 16, 34, 10, 17, 39, 24, 0, 15, 9, 30, 18, 6, 25, 29, 41, 22, 14, 12, 3, 1, 4, 43, 40, 5, 33, 38, 27],
    [37, 30, 2, 16, 5, 34, 31, 29, 13, 0, 21, 38, 19, 26, 39, 17, 7, 32, 6, 11, 43, 28, 22, 8, 20, 44, 15, 24, 25, 41, 42, 10, 40, 33, 36, 4, 3, 35, 23, 1, 18, 27, 12, 14, 9],
    [19, 21, 3, 8, 11, 0, 37, 44, 17, 26, 36, 12, 23, 20, 31, 25, 38, 15, 7, 10, 30, 34, 39, 9, 40, 35, 24, 32, 4, 29, 2, 6, 41, 33, 28, 14, 22, 1, 18, 27, 42, 16, 13, 5, 43],
    [43, 23, 0, 18, 24, 37, 29, 35, 14, 34, 3, 12, 31, 8, 33, 21, 10, 5, 42, 13, 40, 19, 22, 11, 27, 20, 15, 16, 44, 32, 26, 6, 17, 1, 36, 41, 38, 30, 7, 2, 28, 4, 39, 25, 9],
    [22, 17, 30, 31, 32, 24, 25, 44, 3, 29, 28, 12, 23, 14, 9, 11, 43, 19, 7, 5, 33, 27, 6, 15, 13, 36, 35, 26, 16, 21, 34, 37, 39, 18, 40, 4, 38, 8, 41, 10, 2, 20, 42, 1, 0],
    [35, 18, 2, 42, 6, 22, 44, 21, 20, 23, 41, 13, 33, 9, 1, 43, 0, 32, 28, 37, 16, 27, 39, 25, 17, 12, 29, 14, 11, 38, 15, 5, 26, 19, 31, 24, 3, 34, 30, 40, 36, 8, 7, 10, 4],
    [1, 2, 25, 33, 17, 34, 8, 19, 43, 13, 21, 5, 44, 16, 22, 24, 14, 4, 38, 6, 39, 3, 28, 9, 0, 30, 18, 41, 23, 36, 12, 42, 31, 40, 37, 20, 26, 10, 15, 35, 7, 11, 27, 32, 29],
    [2, 16, 10, 3, 22, 21, 0, 36, 39, 37, 13, 5, 31, 12, 42, 40, 35, 6, 26, 4, 15, 29, 20, 19, 28, 17, 33, 27, 9, 23, 14, 38, 1, 34, 7, 30, 18, 41, 43, 8, 44, 11, 32, 25, 24],
    [19, 1, 27, 36, 28, 41, 15, 18, 25, 34, 21, 17, 10, 3, 35, 2, 12, 43, 5, 14, 29, 4, 20, 42, 38, 33, 9, 13, 11, 24, 26, 31, 23, 0, 22, 40, 30, 6, 39, 44, 8, 37, 7, 16, 32],
    [29, 34, 36, 24, 15, 41, 5, 37, 28, 1, 22, 0, 12, 20, 42, 30, 21, 33, 9, 11, 38, 14, 7, 18, 19, 27, 25, 6, 16, 8, 35, 44, 40, 32, 2, 3, 23, 13, 43, 4, 26, 17, 31, 39, 10],
    [42, 6, 18, 5, 21, 28, 36, 40, 10, 39, 23, 20, 13, 4, 38, 7, 34, 16, 12, 9, 27, 32, 41, 17, 25, 2, 1, 29, 11, 15, 44, 19, 30, 8, 0, 14, 37, 43, 31, 33, 3, 35, 24, 26, 22],
    [36, 31, 39, 1, 29, 21, 32, 9, 8, 23, 40, 12, 42, 41, 38, 14, 4, 24, 28, 37, 0, 26, 11, 33, 17, 18, 3, 7, 20, 16, 13, 15, 43, 44, 34, 30, 22, 5, 25, 35, 27, 19, 2, 10, 6],
    [19, 4, 0, 3, 12, 26, 43, 37, 39, 13, 27, 22, 35, 38, 41, 23, 28, 18, 14, 6, 1, 34, 20, 44, 21, 15, 10, 7, 8, 32, 16, 30, 36, 29, 9, 11, 24, 42, 25, 5, 31, 40, 33, 2, 17],
    [42, 31, 18, 36, 38, 15, 6, 19, 5, 16, 11, 22, 35, 33, 12, 29, 44, 43, 26, 40, 17, 41, 37, 14, 34, 3, 39, 24, 8, 25, 1, 27, 7, 28, 4, 20, 10, 32, 9, 2, 13, 30, 0, 23, 21],
    [34, 29, 21, 37, 7, 23, 28, 12, 10, 5, 42, 9, 16, 25, 26, 13, 14, 24, 27, 1, 43, 15, 31, 17, 44, 19, 38, 2, 35, 20, 3, 18, 33, 22, 11, 39, 36, 32, 41, 40, 8, 30, 6, 0, 4],
    [24, 28, 13, 4, 26, 29, 27, 18, 44, 14, 16, 36, 22, 2, 15, 32, 12, 10, 38, 20, 25, 8, 6, 39, 3, 21, 17, 43, 9, 33, 37, 42, 34, 7, 19, 1, 0, 30, 40, 41, 35, 23, 31, 5, 11],
    [41, 37, 8, 34, 28, 16, 24, 25, 33, 43, 18, 19, 0, 10, 17, 2, 38, 20, 7, 5, 6, 36, 35, 1, 14, 11, 42, 39, 15, 32, 40, 21, 27, 3, 26, 30, 13, 31, 23, 4, 44, 22, 9, 12, 29],
    [4, 10, 17, 38, 27, 29, 30, 39, 44, 11, 20, 26, 9, 3, 2, 34, 13, 37, 23, 0, 28, 33, 7, 31, 14, 8, 16, 43, 19, 40, 5, 1, 24, 15, 35, 18, 12, 22, 25, 42, 6, 21, 36, 41, 32],
    [18, 0, 7, 23, 20, 25, 43, 3, 9, 42, 29, 31, 24, 22, 15, 28, 10, 19, 5, 39, 40, 8, 11, 2, 17, 26, 44, 32, 13, 4, 34, 33, 41, 21, 35, 30, 1, 14, 16, 6, 38, 27, 36, 37, 12]
]


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', help='GPU device ID', type=int, default=-1)
    parser.add_argument('--data_type', help='Type of Data (MNIST/FASHION/CIFAR10)', type=str, default='MNIST')
    parser.add_argument('--data_unlabel', help='Number of instances of unlabeled data for training', type=int, default=0)
    parser.add_argument('--use_true_label', help='Use true label of unlabeled data for training (instead of label from Mako labelers', action='store_true', default=False)
    parser.add_argument('--model_type', help='Architecture of Model(STL/SNN/HPS/DEN/TF/PROG)', type=str, default='STL')
    parser.add_argument('--save_mat_name', help='Name of file to save training results', type=str, default='delete_this.mat')
    parser.add_argument('--test_type', help='For hyper-parameter search', type=int, default=0)
    parser.add_argument('--task_order_type', help='Choose the sequence of tasks presented to LL model', type=int, default=0)
    parser.add_argument('--cnn_padtype_valid', help='Set CNN padding type VALID', action='store_false', default=True)
    parser.add_argument('--lifelong', help='Train in lifelong learning setting', action='store_true', default=False)
    parser.add_argument('--saveparam', help='Save parameter of NN', action='store_true', default=False)
    parser.add_argument('--savegraph', help='Save graph of NN', action='store_true', default=False)
    parser.add_argument('--tensorfactor_param_path', help='Path to parameters initializing tensor factorized model(below Result, above run0/run1/etc', type=str, default=None)
    parser.add_argument('--num_classes', help='Number of classes for each sub-task', type=int, default=2)

    parser.add_argument('--skip_connect_test_type', help='For testing several ways to make skip connections', type=int, default=0)
    parser.add_argument('--highway_connect_test_type', help='For testing several ways to make skip connections (highway net)', type=int, default=0)

    parser.add_argument('--num_clayers', help='Number of conv layers for Office-Home experiment', type=int, default=-1)
    parser.add_argument('--phase1_max_epoch', help='Number of epochs in training phase 1 of Hybrid DF-CNN auto sharing', type=int, default=100)

    parser.add_argument('--reset_prior', help='Reset prior of configs each epoch', action='store_true', default=False)
    parser.add_argument('--em_fix_maxC', help='EM analysis - configuration of max assigned probability', type=str, default='top1')
    parser.add_argument('--em_maxP', help='EM analysis - probability of the config with max assigned probability', type=float, default=0.9)
    parser.add_argument('--em_cnt_prior', help='EM method - prior probability is based on the count of mini-batch', action='store_true', default=False)
    parser.add_argument('--em_prior_init_cnt', help='EM method - initial cnt for count-based prior probability', type=float, default=1)

    parser.add_argument('--darts_approx_order', help='Order of approximation of DARTS', type=int, default=1)

    parser.add_argument('--data_augment', help='Do data augmentation in mini-batch', action='store_true', default=False)

    parser.add_argument('--instance_incremental', help='Run instance-incremental instead of class-incremental experiment (note: requires SSN model_type)', action='store_true', default=False)
    parser.add_argument('--ordisco_ci_baseline', help='Run ordicso baseline class-incremental experiment', action='store_true', default=False)
    parser.add_argument('--data_group', help='Run a specific data group for debugging, set to -1 to ignore', type=int, default=-1)

    args = parser.parse_args()

    gpu_device_num = args.gpu
    if gpu_device_num > -1:
        use_gpu = True
    else:
        use_gpu = False
    do_lifelong = args.lifelong

    if not 'Result' in listdir(getcwd()):
        mkdir('Result')

    mat_file_name = args.save_mat_name

    data_type, data_num_unlabel, use_true_label = args.data_type.lower(), args.data_unlabel, args.use_true_label
    data_hyperpara = {}
    data_hyperpara['num_train_group'] = 5
    data_hyperpara['multi_class_label'] = False

    train_hyperpara = {}
    train_hyperpara['improvement_threshold'] = 1.002      # for accuracy (maximizing it)
    train_hyperpara['patience_multiplier'] = 1.5

    train_hyperpara['stl_analysis'] = False
    train_hyperpara['LEEP_score'] = False

    train_hyperpara['em_cnt_prior'] = args.em_cnt_prior
    train_hyperpara['em_prior_init_cnt'] = args.em_prior_init_cnt

    train_hyperpara['data_augment'] = args.data_augment

    train_hyperpara['mutual_info'] = False
    train_hyperpara['mutual_info_alpha'] = 1.01
    train_hyperpara['mutual_info_kernel_h'] = 1.00
    train_hyperpara['mutual_info_kernel_h_backward'] = 1.00

    if 'mnist_mako' in data_type:
        data_hyperpara['image_dimension'] = [28, 28, 1]
        if args.instance_incremental:
            pass  # set dynamically after instance incremental batches are created based on amount of train data
        else:
            data_hyperpara['num_tasks'] = 45

        train_hyperpara['num_run_per_model'] = 20
        train_hyperpara['num_train_valid_data_group'] = 10
        train_hyperpara['train_valid_data_group'] = list(range(train_hyperpara['num_train_valid_data_group']))
        train_hyperpara['lr'] = 0.001
        train_hyperpara['lr_decay'] = 1.0/250.0

        if args.instance_incremental:
            pass  # Note: set later after number of "tasks" (instance-incremental batches) is determined by data loader
        else:
            train_task_order = _randomly_generated_task_orders_45t
            if do_lifelong:
                train_hyperpara['patience'] = 20
                train_hyperpara['learning_step_max'] = data_hyperpara['num_tasks'] * train_hyperpara['patience']
            else:
                train_hyperpara['patience'] = 500
                train_hyperpara['learning_step_max'] = 500

        if args.instance_incremental:
            info_file_name = 'mako_mnist_ii'
            train_data, validation_data, test_data = data_loader_multiclass_instance_incremental_rep(getcwd() + '/Data/mako_labels/mnist_inst_incr_labels', args.num_classes, info_file_name, train_hyperpara['num_train_valid_data_group'], data_num_unlabel, use_true_label=use_true_label, dataset=data_type)
            data_hyperpara['num_tasks'] = len(train_data[0])
            train_task_order = list(range(data_hyperpara['num_tasks']))
            if do_lifelong:
                train_hyperpara['patience'] = 5
                train_hyperpara['learning_step_max'] = data_hyperpara['num_tasks'] * train_hyperpara['patience']
            else:
                train_hyperpara['patience'] = 500
                train_hyperpara['learning_step_max'] = 500
        else:
            label_conf_threshold = 0.51
            info_file_name = 'mako_mnist_u'+ (str(data_num_unlabel) if data_num_unlabel > 0 else str(0))
            train_data, validation_data, test_data = data_loader_binarytask_rep(getcwd() + '/Data/mako_labels/mnist_labels', 10, info_file_name, train_hyperpara['num_train_valid_data_group'], data_num_unlabel, use_true_label=use_true_label, confidence_threshold=label_conf_threshold)

        mnist_data_print_info(train_data, validation_data, test_data)
        classification_prob=True

    elif 'cifar10_mako' in data_type:
        data_hyperpara['image_dimension'] = [32, 32, 3]
        if args.ordisco_ci_baseline:
            data_hyperpara['num_tasks'] = 5
            baseline_config = 1
        else:
            data_hyperpara['num_tasks'] = 45
            baseline_config = 0

        train_hyperpara['num_run_per_model'] = 1
        train_hyperpara['num_train_valid_data_group'] = 10
        if args.ordisco_ci_baseline:
            train_hyperpara['train_valid_data_group'] = list(range(train_hyperpara['num_train_valid_data_group']))
        else:
            train_hyperpara['train_valid_data_group'] = list(range(train_hyperpara['num_train_valid_data_group'])) + list(range(train_hyperpara['num_train_valid_data_group']))
        train_hyperpara['lr'] = 0.00025
        train_hyperpara['lr_decay'] = 1.0/1000.0
        if args.instance_incremental:
            pass  # Note: set later after number of "tasks" (instance-incremental batches) is determined by data loader
        else:
            if args.ordisco_ci_baseline:
                train_hyperpara['task_order'] = list(range(data_hyperpara['num_tasks']))
                train_task_order = train_hyperpara['task_order']
                if do_lifelong:
                    train_hyperpara['patience'] = 50
                    train_hyperpara['learning_step_max'] = data_hyperpara['num_tasks'] * train_hyperpara['patience']
                else:
                    train_hyperpara['patience'] = 2000
                    train_hyperpara['learning_step_max'] = 2000
            else:
                train_task_order = _randomly_generated_task_orders_45t
                if do_lifelong:
                    train_hyperpara['patience'] = 100
                    train_hyperpara['learning_step_max'] = data_hyperpara['num_tasks'] * train_hyperpara['patience']
                else:
                    train_hyperpara['patience'] = 2000
                    train_hyperpara['learning_step_max'] = 2000

        if args.instance_incremental:
            info_file_name = 'mako_cifar10_ii'
            train_data, validation_data, test_data = data_loader_multiclass_instance_incremental_rep(getcwd() + '/Data/mako_labels/cifar10_inst_incr_labels', args.num_classes, info_file_name, train_hyperpara['num_train_valid_data_group'], data_num_unlabel, use_true_label=use_true_label, dataset=data_type)
            data_hyperpara['num_tasks'] = len(train_data[0])
            train_task_order = list(range(data_hyperpara['num_tasks']))
            if do_lifelong:
                train_hyperpara['patience'] = 5
                train_hyperpara['learning_step_max'] = data_hyperpara['num_tasks'] * train_hyperpara['patience']
            else:
                train_hyperpara['patience'] = 500
                train_hyperpara['learning_step_max'] = 500
        else:
            label_conf_threshold = 0.51
            info_file_name = 'mako_cifar10_u' + (str(data_num_unlabel) if data_num_unlabel > 0 else str(0))
            train_data, validation_data, test_data = data_loader_binarytask_rep(getcwd() + '/Data/mako_labels/cifar10_labels', 10, info_file_name, train_hyperpara['num_train_valid_data_group'], data_num_unlabel, use_true_label=use_true_label, confidence_threshold=label_conf_threshold, baseline_config=baseline_config)
        cifar_data_print_info(train_data, validation_data, test_data)
        classification_prob=True

    elif 'cifar100_mako' in data_type:
        data_hyperpara['image_dimension'] = [32, 32, 3]
        data_hyperpara['num_tasks'] = 20

        train_hyperpara['num_run_per_model'] = 1
        train_hyperpara['num_train_valid_data_group'] = 10
        train_hyperpara['train_valid_data_group'] = list(range(train_hyperpara['num_train_valid_data_group']))
        train_hyperpara['lr'] = 0.00025
        train_hyperpara['lr_decay'] = 1.0/1000.0
        train_hyperpara['task_order'] = list(range(data_hyperpara['num_tasks']))
        train_task_order = train_hyperpara['task_order']
        if do_lifelong:
            train_hyperpara['patience'] = 25
            train_hyperpara['learning_step_max'] = data_hyperpara['num_tasks'] * train_hyperpara['patience']
        else:
            train_hyperpara['patience'] = 2000
            train_hyperpara['learning_step_max'] = 2000

        label_conf_threshold = 1/5 + .01
        info_file_name = 'mako_cifar100_u' + (str(data_num_unlabel) if data_num_unlabel > 0 else str(0))
        num_valid_per_task = 500
        train_data, validation_data, test_data = data_loader_multitask_rep(getcwd() + '/Data/mako_labels/cifar100_5_way_labels', 100, info_file_name, train_hyperpara['num_train_valid_data_group'], data_num_unlabel-num_valid_per_task, use_true_label=use_true_label, confidence_threshold=label_conf_threshold, classes_per_task=args.num_classes, num_validation_data=num_valid_per_task)
        cifar_data_print_info(train_data, validation_data, test_data)
        classification_prob=True

    else:
        raise ValueError("The given dataset has no pre-defined experiment design. Check dataset again!")


    train_hyperpara['em_analysis_maxC'], train_hyperpara['em_analysis_maxC_prob'] = args.em_fix_maxC, args.em_maxP
    train_hyperpara['reset_prior'] = args.reset_prior

    ## Model Set-up
    model_architecture, model_hyperpara = model_setup(data_type, data_hyperpara['image_dimension'], args.model_type, args.test_type, args.cnn_padtype_valid, args.skip_connect_test_type, args.highway_connect_test_type, args.num_clayers, args.phase1_max_epoch, args.darts_approx_order)
    train_hyperpara['num_tasks'] = data_hyperpara['num_tasks']
    print(model_architecture)

    saveparam = args.saveparam or 'bruteforce' in model_architecture
    save_param_path = None
    if saveparam:
        if not 'params' in listdir(getcwd()+'/Result'):
            mkdir('./Result/params')
        save_param_dir_name = data_type + '_' + str(data_num_unlabel) + 'u_' + args.model_type + '_t' + str(args.test_type)
        if args.highway_connect_test_type > 0:
            save_param_dir_name += '_h' + str(args.highway_connect_test_type)
        elif args.skip_connect_test_type > 0:
            save_param_dir_name += '_s' + str(args.skip_connect_test_type)
        while save_param_dir_name in listdir(getcwd()+'/Result/params'):
            save_param_dir_name += 'a'
        save_param_path = getcwd()+'/Result/params/'+save_param_dir_name
        mkdir(save_param_path)

    # Training the Model
    saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, mat_file_name, classification_prob, saved_result=None, useGPU=use_gpu, GPU_device=gpu_device_num, doLifelong=do_lifelong, saveParam=saveparam, saveParamDir=save_param_path, saveGraph=args.savegraph, tfInitParamPath=None, task_order=train_task_order, debug_data_group=args.data_group)


if __name__ == '__main__':
    generate_all_data()
    generate_all_labels()
    main()