import pickle
import numpy as np
import copy
from os import getcwd, listdir, mkdir

# Added baseline_config parameter to override experiments for specific baselines: {0: no change, 1: ORDISCO class-incremental}
def data_loader_binarytask_rep(path_to_data, num_classes, info_file_name, num_repetition, num_unlabel_data, num_validation_data=-1, flatten_img=True, use_true_label=False, confidence_threshold=0.0, baseline_config=0):
    train_data_rep, valid_data_rep = [], []
    for rep_cnt in range(num_repetition):
        data_file_name = info_file_name+'_rep'+str(rep_cnt)+'.pkl' if not use_true_label else info_file_name+'_T_rep'+str(rep_cnt)+'.pkl'
        train_data, valid_data, test_data = data_loader_binarytask(path_to_data, num_classes, data_file_name, num_unlabel_data, num_validation_data=num_validation_data, flatten_img=flatten_img, use_true_label=use_true_label, confidence_threshold=confidence_threshold, baseline_config=baseline_config)
        train_data_rep.append(train_data)
        valid_data_rep.append(valid_data)
        print('Complete loading data for repetition %d'%(rep_cnt))
    return train_data_rep, valid_data_rep, test_data

def data_loader_binarytask(path_to_data, num_classes, info_file_name, num_unlabel_data, num_validation_data=-1, flatten_img=True, use_true_label=False, confidence_threshold=0.0, baseline_config=0):
    def index_of_second_class_binarytask(label_data):
        return_index = -1
        for i in range(1, label_data.shape[0]):
            if label_data[i-1] == 0 and label_data[i] == 1:
                return_index = i
                break
        if return_index > 0:
            return return_index
        else:
            raise ValueError

    def get_randomized_indices(num_total, num_selection, num_selection2, confidence_threshold, logits=None):
        # Get small fraction of training instances in original set
        list_of_indices = np.arange(num_total)
        np.random.shuffle(list_of_indices)
        if logits is None:
            return_list1, return_list2 = list_of_indices[:num_selection], list_of_indices[num_selection:num_selection+num_selection2]
        else:
            return_list1, return_list2 = [], []
            for ind in list_of_indices:
                if logits[ind] > confidence_threshold and len(return_list1) < num_selection:
                    return_list1.append(ind)
                elif logits[ind] > confidence_threshold and len(return_list2) < num_selection2:
                    return_list2.append(ind)
                if len(return_list1) >= num_selection and len(return_list2) >= num_selection2:
                    return_list1, return_list2 = np.array(return_list1, dtype=np.int), np.array(return_list2, dtype=np.int)
                    break
        return return_list1, return_list2

    if num_validation_data < 0:
        num_validation_data = num_unlabel_data//2

    if info_file_name in listdir(path_to_data):
        with open(path_to_data+'/'+info_file_name, 'rb') as fobj:
            task_data_info = pickle.load(fobj)
        make_new_info_file = False
    else:
        task_data_info, make_new_info_file = {'train': {}, 'validation': {}}, True

    train_data, validation_data, test_data = [], [], []
    if baseline_config == 1:  # Special case for class selection with ORDISCO baseline experiments
        for class_cnt in range(num_classes//2):
            class_cnt0 = class_cnt*2
            class_cnt1 = class_cnt0 + 1
            x_l = np.load(path_to_data+'/'+str(class_cnt0)+'_'+str(class_cnt1)+'/X_l.npy')
            x_u = np.load(path_to_data+'/'+str(class_cnt0)+'_'+str(class_cnt1)+'/X_u.npy')
            x_test = np.load(path_to_data+'/'+str(class_cnt0)+'_'+str(class_cnt1)+'/X_test.npy')

            y_l = np.load(path_to_data + '/' + str(class_cnt0) + '_' + str(class_cnt1) + '/y_l.npy')
            y_u = np.load(path_to_data + '/' + str(class_cnt0) + '_' + str(class_cnt1) + '/y_u.npy')
            y_u_prime = np.load(path_to_data + '/' + str(class_cnt0) + '_' + str(class_cnt1) + '/y_u_prime.npy')
            y_test = np.load(path_to_data + '/' + str(class_cnt0) + '_' + str(class_cnt1) + '/y_test.npy')

            logit_u_prime = np.load(path_to_data + '/' + str(class_cnt0) + '_' + str(class_cnt1) + '/logit_u_prime.npy')

            start_index_second_class = index_of_second_class_binarytask(y_u)
            if make_new_info_file:
                num_u_class0, num_u_class1 = start_index_second_class, y_u.shape[0]-start_index_second_class
                if (str(class_cnt0) not in task_data_info['train'].keys()) or (str(class_cnt0) not in task_data_info['validation'].keys()):
                    index_u_train, index_valid = get_randomized_indices(num_u_class0, num_unlabel_data//2, max(num_validation_data//2, 30), confidence_threshold, logits=logit_u_prime[0:start_index_second_class, 0])
                    task_data_info['train'][str(class_cnt0)] = index_u_train
                    task_data_info['validation'][str(class_cnt0)] = index_valid
                if (str(class_cnt1) not in task_data_info['train'].keys()) or (str(class_cnt1) not in task_data_info['validation'].keys()):
                    index_u_train, index_valid = get_randomized_indices(num_u_class1, num_unlabel_data//2, max(num_validation_data//2, 30), confidence_threshold, logits=logit_u_prime[start_index_second_class:, 1])
                    task_data_info['train'][str(class_cnt1)] = index_u_train
                    task_data_info['validation'][str(class_cnt1)] = index_valid

            indices_train_unlabeled = list(task_data_info['train'][str(class_cnt0)]) + [a+start_index_second_class for a in task_data_info['train'][str(class_cnt1)]]
            indices_valid_unlabeled = list(task_data_info['validation'][str(class_cnt0)]) + [a+start_index_second_class for a in task_data_info['validation'][str(class_cnt1)]]
            train_x, valid_x = np.transpose(np.concatenate((x_l, x_u[indices_train_unlabeled]), axis=0), (0, 2, 3, 1)), np.transpose(x_u[indices_valid_unlabeled], (0, 2, 3, 1))
            if use_true_label:
                train_y, valid_y = np.concatenate((y_l, y_u[indices_train_unlabeled]), axis=0), y_u[indices_valid_unlabeled]
            else:
                train_y, valid_y = np.concatenate((y_l, y_u_prime[indices_train_unlabeled]), axis=0), y_u_prime[indices_valid_unlabeled]
            test_x = np.transpose(x_test, (0, 2, 3, 1))
            if flatten_img:
                num_train, num_valid, num_test = train_x.shape[0], valid_x.shape[0], test_x.shape[0]
                train_data.append( (train_x.reshape([num_train, -1]), train_y) )
                validation_data.append( (valid_x.reshape([num_valid, -1]), valid_y) )
                test_data.append( (test_x.reshape([num_test, -1]), y_test) )
            else:
                train_data.append( (train_x, train_y) )
                validation_data.append( (valid_x, valid_y) )
                test_data.append( (test_x, y_test) )
    else:
        for class_cnt0 in range(num_classes-1):
            for class_cnt1 in range(class_cnt0+1, num_classes):
                x_l = np.load(path_to_data+'/'+str(class_cnt0)+'_'+str(class_cnt1)+'/X_l.npy')
                x_u = np.load(path_to_data+'/'+str(class_cnt0)+'_'+str(class_cnt1)+'/X_u.npy')
                x_test = np.load(path_to_data+'/'+str(class_cnt0)+'_'+str(class_cnt1)+'/X_test.npy')

                y_l = np.load(path_to_data + '/' + str(class_cnt0) + '_' + str(class_cnt1) + '/y_l.npy')
                y_u = np.load(path_to_data + '/' + str(class_cnt0) + '_' + str(class_cnt1) + '/y_u.npy')
                y_u_prime = np.load(path_to_data + '/' + str(class_cnt0) + '_' + str(class_cnt1) + '/y_u_prime.npy')
                y_test = np.load(path_to_data + '/' + str(class_cnt0) + '_' + str(class_cnt1) + '/y_test.npy')

                logit_u_prime = np.load(path_to_data + '/' + str(class_cnt0) + '_' + str(class_cnt1) + '/logit_u_prime.npy')

                start_index_second_class = index_of_second_class_binarytask(y_u)
                if make_new_info_file:
                    num_u_class0, num_u_class1 = start_index_second_class, y_u.shape[0]-start_index_second_class
                    if (str(class_cnt0) not in task_data_info['train'].keys()) or (str(class_cnt0) not in task_data_info['validation'].keys()):
                        index_u_train, index_valid = get_randomized_indices(num_u_class0, num_unlabel_data//2, max(num_validation_data//2, 30), confidence_threshold, logits=logit_u_prime[0:start_index_second_class, 0])
                        task_data_info['train'][str(class_cnt0)] = index_u_train
                        task_data_info['validation'][str(class_cnt0)] = index_valid
                    if (str(class_cnt1) not in task_data_info['train'].keys()) or (str(class_cnt1) not in task_data_info['validation'].keys()):
                        index_u_train, index_valid = get_randomized_indices(num_u_class1, num_unlabel_data//2, max(num_validation_data//2, 30), confidence_threshold, logits=logit_u_prime[start_index_second_class:, 1])
                        task_data_info['train'][str(class_cnt1)] = index_u_train
                        task_data_info['validation'][str(class_cnt1)] = index_valid

                indices_train_unlabeled = list(task_data_info['train'][str(class_cnt0)]) + [a+start_index_second_class for a in task_data_info['train'][str(class_cnt1)]]
                indices_valid_unlabeled = list(task_data_info['validation'][str(class_cnt0)]) + [a+start_index_second_class for a in task_data_info['validation'][str(class_cnt1)]]
                train_x, valid_x = np.transpose(np.concatenate((x_l, x_u[indices_train_unlabeled]), axis=0), (0, 2, 3, 1)), np.transpose(x_u[indices_valid_unlabeled], (0, 2, 3, 1))
                if use_true_label:
                    train_y, valid_y = np.concatenate((y_l, y_u[indices_train_unlabeled]), axis=0), y_u[indices_valid_unlabeled]
                else:
                    train_y, valid_y = np.concatenate((y_l, y_u_prime[indices_train_unlabeled]), axis=0), y_u_prime[indices_valid_unlabeled]
                test_x = np.transpose(x_test, (0, 2, 3, 1))
                if flatten_img:
                    num_train, num_valid, num_test = train_x.shape[0], valid_x.shape[0], test_x.shape[0]
                    train_data.append( (train_x.reshape([num_train, -1]), train_y) )
                    validation_data.append( (valid_x.reshape([num_valid, -1]), valid_y) )
                    test_data.append( (test_x.reshape([num_test, -1]), y_test) )
                else:
                    train_data.append( (train_x, train_y) )
                    validation_data.append( (valid_x, valid_y) )
                    test_data.append( (test_x, y_test) )

    if make_new_info_file:
        with open(path_to_data+'/'+info_file_name, 'wb') as fobj:
            pickle.dump(task_data_info, fobj)
    return train_data, validation_data, test_data


def data_loader_multitask_rep(path_to_data, num_classes, info_file_name, num_repetition, num_unlabel_data, num_validation_data=-1, flatten_img=True, use_true_label=False, confidence_threshold=0.0, classes_per_task=5):
    train_data_rep, valid_data_rep = [], []
    for rep_cnt in range(num_repetition):
        data_file_name = info_file_name+'_rep'+str(rep_cnt)+'.pkl' if not use_true_label else info_file_name+'_T_rep'+str(rep_cnt)+'.pkl'
        train_data, valid_data, test_data = data_loader_multitask(path_to_data, num_classes, data_file_name, num_unlabel_data, num_validation_data=num_validation_data, flatten_img=flatten_img, use_true_label=use_true_label, confidence_threshold=confidence_threshold, classes_per_task=classes_per_task)
        train_data_rep.append(train_data)
        valid_data_rep.append(valid_data)
        print('Complete loading data for repetition %d'%(rep_cnt))
    return train_data_rep, valid_data_rep, test_data

def data_loader_multitask(path_to_data, num_classes, info_file_name, num_unlabel_data, num_validation_data=-1, flatten_img=True, use_true_label=False, confidence_threshold=0.0, classes_per_task=5):
    def index_of_class(label_data, c_ind):
        return_index = -1
        for i in range(1, label_data.shape[0]):
            if label_data[i-1] == c_ind-1 and label_data[i] == c_ind:
                return_index = i
                break
        if return_index > 0:
            return return_index
        else:
            raise ValueError

    def get_randomized_indices(num_total, num_selection, num_selection2):
        list_of_indices = np.arange(num_total)
        np.random.shuffle(list_of_indices)
        return list_of_indices[:num_selection], list_of_indices[num_selection:num_selection+num_selection2]

    if num_validation_data < 0:
        num_validation_data = num_unlabel_data//2

    num_unlabeled_per_class = num_unlabel_data // classes_per_task
    num_valid_per_class = num_validation_data // classes_per_task

    if info_file_name in listdir(path_to_data):
        with open(path_to_data+'/'+info_file_name, 'rb') as fobj:
            task_data_info = pickle.load(fobj)
        make_new_info_file = False
    else:
        task_data_info, make_new_info_file = {'train': {}, 'validation': {}}, True

    train_data, validation_data, test_data = [], [], []
    for class_cnt in range(num_classes//classes_per_task):
        class_cnt0 = class_cnt*classes_per_task
        class_cnt1 = class_cnt0 + classes_per_task - 1
        x_l = np.load(path_to_data+'/'+str(class_cnt)+'/X_l.npy')
        x_u = np.load(path_to_data+'/'+str(class_cnt)+'/X_u.npy')
        x_test = np.load(path_to_data+'/'+str(class_cnt)+'/X_test.npy')

        y_l = np.load(path_to_data + '/' + str(class_cnt) + '/y_l.npy')
        y_u = np.load(path_to_data + '/' + str(class_cnt) + '/y_u.npy')
        y_u_prime = np.load(path_to_data + '/' + str(class_cnt) + '/y_u_prime.npy')
        y_test = np.load(path_to_data + '/' + str(class_cnt) + '/y_test.npy')

        logit_u_prime = np.load(path_to_data + '/' + str(class_cnt) + '/logit_u_prime.npy')

        class_indices = [0]
        instances_per_class = []
        for i in range(1, classes_per_task):
            c_i = index_of_class(y_u, i)
            instances_per_class.append(c_i - class_indices[-1])
            class_indices.append(c_i)
        instances_per_class.append(len(y_u) - class_indices[-1])

        # shuffle data per class to sample unlabeled instances
        shuffled_indices_per_class = []
        for i in range(classes_per_task):
            if i == classes_per_task -1:
                end_i = len(x_u)
            else:
                end_i = class_indices[i+1]
            inds = np.array(range(class_indices[i], end_i))
            np.random.shuffle(inds)
            shuffled_indices_per_class.append(inds)

        indices_train_unlabeled = []
        indices_valid_unlabeled = []

        for c in range(classes_per_task):
            indices_train_unlabeled.extend(shuffled_indices_per_class[c][0:num_unlabeled_per_class])
            indices_valid_unlabeled.extend(shuffled_indices_per_class[c][num_unlabeled_per_class:num_unlabeled_per_class+num_valid_per_class])

        # include labeled data set in each instance incremental batch for CNNL experiments
        train_x, valid_x = np.transpose(np.concatenate((x_l, x_u[indices_train_unlabeled]), axis=0), (0, 2, 3, 1)), np.transpose(x_u[indices_valid_unlabeled], (0, 2, 3, 1))

        if use_true_label:
            train_y, valid_y = np.concatenate((y_l, y_u[indices_train_unlabeled]), axis=0), y_u[indices_valid_unlabeled]
        else:
            train_y, valid_y = np.concatenate((y_l, y_u_prime[indices_train_unlabeled]), axis=0), y_u_prime[indices_valid_unlabeled]
        test_x = np.transpose(x_test, (0, 2, 3, 1))
        if flatten_img:
            num_train, num_valid, num_test = train_x.shape[0], valid_x.shape[0], test_x.shape[0]
            train_data.append( (train_x.reshape([num_train, -1]), train_y) )
            validation_data.append( (valid_x.reshape([num_valid, -1]), valid_y) )
            test_data.append( (test_x.reshape([num_test, -1]), y_test) )
        else:
            train_data.append( (train_x, train_y) )
            validation_data.append( (valid_x, valid_y) )
            test_data.append( (test_x, y_test) )
    return train_data, validation_data, test_data

def data_loader_multiclass_instance_incremental_rep(path_to_data, num_classes, info_file_name, num_repetition, num_unlabel_data, num_validation_data=-1, flatten_img=True, use_true_label=False, dataset='mnist'):
    train_data_rep, valid_data_rep = [], []
    for rep_cnt in range(num_repetition):
        data_file_name = info_file_name+'_rep'+str(rep_cnt)+'.pkl' if not use_true_label else info_file_name+'_T_rep'+str(rep_cnt)+'.pkl'
        train_data, valid_data, test_data = data_loader_multiclass_instance_incremental(path_to_data, num_classes, data_file_name, num_unlabel_data, num_validation_data=num_validation_data, flatten_img=flatten_img, use_true_label=use_true_label, dataset=dataset)
        train_data_rep.append(train_data)
        valid_data_rep.append(valid_data)
        print('Complete loading data for repetition %d'%(rep_cnt))
    return train_data_rep, valid_data_rep, test_data

def data_loader_multiclass_instance_incremental(path_to_data, num_classes, info_file_name, num_unlabel_data, num_validation_data=-1, flatten_img=True, use_true_label=False, dataset='mnist'):

    def index_of_class(label_data, c_ind):
        return_index = -1
        for i in range(1, label_data.shape[0]):
            if label_data[i-1] == c_ind-1 and label_data[i] == c_ind:
                return_index = i
                break
        if return_index > 0:
            return return_index
        else:
            raise ValueError

    def get_randomized_indices(num_total, num_selection, num_selection2):
        list_of_indices = np.arange(num_total)
        np.random.shuffle(list_of_indices)
        return list_of_indices[:num_selection], list_of_indices[num_selection:num_selection+num_selection2]

    # CNNL experiments don't use validation data *shrug*
    # if num_validation_data < 0:
    #     num_validation_data = num_unlabel_data//2

    if info_file_name in listdir(path_to_data):
        with open(path_to_data+'/'+info_file_name, 'rb') as fobj:
            task_data_info = pickle.load(fobj)
        make_new_info_file = False
    else:
        task_data_info, make_new_info_file = {'train': {}, 'validation': {}}, True

    if 'mnist_mako' in dataset:
        dataset_name = '5-9'
        incr_batch_size = 1000
    elif 'cifar10_mako' in dataset:
        dataset_name = '0-9'
        incr_batch_size = 1000

    train_data, validation_data, test_data = [], [], []

    # ***** Load all data, split up into "tasks" (i.e. incremental instance batches) later *****
    x_l = np.load(path_to_data+'/'+dataset_name+'/X_l.npy')
    x_u = np.load(path_to_data+'/'+dataset_name+'/X_u.npy')
    x_test = np.load(path_to_data+'/'+dataset_name+'/X_test.npy')

    y_l = np.load(path_to_data + '/' + dataset_name + '/y_l.npy')
    y_u = np.load(path_to_data + '/' + dataset_name + '/y_u.npy')
    y_u_prime = np.load(path_to_data + '/' + dataset_name + '/y_u_prime.npy')
    logit_u_prime = np.load(path_to_data + '/' + dataset_name + '/logit_u_prime.npy')
    y_test = np.load(path_to_data + '/' + dataset_name + '/y_test.npy')

    # ***** Filter data to remove low-confidence Mako-labeled instances *****
    print('Unlabled data before filtering:', len(x_u))
    beta = 0.01
    conf_threshold = 1/num_classes + beta
    for i in range(len(x_u)-1, -1, -1):
        if max(logit_u_prime[i]) < conf_threshold:
            x_u = np.delete(x_u, i)
            y_u = np.delete(y_u, i)
            y_u_prime = np.delete(y_u_prime, i)
            logit_u_prime = np.delete(logit_u_prime, i)
    print('Unlabeled data after filtering for confidence:', len(x_u))

    class_indices = [0]
    instances_per_class = []
    for i in range(1, num_classes):
        c_i = index_of_class(y_u, i)
        instances_per_class.append(c_i - class_indices[-1])
        class_indices.append(c_i)
    instances_per_class.append(len(y_u) - class_indices[-1])

    # determine how many batches we can make while maintaining class balance
    min_class_size = min(instances_per_class)
    class_instances_per_batch = incr_batch_size // num_classes
    num_batches = min_class_size // class_instances_per_batch

    print('Creating', num_batches, 'instance incremental batches with', class_instances_per_batch, 'instances per class per batch (min class size:', min_class_size,'instances)')

    # shuffle data per class to sample instance incremental instances without replacement
    shuffled_indices_per_class = []
    for i in range(num_classes):
        if i == num_classes -1:
            end_i = len(x_u)
        else:
            end_i = class_indices[i+1]
        inds = np.array(range(class_indices[i], end_i))
        np.random.shuffle(inds)
        shuffled_indices_per_class.append(inds)

    for batch in range(num_batches + 1):
        indices_train_unlabeled = []

        if batch == 0:
            indices_train_unlabeled = []  # set initial batch as labeled data set only, i.e. no unlabeled data
        else:
            for c in range(num_classes):
                indices_train_unlabeled.extend(shuffled_indices_per_class[c][batch*class_instances_per_batch:batch*class_instances_per_batch+class_instances_per_batch])
        # include labeled data set in each instance incremental batch for CNNL experiments
        train_x = np.transpose(np.concatenate((x_l, x_u[indices_train_unlabeled]), axis=0), (0, 2, 3, 1))
        valid_x = copy.deepcopy(train_x)  # no validation data for CNNL experiments, just copy train data so we can use the same training routines
        if use_true_label:
            train_y = np.concatenate((y_l, y_u[indices_train_unlabeled]), axis=0)
            valid_y = copy.deepcopy(train_y)  # no validation data for CNNL experiments, just copy train data so we can use the same training routines
        else:
            train_y = np.concatenate((y_l, y_u_prime[indices_train_unlabeled]), axis=0)
            valid_y = copy.deepcopy(train_y)  # no validation data for CNNL experiments, just copy train data so we can use the same training routines
        test_x = np.transpose(x_test, (0, 2, 3, 1))  # repeat the same test set for each "task" (i.e. instance-incremental batch)
        if flatten_img:
            num_train, num_valid, num_test = train_x.shape[0], valid_x.shape[0], test_x.shape[0]
            train_data.append( (train_x.reshape([num_train, -1]), train_y) )
            validation_data.append( (valid_x.reshape([num_valid, -1]), valid_y) )
            test_data.append( (test_x.reshape([num_test, -1]), y_test) )
        else:
            train_data.append( (train_x, train_y) )
            validation_data.append( (valid_x, valid_y) )
            test_data.append( (test_x, y_test) )
    return train_data, validation_data, test_data

#### function to print information of data file (number of parameters, dimension, etc.)
def mnist_data_print_info(train_data, valid_data, test_data, no_group=False, print_info=True):
    if no_group:
        num_task = len(train_data)

        num_train, num_valid, num_test = [train_data[x][0].shape[0] for x in range(num_task)], [valid_data[x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0].shape[1], 0
        y_depth = [int(np.amax(x[1])+1) for x in train_data]
        if print_info:
            print("Tasks : %d\nTrain data : %d, Validation data : %d, Test data : %d" %(num_task, num_train, num_valid, num_test))
            print("Input dim : %d, Label dim : %d" %(x_dim, y_dim))
            print("Maximum label : ", y_depth, "\n")
        return (num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth)
    else:
        assert (len(train_data) == len(valid_data)), "Different number of groups in train/validation data"
        num_group = len(train_data)

        bool_num_task = [(len(train_data[0]) == len(train_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in train data"
        bool_num_task = [(len(valid_data[0]) == len(valid_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in validation data"
        assert (len(train_data[0])==len(valid_data[0]) and len(train_data[0])==len(test_data)), "Different number of tasks in train/validation/test data"
        num_task = len(train_data[0])

        num_train, num_valid, num_test = [train_data[0][x][0].shape[0] for x in range(num_task)], [valid_data[0][x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0][0].shape[1], 0
        y_depth = [int(np.amax(x[1])+1) for x in train_data[0]]
        if print_info:
            print("Tasks : %d, Groups of training/valid : %d\n" %(num_task, num_group))
            print("Train data : ", num_train, ", Validation data : ", num_valid, ", Test data : ", num_test)
            print("Input dim : %d, Label dim : %d" %(x_dim, y_dim))
            print("Maximum label : ", y_depth, "\n")
        return (num_task, num_group, num_train, num_valid, num_test, x_dim, y_dim, y_depth)


#### function to print information of data file (number of parameters, dimension, etc.)
def cifar_data_print_info(train_data, valid_data, test_data, no_group=False, print_info=True, grouped_test=False):
    if no_group:
        num_task = len(train_data)

        num_train, num_valid, num_test = [train_data[x][0].shape[0] for x in range(num_task)], [valid_data[x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0].shape[1], 0
        y_depth = [int(np.amax(x[1])+1) for x in train_data]
        if print_info:
            print("Tasks : %d\nTrain data : %d, Validation data : %d, Test data : %d" %(num_task, num_train, num_valid, num_test))
            print("Input dim : %d, Label dim : %d" %(x_dim, y_dim))
            print("Maximum label : ", y_depth, "\n")
        return (num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth)
    else:
        if grouped_test:
            assert (len(train_data) == len(valid_data) and len(valid_data) == len(test_data)), "Different number of groups in train/validation data"
            assert (len(train_data[0])==len(valid_data[0]) and len(train_data[0])==len(test_data[0])), "Different number of tasks in train/validation/test data"
        else:
            assert (len(train_data) == len(valid_data)), "Different number of groups in train/validation data"
            assert (len(train_data[0])==len(valid_data[0]) and len(train_data[0])==len(test_data)), "Different number of tasks in train/validation/test data"
        num_group = len(train_data)

        bool_num_task = [(len(train_data[0]) == len(train_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in train data"
        bool_num_task = [(len(valid_data[0]) == len(valid_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in validation data"
        if grouped_test:
            bool_num_task = [(len(test_data[0]) == len(test_data[x])) for x in range(1, num_group)]
            assert all(bool_num_task), "Different number of tasks in some of groups in test data"
        num_task = len(train_data[0])

        num_train, num_valid = [train_data[0][x][0].shape[0] for x in range(num_task)], [valid_data[0][x][0].shape[0] for x in range(num_task)]
        if grouped_test:
            num_test = [test_data[0][x][0].shape[0] for x in range(num_task)]
        else:
            num_test = [test_data[x][0].shape[0] for x in range(num_task)]

        x_dim, y_dim = train_data[0][0][0].shape[1], 0
        y_depth = [int(np.amax(x[1])+1) for x in train_data[0]]
        if print_info:
            print("Tasks : %d, Groups of training/valid : %d\n" %(num_task, num_group))
            print("Train data : ", num_train, ", Validation data : ", num_valid, ", Test data : ", num_test)
            print("Input dim : %d, Label dim : %d" %(x_dim, y_dim))
            print("Maximum label : ", y_depth, "\n")
        return (num_task, num_group, num_train, num_valid, num_test, x_dim, y_dim, y_depth)