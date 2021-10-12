import os
import timeit
from time import sleep

import numpy as np
import tensorflow as tf
from scipy.io import savemat

from lml.utils.utils_data import mnist_data_print_info, cifar_data_print_info

from lml.classification.model.cnn_baseline_model import LL_several_CNN_minibatch, LL_single_CNN_minibatch, LL_CNN_tensorfactor_minibatch
from lml.classification.model.cnn_den_model import CNN_FC_DEN
from lml.classification.model.cnn_dfcnn_model import LL_hybrid_DFCNN_minibatch

_tf_ver = tf.__version__.split('.')
_up_to_date_tf = int(_tf_ver[0]) > 1 or (int(_tf_ver[0])==1 and int(_tf_ver[1]) > 14)

_debug_mode = False

#### function to generate appropriate deep neural network
def model_generation(model_architecture, model_hyperpara, train_hyperpara, data_info):
    learning_model, gen_model_success = None, True

    ###### CNN models
    if model_architecture == 'stl_cnn':
        print("Training STL-CNNs model (collection of NN per a task) - Lifelong Learning")
        learning_model = LL_several_CNN_minibatch(model_hyperpara, train_hyperpara)
    elif model_architecture == 'singlenn_cnn':
        print("Training a single CNN model for all tasks - Lifelong Learning")
        learning_model =  LL_single_CNN_minibatch(model_hyperpara, train_hyperpara)
    elif model_architecture == 'mtl_tf_cnn' or model_architecture == 'hybrid_tf_cnn':
        print("Training LL-CNN model (Tensorfactorization ver.)")
        learning_model = LL_CNN_tensorfactor_minibatch(model_hyperpara, train_hyperpara)
        print("\tConfig of sharing: ", model_hyperpara['conv_sharing'])
    elif model_architecture == 'den_cnn':
        print("Training LL-CNN Dynamically Expandable model")
        learning_model = CNN_FC_DEN(model_hyperpara, train_hyperpara, data_info)
    elif model_architecture == 'hybrid_dfcnn':
        cnn_sharing = model_hyperpara['conv_sharing']
        print("Training Hybrid DF-CNNs model - Lifelong Learning")
        learning_model = LL_hybrid_DFCNN_minibatch(model_hyperpara, train_hyperpara)
        print("\tConfig of sharing: ", cnn_sharing)
    else:
        print("No such model exists!!")
        gen_model_success = False

    if learning_model is not None:
        learning_model.model_architecture = model_architecture
    sleep(5)
    return (learning_model, gen_model_success)


#### module of training/testing one model
def train_lifelong(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, classification_prob, useGPU=False, GPU_device=0, save_param=False, param_folder_path='saved_param', save_param_interval=100, save_graph=False, tfInitParam=None, run_cnt=0, task_order=None):
    print("Training function for lifelong learning!")
    assert ('den' not in model_architecture and 'dynamically' not in model_architecture), "Use train function appropriate to the architecture"

    config = tf.ConfigProto()
    if useGPU:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_device)
        if _up_to_date_tf:
            ## TF version >= 1.14
            gpu = tf.config.experimental.list_physical_devices('GPU')[0]
            tf.config.experimental.set_memory_growth(gpu, True)
        else:
            ## TF version < 1.14
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
        print("GPU %d is used" %(GPU_device))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        print("CPU is used")

    if task_order is None:
        task_training_order = list(range(train_hyperpara['num_tasks']))
    else:
        task_training_order = list(task_order)
    print("\tTask order : ", task_training_order)
    task_change_epoch = [1]

    ### set-up data
    train_data, validation_data, test_data = dataset
    if 'mnist' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = mnist_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif ('cifar10' in data_type) and not ('cifar100' in data_type):
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = cifar_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif 'cifar100' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = cifar_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    else:
        raise ValueError("No specified dataset!")

    ### Set hyperparameter related to training process
    learning_step_max = train_hyperpara['learning_step_max']
    improvement_threshold = train_hyperpara['improvement_threshold']
    patience = train_hyperpara['patience']
    patience_multiplier = train_hyperpara['patience_multiplier']
    if 'batch_size' in model_hyperpara:
        batch_size = model_hyperpara['batch_size']

    ### Generate Model
    learning_model, generation_success = model_generation(model_architecture, model_hyperpara, train_hyperpara, [x_dim, y_dim, y_depth, num_task])
    if not generation_success:
        return (None, None)

    ### Training Procedure
    learning_step = -1
    if num_task > 1:
        indices = [list(range(num_train[x])) for x in range(num_task)]
    else:
        indices = [list(range(num_train[0]))]

    best_valid_error, test_error_at_best_epoch, best_epoch, epoch_bias = np.inf, np.inf, -1, 0
    train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist = [], [], [], []

    start_time = timeit.default_timer()
    for train_task_cnt, (task_for_train) in enumerate(task_training_order):
        tf.reset_default_graph()

        with tf.Session(config=config) as sess:
            print("\n%d-th Task - %d"%(train_task_cnt, task_for_train))
            learning_model.add_new_task(y_depth[task_for_train], task_for_train, single_input_placeholder=True)
            num_learned_tasks = learning_model.number_of_learned_tasks()

            sess.run(tf.global_variables_initializer())
            if save_graph:
                tfboard_writer = tf.summary.FileWriter('./graphs/%s/run%d/task%d'%(model_architecture, run_cnt, train_task_cnt), sess.graph)

            if save_param and _debug_mode:
                para_file_name = param_folder_path + '/init_model_parameter_taskC%d.mat'%(train_task_cnt)
                curr_param = learning_model.get_params_val(sess)
                savemat(para_file_name, {'parameter': curr_param})

            while learning_step < min(learning_step_max, epoch_bias + patience):
                learning_step = learning_step+1

                #### training & performance measuring process
                if learning_step > 0:
                    learning_model.train_one_epoch(sess, train_data[task_for_train][0], train_data[task_for_train][1], learning_step-1, task_for_train, indices[task_for_train], dropout_prob=0.5)

                train_error_tmp = [0.0 for _ in range(num_task)]
                validation_error_tmp = [0.0 for _ in range(num_task)]
                test_error_tmp = [0.0 for _ in range(num_task)]
                for tmp_cnt, (task_index_to_eval) in enumerate(task_training_order[:train_task_cnt+1]):
                    if task_index_to_eval in task_training_order[:tmp_cnt]:
                        continue
                    train_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, train_data[task_index_to_eval][0], train_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)
                    validation_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, validation_data[task_index_to_eval][0], validation_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)
                    test_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, test_data[task_index_to_eval][0], test_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)

                if classification_prob:
                    ## for classification, error_tmp is actually ACCURACY, thus, change the sign for checking improvement
                    train_error, valid_error, test_error = -(sum(train_error_tmp)/(num_learned_tasks)), -(sum(validation_error_tmp)/(num_learned_tasks)), -(sum(test_error_tmp)/(num_learned_tasks))
                    train_error_to_compare, valid_error_to_compare, test_error_to_compare = -train_error_tmp[task_for_train], -validation_error_tmp[task_for_train], -test_error_tmp[task_for_train]
                else:
                    train_error, valid_error, test_error = np.sqrt(np.array(train_error_tmp)/(num_learned_tasks)), np.sqrt(np.array(validation_error_tmp)/(num_learned_tasks)), np.sqrt(np.array(test_error_tmp)/(num_learned_tasks))
                    train_error_tmp, validation_error_tmp, test_error_tmp = list(np.sqrt(np.array(train_error_tmp))), list(np.sqrt(np.array(validation_error_tmp))), list(np.sqrt(np.array(test_error_tmp)))
                    train_error_to_compare, valid_error_to_compare, test_error_to_compare = train_error_tmp[task_for_train], validation_error_tmp[task_for_train], test_error_tmp[task_for_train]

                #### error related process
                print('epoch %d - Train : %f, Validation : %f' % (learning_step, abs(train_error_to_compare), abs(valid_error_to_compare)))

                if valid_error_to_compare < best_valid_error:
                    str_temp = ''
                    if valid_error_to_compare < best_valid_error * improvement_threshold:
                        patience = max(patience, (learning_step-epoch_bias)*patience_multiplier)
                        str_temp = '\t<<'
                    best_valid_error, best_epoch = valid_error_to_compare, learning_step
                    test_error_at_best_epoch = test_error_to_compare
                    print('\t\t\t\t\t\t\tTest : %f%s' % (abs(test_error_at_best_epoch), str_temp))

                train_error_hist.append(train_error_tmp + [abs(train_error)])
                valid_error_hist.append(validation_error_tmp + [abs(valid_error)])
                test_error_hist.append(test_error_tmp + [abs(test_error)])
                best_test_error_hist.append(abs(test_error_at_best_epoch))

                if learning_step >= epoch_bias+min(patience, learning_step_max//len(task_training_order)):
                    if save_param:
                        para_file_name = param_folder_path + '/model_parameter_taskC%d_task%d.mat'%(train_task_cnt, task_for_train)
                        curr_param = learning_model.get_params_val(sess)
                        savemat(para_file_name, {'parameter': curr_param})

                    if train_task_cnt == len(task_training_order)-1:
                        if save_param:
                            para_file_name = param_folder_path + '/final_model_parameter.mat'
                            curr_param = learning_model.get_params_val(sess)
                            savemat(para_file_name, {'parameter': curr_param})
                    else:
                        # update epoch_bias, task_for_train, task_change_epoch
                        epoch_bias = learning_step
                        task_change_epoch.append(learning_step+1)

                        # initialize best_valid_error, best_epoch, patience
                        patience = train_hyperpara['patience']
                        best_valid_error, best_epoch = np.inf, -1

                        learning_model.convert_tfVar_to_npVar(sess)
                        print('\n\t>>Change to new task!<<\n')
                    break

    end_time = timeit.default_timer()
    print("End of Training")
    print("Time consumption for training : %.2f" %(end_time-start_time))

    result_summary = {}
    result_summary['training_time'] = end_time - start_time
    result_summary['num_epoch'] = learning_step
    result_summary['history_train_error'] = train_error_hist
    result_summary['history_validation_error'] = valid_error_hist
    result_summary['history_test_error'] = test_error_hist
    result_summary['history_best_test_error'] = best_test_error_hist

    tmp_valid_error_hist = np.array(valid_error_hist)
    chk_epoch = [(task_change_epoch[x], task_change_epoch[x+1]) for x in range(len(task_change_epoch)-1)] + [(task_change_epoch[-1], learning_step+1)]
    result_summary['task_changed_epoch'] = task_change_epoch

    if save_graph:
        tfboard_writer.close()
    return result_summary, learning_model.num_trainable_var



#### module of training/testing one model
def train_den_net(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, classification_prob, doLifelong, useGPU=False, GPU_device=0, save_param=False, param_folder_path='saved_param', save_param_interval=100, save_graph=False, task_order=None):
    assert (('den' in model_architecture or 'dynamically' in model_architecture) and classification_prob and doLifelong), "Use train function appropriate to the architecture (Dynamically Expandable Net)"
    print("\nTrain function for Dynamically Expandable Net is called!\n")

    config = tf.ConfigProto()
    if useGPU:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_device)
        if _up_to_date_tf:
            ## TF version >= 1.14
            gpu = tf.config.experimental.list_physical_devices('GPU')[0]
            tf.config.experimental.set_memory_growth(gpu, True)
        else:
            ## TF version < 1.14
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
        print("GPU %d is used" %(GPU_device))
    else:
        print("CPU is used")

    if task_order is None:
        task_training_order = list(range(train_hyperpara['num_tasks']))
    else:
        task_training_order = list(task_order)

    ### set-up data
    train_data, validation_data, test_data = dataset
    if 'mnist' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = mnist_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif 'cifar10' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = cifar_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif data_type == 'cifar100':
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = cifar_data_print_info(train_data, validation_data, test_data, True, print_info=False)

    ### reformat data for DEN
    trainX, trainY = [train_data[t][0] for t in range(num_task)], [train_data[t][1] for t in range(num_task)]
    validX, validY = [validation_data[t][0] for t in range(num_task)], [validation_data[t][1] for t in range(num_task)]
    testX, testY = [test_data[t][0] for t in range(num_task)], [test_data[t][1] for t in range(num_task)]


    if save_graph:
        if 'graphs' not in os.listdir(os.getcwd()):
            os.mkdir(os.getcwd()+'/graphs')


    ### Set hyperparameter related to training process
    learning_step_max = train_hyperpara['learning_step_max']
    improvement_threshold = train_hyperpara['improvement_threshold']
    patience = train_hyperpara['patience']
    patience_multiplier = train_hyperpara['patience_multiplier']
    if 'batch_size' in model_hyperpara:
        batch_size = model_hyperpara['batch_size']

    ### Generate Model
    learning_model, generation_success = model_generation(model_architecture, model_hyperpara, train_hyperpara, [x_dim, y_dim, y_depth, num_task])
    if not generation_success:
        return (None, None)

    learning_model.set_sess_config(config)

    params = dict()
    train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy = [], [], [], []
    start_time = timeit.default_timer()
    for train_task_cnt, (task_for_train) in enumerate(task_training_order):
        print("\n\nStart training new %d-th task (%d)" %(train_task_cnt, task_for_train))
        data = (trainX, trainY, validX, validY, testX, testY)

        learning_model.sess = tf.Session(config=config)

        learning_model.T = learning_model.T + 1
        learning_model.task_indices.append(train_task_cnt+1)
        learning_model.load_params(params, time = 1)
        tr_acc, v_acc, te_acc, best_te_acc = learning_model.add_task(train_task_cnt+1, data, save_param, save_graph)
        train_accuracy = train_accuracy+tr_acc
        valid_accuracy = valid_accuracy+v_acc
        test_accuracy = test_accuracy+te_acc
        best_test_accuracy = best_test_accuracy+best_te_acc

        params = learning_model.get_params()

        learning_model.destroy_graph()
        learning_model.sess.close()

    num_trainable_var = learning_model.num_trainable_var(params_list=params)
    end_time = timeit.default_timer()
    print("End of Training")
    print("Time consumption for training : %.2f" %(end_time-start_time))

    task_change_epoch = learning_model.task_change_epoch

    tmp_valid_acc_hist = np.array(valid_accuracy)
    chk_epoch = [(task_change_epoch[x], task_change_epoch[x+1]) for x in range(len(task_change_epoch)-1)]
    tmp_best_valid_acc_list = [np.amax(tmp_valid_acc_hist[x[0]:x[1], t]) for x, t in zip(chk_epoch, range(num_task))]

    result_summary = {}
    result_summary['training_time'] = end_time - start_time
    result_summary['num_epoch'] = learning_model.num_training_epoch
    result_summary['history_train_error'] = np.array(train_accuracy)
    result_summary['history_validation_error'] = np.array(valid_accuracy)
    result_summary['history_test_error'] = np.array(test_accuracy)
    result_summary['history_best_test_error'] = np.array(best_test_accuracy)
    result_summary['best_validation_error'] = sum(tmp_best_valid_acc_list) / float(len(tmp_best_valid_acc_list))
    result_summary['test_error_at_best_epoch'] = 0.0
    result_summary['task_changed_epoch'] = task_change_epoch[:-1]
    return result_summary, num_trainable_var