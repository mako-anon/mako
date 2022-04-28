import tensorflow as tf
import numpy as np

from utils.utils import get_value_of_valid_tensors, savemat_wrapper, savemat_wrapper_nested_list, count_trainable_var2, get_list_of_valid_tensors
from utils.utils_nn import new_cnn_fc_net, new_hybrid_tensorfactored_cnn_fc_net
from classification.model.lifelong_model_frame import Lifelong_Model_Frame

_tf_ver = tf.__version__.split('.')
_up_to_date_tf = int(_tf_ver[0]) > 1 or (int(_tf_ver[0])==1 and int(_tf_ver[1]) >= 14)
if _up_to_date_tf:
    _tf_tensor = tf.is_tensor
else:
    _tf_tensor = tf.contrib.framework.is_tensor


###############################################################
#### Single task learner (CNN + FC) for Multi-task setting ####
###############################################################
#### Convolutional & Fully-connected Neural Net
class LL_several_CNN_minibatch(Lifelong_Model_Frame):
    def __init__(self, model_hyperpara, train_hyperpara):
        super().__init__(model_hyperpara, train_hyperpara)

    def _build_task_model(self, net_input, output_size, params=None, trainable=False):
        if params is not None:
            assert (len(params) == 2*(self.num_conv_layers+self.num_fc_layers)), "Given trained parameters of network doesn't match to the hyper-parameters!"
        else:
            params = [None for _ in range(2*(self.num_conv_layers+self.num_fc_layers))]
        task_net, conv_params, fc_params = new_cnn_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], cnn_params=params[0:2*self.num_conv_layers], cnn_activation_fn=self.hidden_act, fc_params=params[2*self.num_conv_layers:2*(self.num_conv_layers+self.num_fc_layers)], fc_activation_fn=self.hidden_act, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, padding_type=self.padding_type, input_size=self.input_size[0:2], trainable=trainable, skip_connections=list(self.skip_connect))
        return task_net, conv_params, fc_params

    def _build_whole_model(self):
        for task_cnt, (num_classes, x_b) in enumerate(zip(self.output_sizes, self.x_batch)):
            task_net, conv_params, fc_params = self._build_task_model(x_b, num_classes, params=None if (task_cnt==self.current_task) and (self.task_is_new) else self.np_params[task_cnt], trainable=(task_cnt==self.current_task))
            self.task_models.append(task_net)
            self.conv_params.append(conv_params)
            self.fc_params.append(fc_params)
            self.params.append(conv_params+fc_params)
            self.num_trainable_var += count_trainable_var2(self.params[-1])

    def get_params_val(self, sess):
        cnn_param_val = [get_value_of_valid_tensors(sess, cnn_param) for cnn_param in self.conv_params]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_params]

        parameters_val = {}
        parameters_val['conv_trainable_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val


########################################################
####     Single CNN + FC for Multi-task Learning    ####
########################################################
#### Convolutional & Fully-connected Neural Net
class LL_single_CNN_minibatch(Lifelong_Model_Frame):
    def __init__(self, model_hyperpara, train_hyperpara):
        super().__init__(model_hyperpara, train_hyperpara)

    def _build_task_model(self, net_input, output_size, params=None, trainable=False):
        if params is not None:
            assert (len(params) == 2*(self.num_conv_layers+self.num_fc_layers)), "Given trained parameters of network doesn't match to the hyper-parameters!"
        else:
            params = [None for _ in range(2*(self.num_conv_layers+self.num_fc_layers))]
        task_net, conv_params, fc_params = new_cnn_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], cnn_params=params[0:2*self.num_conv_layers], cnn_activation_fn=self.hidden_act, fc_params=params[2*self.num_conv_layers:2*(self.num_conv_layers+self.num_fc_layers)], fc_activation_fn=self.hidden_act, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, padding_type=self.padding_type, input_size=self.input_size[0:2], trainable=trainable)
        return task_net, conv_params, fc_params

    def _build_whole_model(self):
        for task_cnt, (num_classes, x_b) in enumerate(zip(self.output_sizes, self.x_batch)):
            if self.num_tasks == 1:
                task_net, conv_params, fc_params = self._build_task_model(x_b, num_classes, trainable=True)
            elif task_cnt < 1:
                task_net, conv_params, fc_params = self._build_task_model(x_b, num_classes, self.np_params[task_cnt], trainable=True)
            else:
                task_net, _, _ = self._build_task_model(x_b, num_classes, conv_params+fc_params, trainable=True)
            self.task_models.append(task_net)
            self.conv_params.append(conv_params)
            self.fc_params.append(fc_params)
            self.params.append(conv_params+fc_params)
        self.num_trainable_var = count_trainable_var2(self.params[0])

    def add_new_task(self, output_dim, curr_task_index, single_input_placeholder=False):
        if len(self.output_sizes)> 0:
            assert (output_dim == self.output_sizes[0]), "The output layer of tasks models have different numbers of units!"
        super().add_new_task(output_dim, curr_task_index, single_input_placeholder=single_input_placeholder)

    def get_params_val(self, sess):
        cnn_param_val = [get_value_of_valid_tensors(sess, cnn_param) for cnn_param in self.conv_params]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_params]

        parameters_val = {}
        parameters_val['conv_trainable_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val


########################################################
####      CNN + FC model for Lifelong Learning      ####
########          Tensor Factorization          ########
########################################################
class LL_CNN_tensorfactor_minibatch(Lifelong_Model_Frame):
    def __init__(self, model_hyperpara, train_hyperpara):
        super().__init__(model_hyperpara, train_hyperpara)
        self.conv_sharing = model_hyperpara['conv_sharing']
        self.aux_loss_weight = model_hyperpara['auxiliary_loss_weight']

    def _build_task_model(self, net_input, output_size, task_cnt, params=None, trainable=False, trainable_shared=False):
        if params is None:
            params_KB, params_TS, params_conv, params_fc = None, None, None, None
        else:
            params_KB, params_TS, params_conv, params_fc = params['KB'], params['TS'], params['Conv'], params['FC']

        if params_KB is not None:
            assert (len(params_KB) == self.num_conv_layers), "Given trained parameters of DF KB doesn't match to the hyper-parameters!"
        if params_TS is not None:
            assert (len(params_TS) == 5*self.num_conv_layers), "Given trained parameters of DF TS doesn't match to the hyper-parameters!"
        if params_conv is not None:
            assert (len(params_conv) == 2*self.num_conv_layers), "Given trained parameters of conv doesn't match to the hyper-parameters!"
        else:
            params_conv = [None for _ in range(2*self.num_conv_layers)]
        if params_fc is not None:
            assert (len(params_fc) == 2*self.num_fc_layers), "Given trained parameters of fc doesn't match to the hyper-parameters!"
        else:
            params_fc = [None for _ in range(2*self.num_fc_layers)]

        task_net, shared_conv_KB_param_tmp, conv_TS_param_tmp, gen_conv_param_tmp, conv_params, _, fc_params = new_hybrid_tensorfactored_cnn_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], self.conv_sharing, cnn_activation_fn=self.hidden_act, cnn_KB_params=params_KB, cnn_TS_params=params_TS, cnn_params=params_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable)
        if self.shared_conv_KB_params is None:
            self.shared_conv_KB_params = shared_conv_KB_param_tmp
        self.conv_TS_params.append(conv_TS_param_tmp)
        self.gen_conv_params.append(gen_conv_param_tmp)
        return task_net, conv_params, fc_params

    def _build_whole_model(self):
        for task_cnt, (num_classes, x_b) in enumerate(zip(self.output_sizes, self.x_batch)):
            if (task_cnt==self.current_task) and (self.task_is_new):
                param_to_reuse = {'KB': self.shared_conv_KB_params, 'TS': None, 'Conv': None, 'FC': None}
            else:
                param_to_reuse = {'KB': self.shared_conv_KB_params if self.shared_conv_KB_params else self.np_params[0]['KB'], 'TS': self.np_params[task_cnt]['TS'], 'Conv': self.np_params[task_cnt]['Conv'], 'FC': self.np_params[task_cnt]['FC']}
            task_net, conv_params, fc_params = self._build_task_model(x_b, num_classes, task_cnt, params=param_to_reuse, trainable=(task_cnt==self.current_task))

            if task_cnt == 0:
                self.dfcnn_KB_params_size = count_trainable_var2(self.shared_conv_KB_params)

            self.task_models.append(task_net)
            self.conv_params.append(conv_params)
            self.fc_params.append(fc_params)
            self.params.append(self._collect_trainable_variables())
            self.num_trainable_var += count_trainable_var2(self.params[-1]) if task_cnt < 1 else count_trainable_var2(self.params[-1]) - self.dfcnn_KB_params_size

        self.shared_conv_KB_trainable_param = get_list_of_valid_tensors(self.shared_conv_KB_params)
        self.conv_TS_trainable_param = get_list_of_valid_tensors(self.conv_TS_params[self.current_task])
        self.conv_trainable_param = get_list_of_valid_tensors(self.conv_params[self.current_task])
        self.fc_trainable_param = get_list_of_valid_tensors(self.fc_params[self.current_task])

    def add_new_task(self, output_dim, curr_task_index, single_input_placeholder=False):
        self.shared_conv_KB_params, self.conv_TS_params, self.gen_conv_params = None, [], []
        super().add_new_task(output_dim, curr_task_index, single_input_placeholder=single_input_placeholder)

    def define_opt(self):
        with tf.name_scope('Optimization'):
            aux_loss = 0.0
            for ts_param_cnt, (ts_param) in enumerate(self.conv_TS_trainable_param):
                if ts_param_cnt%5 < 4:
                    ts_dim = int(ts_param.get_shape()[0])
                    tensor_for_norm = tf.matmul(tf.transpose(ts_param), ts_param) - tf.eye(ts_dim)
                    aux_loss += tf.norm(tensor_for_norm, ord='fro', axis=(0, 1))

            KB_grads = tf.gradients(self.loss[self.current_task], self.shared_conv_KB_trainable_param)
            KB_grads_vars = [(grad, param) for grad, param in zip(KB_grads, self.shared_conv_KB_trainable_param)]

            TS_grads = tf.gradients(self.loss[self.current_task] + self.aux_loss_weight*aux_loss, self.conv_TS_trainable_param)
            TS_grads_vars = [(grad, param) for grad, param in zip(TS_grads, self.conv_TS_trainable_param)]

            conv_grads = tf.gradients(self.loss[self.current_task], self.conv_trainable_param)
            conv_grads_vars = [(grad, param) for grad, param in zip(conv_grads, self.conv_trainable_param)]

            fc_grads = tf.gradients(self.loss[self.current_task], self.fc_trainable_param)
            fc_grads_vars = [(grad, param) for grad, param in zip(fc_grads, self.fc_trainable_param)]

            self.grads = list(KB_grads) + list(TS_grads) + list(conv_grads) + list(fc_grads)
            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
            grads_vars = KB_grads_vars + TS_grads_vars + conv_grads_vars + fc_grads_vars
            self.update = trainer.apply_gradients(grads_vars)

    def get_params_val(self, sess):
        KB_param_val = get_value_of_valid_tensors(sess, self.shared_conv_KB_params)
        TS_param_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.conv_TS_params]
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.gen_conv_params]
        cnn_param_val = [get_value_of_valid_tensors(sess, cnn_param) for cnn_param in self.conv_params]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_params]

        parameters_val = {}
        parameters_val['conv_KB'] = savemat_wrapper(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['conv_trainable_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val

    def convert_tfVar_to_npVar(self, sess):
        def list_param_converter(list_of_params):
            converted_params = []
            for p in list_of_params:
                if type(p) == np.ndarray:
                    converted_params.append(p)
                elif _tf_tensor(p):
                    converted_params.append(sess.run(p))
                else:
                    converted_params.append(p)  ## append 'None' param
            return converted_params

        def double_list_param_converter(list_of_params):
            converted_params = []
            for task_params in list_of_params:
                converted_params.append(list_param_converter(task_params))
            return converted_params

        self.np_params = []
        np_KB = list_param_converter(self.shared_conv_KB_params)
        np_TS = double_list_param_converter(self.conv_TS_params)
        np_conv = double_list_param_converter(self.conv_params)
        np_fc = double_list_param_converter(self.fc_params)
        for t, c, f in zip(np_TS, np_conv, np_fc):
            self.np_params.append({'KB': np_KB, 'TS': t, 'Conv': c, 'FC': f} if len(self.np_params)< 1 else {'TS': t, 'Conv': c, 'FC': f})

    def _collect_trainable_variables(self):
        return_list = []
        for p in self.shared_conv_KB_params:
            if p is not None:
                return_list.append(p)
        for p in self.conv_TS_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.conv_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.fc_params[-1]:
            if p is not None:
                return_list.append(p)
        return return_list