import numpy as np
import tensorflow as tf

from lml.utils.utils import *

#### function to generate network of cnn->ffnn
def new_cnn_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_activation_fn=tf.nn.relu, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, skip_connections=[], trainable=True, use_numpy_var_in_graph=False):
    cnn_model, cnn_params, cnn_output_dim = new_cnn_net(net_input, k_sizes, ch_sizes, stride_sizes, activation_fn=cnn_activation_fn, params=cnn_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size, skip_connections=skip_connections, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)

    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
    return (cnn_model+fc_model, cnn_params, fc_params)

##############################################################################
################# Tensor Factored Model
##############################################################################
######## Lifelong Learning - based on Adrian Bulat, et al. Incremental Multi-domain Learning with Network Latent Tensor Factorization
def new_TF_KB_param(shape, layer_number, init_tensor=None, trainable=True):
    kb_name = 'KB_'+str(layer_number)
    if init_tensor is None:
        param_to_return = tf.get_variable(name=kb_name, shape=shape, dtype=tf.float32, trainable=trainable)
    elif type(init_tensor) == np.ndarray:
        param_to_return = tf.get_variable(name=kb_name, shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(init_tensor), trainable=trainable)
    else:
        param_to_return = init_tensor
    return param_to_return

def new_TF_TS_param(shape, layer_number, task_number, init_tensor, trainable):
    params_name = ['TF_Wch0_'+str(layer_number)+'_'+str(task_number), 'TF_Wch1_'+str(layer_number)+'_'+str(task_number), 'TF_Wch2_'+str(layer_number)+'_'+str(task_number), 'TF_Wch3_'+str(layer_number)+'_'+str(task_number), 'b_'+str(layer_number)+'_'+str(task_number)]
    params_to_return = []
    for i, (t, n) in enumerate(zip(init_tensor, params_name)):
        if t is None:
            params_to_return.append(tf.get_variable(name=n, shape=shape[i], dtype=tf.float32, trainable=trainable))
        elif type(t) == np.ndarray:
            params_to_return.append(tf.get_variable(name=n, shape=shape[i], dtype=tf.float32, trainable=trainable, initializer=tf.constant_initializer(t)))
        else:
            params_to_return.append(t)
    return params_to_return


def new_tensorfactored_conv_layer(layer_input, k_size, ch_size, stride_size, layer_num, task_num, activation_fn=tf.nn.relu, KB_param=None, TS_param=None, padding_type='SAME', max_pool=False, pool_size=None, skip_connect_input=None, highway_connect_type=0, highway_W=None, highway_b=None, trainable=True, trainable_KB=True):
    with tf.name_scope('TF_conv'):
        KB_param = new_TF_KB_param(k_size+ch_size, layer_num, KB_param, trainable=trainable_KB)

        TS_param = new_TF_TS_param([[a, a] for a in k_size+ch_size]+[ch_size[1]], layer_num, task_num, TS_param if TS_param else [None, None, None, None, None], trainable)

    with tf.name_scope('TF_param_gen'):
        W = KB_param
        for t in TS_param[:-1]:
            W = tf.tensordot(W, t, [[0], [0]])
        b = TS_param[-1]

    ## HighwayNet's skip connection
    highway_params, gate = [], None
    if highway_connect_type > 0:
        with tf.name_scope('highway_connection'):
            if highway_connect_type == 1:
                x = layer_input
                if highway_W is None:
                    highway_W = new_weight([k_size[0], k_size[1], ch_size[0], ch_size[1]])
                if highway_b is None:
                    highway_b = new_bias([ch_size[1]], init_val=-2.0)
                gate, _ = new_cnn_layer(x, k_size+ch_size, stride_size=stride_size, activation_fn=None, weight=highway_W, bias=highway_b, padding_type=padding_type, max_pooling=False)
            elif highway_connect_type == 2:
                x = tf.reshape(layer_input, [-1, int(layer_input.shape[1]*layer_input.shape[2]*layer_input.shape[3])])
                if highway_W is None:
                    highway_W = new_weight([int(x.shape[1]), 1])
                if highway_b is None:
                    highway_b = new_bias([1], init_val=-2.0)
                gate = tf.broadcast_to(tf.stack([tf.stack([tf.matmul(x, highway_W) + highway_b], axis=2)], axis=3), layer_input.get_shape())
            gate = tf.nn.sigmoid(gate)
        highway_params = [highway_W, highway_b]

    layer_eqn, _ = new_cnn_layer(layer_input, k_size+ch_size, stride_size=stride_size, activation_fn=activation_fn, weight=W, bias=b, padding_type=padding_type, max_pooling=max_pool, pool_size=pool_size, skip_connect_input=skip_connect_input, highway_connect_type=highway_connect_type, highway_gate=gate)
    return layer_eqn, [KB_param], TS_param, [W, b], highway_params


def new_hybrid_tensorfactored_cnn_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_sharing, cnn_activation_fn=tf.nn.relu, cnn_KB_params=None, cnn_TS_params=None, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, task_index=0, skip_connections=[], highway_connect_type=0, cnn_highway_params=None, trainable=True, trainable_KB=True):
    _num_TS_param_per_layer = 5

    num_conv_layers = [len(k_sizes)//2, len(ch_sizes)-1, len(stride_sizes)//2, len(cnn_sharing)]
    assert (all([(num_conv_layers[i]==num_conv_layers[i+1]) for i in range(len(num_conv_layers)-1)])), "Parameters related to conv layers are wrong!"
    num_conv_layers = num_conv_layers[0]

    ## add CNN layers
    ## first element : make new KB&TS / second element : make new TS / third element : not make new para / fourth element : make new KB
    control_flag = [(cnn_KB_params is None and cnn_TS_params is None), (not (cnn_KB_params is None) and (cnn_TS_params is None)), not (cnn_KB_params is None or cnn_TS_params is None), ((cnn_KB_params is None) and not (cnn_TS_params is None))]
    if control_flag[1]:
        cnn_TS_params = []
    elif control_flag[3]:
        cnn_KB_params = []
    elif control_flag[0]:
        cnn_KB_params, cnn_TS_params = [], []
    cnn_gen_params = []

    if cnn_params is None:
        cnn_params = [None for _ in range(2*num_conv_layers)]

    layers_for_skip, next_skip_connect = [net_input], None
    with tf.name_scope('Hybrid_TensorFactorized_CNN'):
        cnn_model, cnn_params_to_return, cnn_highway_params_to_return = [], [], []
        cnn_KB_to_return, cnn_TS_to_return = [], []
        for layer_cnt in range(num_conv_layers):
            KB_para_tmp, TS_para_tmp, para_tmp = [None], [None for _ in range(_num_TS_param_per_layer)], [None, None]
            highway_para_tmp = [None, None] if cnn_highway_params is None else cnn_highway_params[2*layer_cnt:2*(layer_cnt+1)]
            cnn_gen_para_tmp = [None, None]

            next_skip_connect = skip_connections.pop(0) if (len(skip_connections) > 0 and next_skip_connect is None) else next_skip_connect
            if next_skip_connect is not None:
                skip_connect_in, skip_connect_out = next_skip_connect
                assert (skip_connect_in > -1 and skip_connect_out > -1), "Given skip connection has error (try connecting non-existing layer)"
            else:
                skip_connect_in, skip_connect_out = -1, -1

            if layer_cnt == skip_connect_out:
                processed_skip_connect_input = layers_for_skip[skip_connect_in]
                for layer_cnt_tmp in range(skip_connect_in, skip_connect_out):
                    if max_pool and (pool_sizes[2*layer_cnt_tmp]>1 or pool_sizes[2*layer_cnt_tmp+1]>1):
                        processed_skip_connect_input = tf.nn.max_pool(processed_skip_connect_input, ksize=[1]+pool_sizes[2*layer_cnt_tmp:2*(layer_cnt_tmp+1)]+[1], strides=[1]+pool_sizes[2*layer_cnt_tmp:2*(layer_cnt_tmp+1)]+[1], padding=padding_type)
            else:
                processed_skip_connect_input = None

            if control_flag[0] and cnn_sharing[layer_cnt]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_tensorfactored_conv_layer(net_input if layer_cnt<1 else cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], layer_cnt, task_index, activation_fn=cnn_activation_fn, KB_param=None, TS_param=None, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
            elif control_flag[1] and cnn_sharing[layer_cnt]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_tensorfactored_conv_layer(net_input if layer_cnt<1 else cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], layer_cnt, task_index, activation_fn=cnn_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=None, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
            elif control_flag[2] and cnn_sharing[layer_cnt]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_tensorfactored_conv_layer(net_input if layer_cnt<1 else cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], layer_cnt, task_index, activation_fn=cnn_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
            elif control_flag[3] and cnn_sharing[layer_cnt]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_tensorfactored_conv_layer(net_input if layer_cnt<1 else cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], layer_cnt, task_index, activation_fn=cnn_activation_fn, KB_param=None, TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
            elif (not cnn_sharing[layer_cnt]):
                layer_tmp, para_tmp = new_cnn_layer(layer_input=net_input if layer_cnt<1 else cnn_model[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=cnn_params[2*layer_cnt], bias=cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable)

            cnn_model.append(layer_tmp)
            layers_for_skip.append(layer_tmp)
            cnn_KB_to_return = cnn_KB_to_return + KB_para_tmp
            cnn_TS_to_return = cnn_TS_to_return + TS_para_tmp
            cnn_params_to_return = cnn_params_to_return + para_tmp
            cnn_gen_params = cnn_gen_params + cnn_gen_para_tmp
            cnn_highway_params_to_return = cnn_highway_params_to_return + highway_para_tmp
            if layer_cnt == skip_connect_out:
                next_skip_connect = None

        #### flattening output
        output_dim = [int(cnn_model[-1].shape[1]*cnn_model[-1].shape[2]*cnn_model[-1].shape[3])]
        cnn_model.append(tf.reshape(cnn_model[-1], [-1, output_dim[0]]))

        #### add dropout layer
        if dropout:
            cnn_model.append(tf.nn.dropout(cnn_model[-1], dropout_prob))

    ## add fc layers
    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, tensorboard_name_scope='fc_net', trainable=trainable)

    return (cnn_model+fc_model, cnn_KB_to_return, cnn_TS_to_return, cnn_gen_params, cnn_params_to_return, cnn_highway_params_to_return, fc_params)
