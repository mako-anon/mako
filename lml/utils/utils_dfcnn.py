import numpy as np
import tensorflow as tf

from lml.utils.utils import *
from lml.utils.utils_nn import *

###########################################################
#####         functions to generate parameter         #####
###########################################################

#### function to generate knowledge-base parameters for ELLA_tensorfactor layer
def new_DFCNN_KB_param(shape, layer_number, task_number, reg_type, init_tensor=None, trainable=True):
    #kb_name = 'KB_'+str(layer_number)+'_'+str(task_number)
    kb_name = 'KB_'+str(layer_number)
    if init_tensor is None:
        param_to_return = tf.get_variable(name=kb_name, shape=shape, dtype=tf.float32, regularizer=reg_type, trainable=trainable)
    elif type(init_tensor) == np.ndarray:
        param_to_return = tf.get_variable(name=kb_name, shape=shape, dtype=tf.float32, regularizer=reg_type, initializer=tf.constant_initializer(init_tensor), trainable=trainable)
    else:
        param_to_return = init_tensor
    return param_to_return

#### function to generate task-specific parameters for ELLA_tensorfactor layer
def new_DFCNN_TS_param(shape, layer_number, task_number, reg_type, init_tensor, trainable):
    ts_w_name, ts_b_name, ts_k_name, ts_p_name = 'TS_DeconvW0_'+str(layer_number)+'_'+str(task_number), 'TS_Deconvb0_'+str(layer_number)+'_'+str(task_number), 'TS_ConvW1_'+str(layer_number)+'_'+str(task_number), 'TS_Convb0_'+str(layer_number)+'_'+str(task_number)
    params_to_return, params_name = [], [ts_w_name, ts_b_name, ts_k_name, ts_p_name]
    for i, (t, n) in enumerate(zip(init_tensor, params_name)):
        if t is None:
            params_to_return.append(tf.get_variable(name=n, shape=shape[i], dtype=tf.float32, regularizer=reg_type if trainable and i<3 else None, trainable=trainable))
        elif type(t) == np.ndarray:
            params_to_return.append(tf.get_variable(name=n, shape=shape[i], dtype=tf.float32, regularizer=reg_type if trainable and i<3 else None, trainable=trainable, initializer=tf.constant_initializer(t)))
        else:
            params_to_return.append(t)
    return params_to_return


###########################################################################
##### functions for adding ELLA network (CNN/Deconv & Tensordot ver)  #####
###########################################################################

#### KB_size : [filter_height(and width), num_of_channel]
#### TS_size : [deconv_filter_height(and width), deconv_filter_channel]
#### TS_stride_size : [stride_in_height, stride_in_width]
def new_DFCNN_layer(layer_input, k_size, ch_size, stride_size, KB_size, TS_size, TS_stride_size, layer_num, task_num, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_size=None, skip_connect_input=None, highway_connect_type=0, highway_W=None, highway_b=None, trainable=True, trainable_KB=True):
    assert (k_size[0] == k_size[1] and k_size[0] == (KB_size[0]-1)*TS_stride_size[0]+1), "CNN kernel size does not match the output size of Deconv from KB"

    with tf.name_scope('ELLA_cdnn_KB'):
        ## KB \in R^{1 \times h \times w \times c}
        KB_param = new_DFCNN_KB_param([1, KB_size[0], KB_size[0], KB_size[1]], layer_num, task_num, KB_reg_type, KB_param, trainable=trainable_KB)

        ## TS1 : Deconv W \in R^{h \times w \times kb_c_out \times c}
        ## TS2 : Deconv bias \in R^{kb_c_out}
        ## TS3 : tensor W \in R^{kb_c_out \times ch_in \times ch_out}
        ## TS4 : Conv bias \in R^{ch_out}
        TS_param = new_DFCNN_TS_param([[TS_size[0], TS_size[0], TS_size[1], KB_size[1]], [1, 1, 1, TS_size[1]], [TS_size[1], ch_size[0], ch_size[1]], [ch_size[1]]], layer_num, task_num, TS_reg_type, [None, None, None, None] if TS_param is None else TS_param, trainable=trainable)

    with tf.name_scope('DFCNN_param_gen'):
        para_tmp = tf.add(tf.nn.conv2d_transpose(KB_param, TS_param[0], [1, k_size[0], k_size[1], TS_size[1]], strides=[1, TS_stride_size[0], TS_stride_size[1], 1]), TS_param[1])
        para_tmp = tf.reshape(para_tmp, [k_size[0], k_size[1], TS_size[1]])
        if para_activation_fn is not None:
            para_tmp = para_activation_fn(para_tmp)
        W = tf.tensordot(para_tmp, TS_param[2], [[2], [0]])
        b = TS_param[3]

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

##############################################################################################################
####  functions for Conv-FC nets whose conv layers are freely set to shared across tasks by DeconvFactor  ####
##############################################################################################################
def new_hybrid_dfcnn_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_sharing, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, cnn_activation_fn=tf.nn.relu, cnn_para_activation_fn=tf.nn.relu, cnn_KB_params=None, cnn_TS_params=None, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, task_index=0, skip_connections=[], highway_connect_type=0, cnn_highway_params=None, trainable=True, trainable_KB=True):
    _num_TS_param_per_layer = 4

    num_conv_layers = [len(k_sizes)//2, len(ch_sizes)-1, len(stride_sizes)//2, len(cnn_sharing), len(cnn_KB_sizes)//2, len(cnn_TS_sizes)//2, len(cnn_TS_stride_sizes)//2]
    assert (all([(num_conv_layers[i]==num_conv_layers[i+1]) for i in range(len(num_conv_layers)-1)])), "Parameters related to conv layers are wrong!"
    num_conv_layers = num_conv_layers[0]
    '''
    if cnn_KB_params is not None:
        assert (len(cnn_KB_params) == 1), "Given init value of KB (last layer) is wrong!"
    if cnn_TS_params is not None:
        assert (len(cnn_TS_params) == 4), "Given init value of TS (last layer) is wrong!"
    '''

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
    with tf.name_scope('Hybrid_DFCNN'):
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

            if layer_cnt == 0:
                if control_flag[0] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_DFCNN_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[1] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_DFCNN_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[2] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_DFCNN_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[3] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_DFCNN_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=None, TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                elif (not cnn_sharing[layer_cnt]):
                    layer_tmp, para_tmp = new_cnn_layer(layer_input=net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=cnn_params[2*layer_cnt], bias=cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable)
            else:
                if control_flag[0] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_DFCNN_layer(cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[1] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_DFCNN_layer(cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[2] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_DFCNN_layer(cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[3] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_DFCNN_layer(cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=None, TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                elif (not cnn_sharing[layer_cnt]):
                    layer_tmp, para_tmp = new_cnn_layer(layer_input=cnn_model[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=cnn_params[2*layer_cnt], bias=cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable)

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
