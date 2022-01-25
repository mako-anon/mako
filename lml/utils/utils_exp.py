from tensorflow.nn import relu


def model_setup(data_type, data_input_dim, model_type, test_type=0, cnn_padding_type_same=True, skip_connect_test_type=0, highway_connect_test_type=0, num_clayers=-1, phase1_max_epoch=100, darts_approx_order=1):
    model_architecture = None
    model_hyperpara = {}
    if cnn_padding_type_same:
        model_hyperpara['padding_type'] = 'SAME'
    else:
        model_hyperpara['padding_type'] = 'VALID'
    model_hyperpara['max_pooling'] = True
    model_hyperpara['dropout'] = True
    model_hyperpara['image_dimension'] = data_input_dim
    model_hyperpara['skip_connect'] = []
    model_hyperpara['highway_connect'] = highway_connect_test_type
    model_hyperpara['hidden_activation'] = relu

    if 'mnist' in data_type:
        model_hyperpara['batch_size'] = 10
        model_hyperpara['hidden_layer'] = [32]
        model_hyperpara['kernel_sizes'] = [5, 5, 5, 5]
        model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
        model_hyperpara['channel_sizes'] = [32, 64]
        model_hyperpara['pooling_size'] = [2, 2, 2, 2]

    elif ('cifar10' in data_type) and not ('cifar100' in data_type):
        model_hyperpara['batch_size'] = 20
        model_hyperpara['hidden_layer'] = [64]
        model_hyperpara['kernel_sizes'] = [3, 3, 3, 3, 3, 3, 3, 3]
        model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1]
        model_hyperpara['channel_sizes'] = [32, 32, 64, 64]
        model_hyperpara['pooling_size'] = [1, 1, 2, 2, 1, 1, 2, 2]

    elif 'cifar100' in data_type:
        if num_clayers < 1:
            num_clayers = 4

        model_hyperpara['batch_size'] = 10
        if num_clayers == 4:
            model_hyperpara['hidden_layer'] = [64]
            model_hyperpara['kernel_sizes'] = [3, 3, 3, 3, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [32, 32, 64, 64]
            model_hyperpara['pooling_size'] = [1, 1, 2, 2, 1, 1, 2, 2]

        elif num_clayers == 6:
            model_hyperpara['hidden_layer'] = [64]
            model_hyperpara['kernel_sizes'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [32, 32, 64, 64, 128, 128]
            model_hyperpara['pooling_size'] = [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]

        elif num_clayers == 8:
            model_hyperpara['hidden_layer'] = [64]
            model_hyperpara['kernel_sizes'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [32, 32, 64, 64, 128, 128, 128, 128]
            model_hyperpara['pooling_size'] = [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]

        elif num_clayers == 17:
            ## ResNet-18 Architecture
            model_hyperpara['hidden_layer'] = [1000]
            model_hyperpara['kernel_sizes'] = [7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
            model_hyperpara['pooling_size'] = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1]
            model_hyperpara['skip_connect'] = [[1, 2], [3, 4], [7, 8], [11, 12], [15, 16]]

        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    if model_type.lower() == 'stl':
        model_architecture = 'stl_cnn'

    elif model_type.lower() == 'snn':
        model_architecture = 'singlenn_cnn'

        # Note: parameters here specifically for instance-incremental CNNL experiments!
        if 'mnist' in data_type:
            model_hyperpara['hidden_layer'] = [128]
            model_hyperpara['kernel_sizes'] = [3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [32, 32]
            model_hyperpara['pooling_size'] = [1, 1, 2, 2]
        if 'cifar10' in data_type and 'cifar100' not in data_type:
            model_hyperpara['hidden_layer'] = [128]
            model_hyperpara['kernel_sizes'] = [3, 3, 3, 3, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [32, 32, 32, 32]
            model_hyperpara['pooling_size'] = [1, 1, 2, 2, 1, 1, 2, 2]

    elif model_type.lower() == 'hps' or model_type.lower() == 'hybrid_hps':
        model_architecture = 'hybrid_hps_cnn'

    elif model_type.lower() == 'tf':
        model_architecture = 'mtl_tf_cnn'
        model_hyperpara['tensor_factor_type'] = 'Tucker'
        if test_type == 0:
            model_hyperpara['tensor_factor_error_threshold'] = 0.3
        elif test_type == 1:
            model_hyperpara['tensor_factor_error_threshold'] = 0.1
        elif test_type == 2:
            model_hyperpara['tensor_factor_error_threshold'] = 1e-3
        elif test_type == 3:
            model_hyperpara['tensor_factor_error_threshold'] = 1e-5
        elif test_type == 4:
            model_hyperpara['tensor_factor_error_threshold'] = 1e-7
        elif test_type == 5:
            model_hyperpara['tensor_factor_error_threshold'] = 1e-9

    elif model_type.lower() == 'hybrid_tf':
        model_architecture = 'hybrid_tf_cnn'
        if test_type%5 == 0:
            model_hyperpara['auxiliary_loss_weight'] = 0.1
        elif test_type%5 == 1:
            model_hyperpara['auxiliary_loss_weight'] = 0.05
        elif test_type%5 == 2:
            model_hyperpara['auxiliary_loss_weight'] = 0.01
        elif test_type%5 == 3:
            model_hyperpara['auxiliary_loss_weight'] = 0.005
        elif test_type%5 == 4:
            model_hyperpara['auxiliary_loss_weight'] = 0.001

    elif model_type.lower() == 'prog' or model_type.lower() == 'prognn':
        model_architecture = 'prognn_cnn'
        if test_type == 0:
            model_hyperpara['dim_reduction_scale'] = 1.0
        elif test_type == 1:
            model_hyperpara['dim_reduction_scale'] = 2.0

    elif model_type.lower() == 'den':
        model_architecture = 'den_cnn'
        model_hyperpara['l1_lambda'] = 1e-6
        model_hyperpara['l2_lambda'] = 0.01
        model_hyperpara['gl_lambda'] = 0.8
        model_hyperpara['reg_lambda'] = 0.5
        model_hyperpara['loss_threshold'] = 0.01
        model_hyperpara['sparsity_threshold'] = [0.98, 0.01]

        if 'mnist' in data_type:
            if num_clayers == 2:
                model_hyperpara['den_expansion'] = 8
                model_hyperpara['hidden_layer'] = [32]
        elif 'cifar' in data_type:
            if num_clayers == 4:
                model_hyperpara['den_expansion'] = 16
                model_hyperpara['hidden_layer'] = [128]

    elif model_type.lower() == 'hybrid_dfcnn':
        set_conv_sharing_bias = False
        model_architecture = 'hybrid_dfcnn'
        if test_type%5 == 0:
            model_hyperpara['regularization_scale'] = [0.0, 1e-7, 0.0, 1e-7]
        elif test_type%5 == 1:
            model_hyperpara['regularization_scale'] = [0.0, 1e-7, 0.0, 1e-9]
        elif test_type%5 == 2:
            model_hyperpara['regularization_scale'] = [0.0, 1e-7, 0.0, 1e-11]
        elif test_type%5 == 3:
            model_hyperpara['regularization_scale'] = [0.0, 1e-9, 0.0, 1e-9]
        elif test_type%5 == 4:
            model_hyperpara['regularization_scale'] = [0.0, 1e-9, 0.0, 1e-11]

        if data_type == 'mnist_mako':
            model_hyperpara['cnn_KB_sizes'] = [3, 12, 3, 24]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
        elif ('cifar10' in data_type) and not ('cifar100' in data_type):
            if num_clayers == 4:
                print("\nHybrid DF-CNN 4 layers\n")
                model_hyperpara['cnn_KB_sizes'] = [2, 16, 2, 24, 2, 32, 2, 36]
                model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48, 3, 64, 3, 72]
                model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
                if set_conv_sharing_bias:
                    model_hyperpara['conv_sharing_bias'] = [0.0, 0.0, 0.0, 0.0]
            else:
                raise NotImplementedError

        elif 'cifar100' in data_type:
            if num_clayers == 4:
                print("\nHybrid DF-CNN 4 layers\n")
                model_hyperpara['cnn_KB_sizes'] = [2, 16, 2, 24, 2, 32, 2, 36]
                model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48, 3, 64, 3, 72]
                model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
                if set_conv_sharing_bias:
                    model_hyperpara['conv_sharing_bias'] = [0.0, 0.0, 0.0, 0.0]
            elif num_clayers == 6:
                print("\nHybrid DF-CNN 6 layers\n")
                model_hyperpara['cnn_KB_sizes'] = [2, 16, 2, 24, 2, 32, 2, 36, 2, 64, 2, 72]
                model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48, 3, 64, 3, 72, 3, 128, 3, 144]
                model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                if set_conv_sharing_bias:
                    model_hyperpara['conv_sharing_bias'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif num_clayers == 8:
                print("\nHybrid DF-CNN 8 layers\n")
                model_hyperpara['cnn_KB_sizes'] = [2, 16, 2, 24, 2, 32, 2, 36, 2, 64, 2, 72, 2, 72, 2, 72]
                model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48, 3, 64, 3, 72, 3, 128, 3, 144, 3, 144, 3, 144]
                model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                if set_conv_sharing_bias:
                    model_hyperpara['conv_sharing_bias'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                raise NotImplementedError

    else:
        model_hyperpara = None

    if (model_type.lower() == 'hps') or (model_type.lower() == 'hybrid_hps') or (model_type.lower() == 'hybrid_tf') or (model_type.lower() == 'hybrid_dfcnn'):
        if num_clayers == 2:
            if test_type < 5:
                model_hyperpara['conv_sharing'] = [True, True]
            elif test_type < 10:
                model_hyperpara['conv_sharing'] = [False, True]
            elif test_type < 15:
                model_hyperpara['conv_sharing'] = [True, False]

        elif num_clayers == 4:
            if test_type < 5:
                model_hyperpara['conv_sharing'] = [True, True, True, True]
            elif test_type < 10:
                model_hyperpara['conv_sharing'] = [False, False, False, True]
            elif test_type < 15:
                model_hyperpara['conv_sharing'] = [False, False, True, True]
            elif test_type < 20:
                model_hyperpara['conv_sharing'] = [False, True, True, True]
            elif test_type < 25:
                model_hyperpara['conv_sharing'] = [True, False, False, False]
            elif test_type < 30:
                model_hyperpara['conv_sharing'] = [True, True, False, False]
            elif test_type < 35:
                model_hyperpara['conv_sharing'] = [True, True, True, False]
            elif test_type < 40:
                model_hyperpara['conv_sharing'] = [False, True, False, True]
            elif test_type < 45:
                model_hyperpara['conv_sharing'] = [True, False, True, False]

        elif num_clayers == 6:
            if test_type < 5:
                model_hyperpara['conv_sharing'] = [True, True, True, True, True, True]
            elif test_type < 10:
                model_hyperpara['conv_sharing'] = [False, False, False, False, False, True]
            elif test_type < 15:
                model_hyperpara['conv_sharing'] = [False, False, False, False, True, True]
            elif test_type < 20:
                model_hyperpara['conv_sharing'] = [False, False, False, True, True, True]
            elif test_type < 25:
                model_hyperpara['conv_sharing'] = [False, False, True, True, True, True]
            elif test_type < 30:
                model_hyperpara['conv_sharing'] = [False, True, True, True, True, True]
            elif test_type < 35:
                model_hyperpara['conv_sharing'] = [True, False, False, False, False, False]
            elif test_type < 40:
                model_hyperpara['conv_sharing'] = [True, True, False, False, False, False]
            elif test_type < 45:
                model_hyperpara['conv_sharing'] = [True, True, True, False, False, False]
            elif test_type < 50:
                model_hyperpara['conv_sharing'] = [True, True, True, True, False, False]
            elif test_type < 55:
                model_hyperpara['conv_sharing'] = [True, True, True, True, True, False]
            elif test_type < 60:
                model_hyperpara['conv_sharing'] = [False, True, False, True, False, True]

        elif num_clayers == 8:
            if test_type < 5:
                model_hyperpara['conv_sharing'] = [True, True, True, True, True, True, True, True]
            elif test_type < 10:
                model_hyperpara['conv_sharing'] = [False, False, False, False, False, False, True, True]
            elif test_type < 15:
                model_hyperpara['conv_sharing'] = [False, False, False, False, True, True, True, True]
            elif test_type < 20:
                model_hyperpara['conv_sharing'] = [False, False, True, True, True, True, True, True]
            elif test_type < 25:
                model_hyperpara['conv_sharing'] = [True, True, False, False, False, False, False, False]
            elif test_type < 30:
                model_hyperpara['conv_sharing'] = [True, True, True, True, False, False, False, False]
            elif test_type < 35:
                model_hyperpara['conv_sharing'] = [True, True, True, True, True, True, False, False]
            elif test_type < 40:
                model_hyperpara['conv_sharing'] = [False, True, False, True, False, True, False, True]
            elif test_type < 45:
                model_hyperpara['conv_sharing'] = [True, False, True, False, True, False, True, False]

        elif num_clayers == 9:
            if test_type < 5:
                model_hyperpara['conv_sharing'] = [True, True, True, True, True, True, True, True, True]

        elif num_clayers == 17:
            if test_type < 5:
                model_hyperpara['conv_sharing'] = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]

        else:
            raise NotImplementedError

    return (model_architecture, model_hyperpara)
