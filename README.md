# Semi-supervised Lifelong Learning Experiments

![alt text](https://github.com/mako-anon/mako/blob/master/figs/fig.png)

This code repository is used for semi-supervised lifelong learning experiments presented in paper **Mako: semi-supervised continual learning with minimal labeled data via data programming**.

## Version and Dependencies
- Python 3.6 or higher

- Snorkel 0.95

## Data

Data generation examples can be found in utils/generate_task_data.py, and Mako labeling examples in utils/run_label_generator.py. The main file now generates and labels all data for all experiments, which will take a very long time.

- MNIST

- CIFAR-10

- CIFAR-100 (Lifelong)

## Lifelong Machine Learning Tools
- Single Neural Net model
    - Construct a single neural network, and treat data of all task as same.

- [Tensor Factorization](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-BulatA.1460.pdf)
    - Factorize parameter of each layer into multiplication of several tensors, and share one tensor over different tasks as knowledge base. (Details in the paper Bulat, Adrian, Jean Kossaifi, Georgios Tzimiropoulos, and Maja Pantic. "Incremental multi-domain learning with network latent tensor factorization." ICML (2020).)

- [Dynamically Expandable Network model](https://arxiv.org/abs/1708.01547)
    - Extended hard-parameter shared model by retraining some neurons selectively/adding new neurons/splitting neurons into disjoint groups for different set of tasks according to the given data.
    - The code (cnn_den_model.py) is provided by authors.

- [DF-CNN](https://proceedings.mlr.press/v139/lee21a.html)
    - Deconvolution (transposed convolution) reconstructs filters of task-specific convolutional layers from the shared knowledge base.
    - The code is available online.

## Sample command to train a specific model
### Comparison to supervised LML
1. Tensor Factorization on MNIST (45 binary tasks): labeled set and *120* instances of unlabeled set per task

    - with Mako labels: ```python3 main_mako.py --gpu 0 --data_type mnist_mako --num_clayers 2 --model_type Hybrid_TF --test_type 4 --lifelong --data_unlabel 120 --save_mat_name mnist_tf_unlabel120.mat```
    - with true labels: ```python3 main_mako.py --gpu 0 --data_type mnist_mako --num_clayers 2 --model_type Hybrid_TF --test_type 4 --lifelong --data_unlabel 120 --use_true_label --save_mat_name mnist_tf_unlabel120_gt.mat```

2. DF-CNN on CIFAR-10 (45 binary tasks): labeled set and *400* instances of unlabeled set per task

    - with Mako labels: ```python3 main_mako.py --gpu 0 --data_type cifar10_mako --num_clayers 4 --model_type Hybrid_DFCNN --test_type 4 --lifelong --data_unlabel 400 --save_mat_name cifar10_dfcnn_unlabel400.mat```
    - with true labels: ```python3 main_mako.py --gpu 0 --data_type cifar10_mako --num_clayers 4 --model_type Hybrid_DFCNN --test_type 4 --lifelong --data_unlabel 400 --use_true_label --save_mat_name cifar10_dfcnn_unlabel400_gt.mat```

### Comparison to semi-supervised LML
1. CNNL MNIST experiment

  - with Mako labels: ```python3 main_mako.py --gpu 0 --data_type mnist_mako --model_type SNN --save_mat_name mnist_5way_inst_incr.mat --lifelong --num_classes 5 --instance_incremental```
  - with true labels: ```python3 main_mako.py --gpu 0 --data_type mnist_mako --model_type SNN --save_mat_name mnist_5way_inst_incr_gt.mat --lifelong --num_classes 5 --instance_incremental --use_true_label```

2. CNNL CIFAR-10 experiment

  - with Mako labels: ```python3 main_mako.py --gpu 0 --data_type cifar10_mako --model_type SNN --save_mat_name cifar10_10way_inst_incr.mat --lifelong --num_classes 10 --instance_incremental```
  - with true labels: ```python3 main_mako.py --gpu 0 --data_type cifar10_mako --model_type SNN --save_mat_name cifar10_10way_inst_incr_gt.mat --lifelong --num_classes 10 --instance_incremental --use_true_label```

3. ORDISCO CIFAR-10 experiment (class-incremental)

    - with Mako labels: ```python3 main_mako.py --gpu 0 --data_type cifar10_mako --model_type hybrid_dfcnn --test_type 4 --num_clayers 4 --save_mat_name cifar10_class_incr_ordisco-dfcnn.mat --lifelong --num_classes 2 --data_unlabel 5000 --ordisco_ci_baseline```
    - with true labels: ```python3 main_mako.py --gpu 0 --data_type cifar10_mako --model_type hybrid_dfcnn --test_type 4 --num_clayers 4 --save_mat_name cifar10_class_incr_ordisco-dfcnn_gt.mat --lifelong --num_classes 2 --data_unlabel 5000 --ordisco_ci_baseline --use_true_label```
    - with Mako labels, tensor factorization: ```python3 main_mako.py --gpu 0 --data_type cifar10_mako --model_type hybrid_tf --test_type 2 --num_clayers 4 --save_mat_name cifar10_class_incr_ordisco-tf.mat --lifelong --num_classes 2 --data_unlabel 5000 --ordisco_ci_baseline```
    - with true labels, tensor factorization: ```python3 main_mako.py --gpu 0 --data_type cifar10_mako --model_type hybrid_tf --test_type 2 --num_clayers 4 --save_mat_name cifar10_class_incr_ordisco-tf_gt.mat --lifelong --num_classes 2 --data_unlabel 5000 --ordisco_ci_baseline --use_true_label```
    - with Mako labels, DEN: ```python3 main_mako.py --gpu 1 --data_type cifar10_mako --model_type den --test_type 2 --num_clayers 4 --save_mat_name cifar10_class_incr_ordisco-den.mat --lifelong --num_classes 2 --data_unlabel 5000 --ordisco_ci_baseline```
    - with Mako labels, DEN: ```python3 main_mako.py --gpu 1 --data_type cifar10_mako --model_type den --test_type 2 --num_clayers 4 --save_mat_name cifar10_class_incr_ordisco-den_gt.mat --lifelong --num_classes 2 --data_unlabel 5000 --ordisco_ci_baseline --use_true_label```

4. DistillMatch CIFAR-100 experiment:

  - with Mako labels, DF-CNN: ```python3 main_mako.py --gpu 0 --data_type cifar100_mako --model_type hybrid_dfcnn --test_type 4 --num_clayers 4 --save_mat_name cifar100_class_incr_distillmatch-dfcnn.mat --lifelong --num_classes 5 --data_unlabel 2000```
  - with true labels, DF-CNN: ```python3 main_mako.py --gpu 1 --data_type cifar100_mako --model_type hybrid_dfcnn --test_type 4 --num_clayers 4 --save_mat_name cifar100_class_incr_distillmatch-dfcnn_gt.mat --lifelong --num_classes 5 --data_unlabel 2000 --use_true_label```
  - with Mako labels, tensor factorization: ```python3 main_mako.py --gpu 0 --data_type cifar100_mako --model_type hybrid_tf --test_type 2 --num_clayers 4 --save_mat_name cifar100_class_incr_distillmatch-tf.mat --lifelong --num_classes 5 --data_unlabel 2000```
  - with true labels, tensor factorization: ```python3 main_mako.py --gpu 1 --data_type cifar100_mako --model_type hybrid_tf --test_type 2 --num_clayers 4 --save_mat_name cifar100_class_incr_distillmatch-tf_gt.mat --lifelong --num_classes 5 --data_unlabel 2000 --use_true_label```
  - with Mako labels, DEN: ```python3 main_mako.py --gpu 0 --data_type cifar100_mako --model_type den --test_type 2 --num_clayers 4 --save_mat_name cifar100_class_incr_distillmatch-den.mat --lifelong --num_classes 5 --data_unlabel 2000```
  - with true labels, DEN: ```python3 main_mako.py --gpu 0 --data_type cifar100_mako --model_type den --test_type 2 --num_clayers 4 --save_mat_name cifar100_class_incr_distillmatch-den_gt.mat --lifelong --num_classes 5 --data_unlabel 2000 --use_true_label```
