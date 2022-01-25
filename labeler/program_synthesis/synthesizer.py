import numpy as np
import itertools

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from mako.labeler.lenet_weak_labeler import LeNetWeakLabeler
from mako.labeler.cifar_weak_labeler import CifarWeakLabeler
from mako.utils.bootstrapping import bootstrap_xy_balanced_class

import torch
if torch.cuda.is_available():
    print("Cuda available")
    DEVICE = "cuda:0"
else:
    print("Cuda not available")
    DEVICE = "cpu"


class Synthesizer(object):
    """
    A class to synthesize heuristics from primitives and validation labels
    """

    # cnn: if we are using LenetWeakLabler as weak labeler
    # num_cnn_labelers: number of guesses for one weak labeler
    # task: 'mnist', 'fashion', 'cifar10', 'omniglot' or 'all': generate labelers with different architectures
    def __init__(self, primitive_matrix, val_ground, b=0.5, cnn=False, num_cnn_labelers=10, task='mnist',
                 lr=1e-3, n_batches=5, n_epochs=80, n_classes=2, bootstrap_size_per_class=25):
        """ 
        Initialize Synthesizer object

        b: class prior of most likely class
        beta: threshold to decide whether to abstain or label for heuristics
        """
        self.val_primitive_matrix = primitive_matrix
        self.val_ground = val_ground
        self.p = np.shape(self.val_primitive_matrix)[1]
        self.b = b
        self.cnn = cnn
        if self.cnn:
            self.num_cnn_labelers = num_cnn_labelers
            self.task = task
            self.lr = lr
            self.n_batches = n_batches
            self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.bootstrap_size_per_class = bootstrap_size_per_class
        
    # 1. Not used in our CNN version
    def generate_feature_combinations(self, cardinality=1):
        """ 
        Create a list of primitive index combinations for given cardinality

        max_cardinality: max number of features each heuristic operates over 
        """
        primitive_idx = range(self.p)
        feature_combinations = []

        for comb in itertools.combinations(primitive_idx, cardinality):
            feature_combinations.append(comb)

        return feature_combinations

    # 2. In our CNN version, use all features
    def fit_function(self, comb, model):
        """ 
        Fits a single logistic regression or decision tree model

        comb: feature combination to fit model over
        model: fit logistic regression or a decision tree
        """
        if not self.cnn:
            X = self.val_primitive_matrix[:, comb]
            if np.shape(X)[0] == 1:
                X = X.reshape(-1, 1)
        else:
            X = self.val_primitive_matrix

        # fit decision tree or logistic regression or knn
        if model == 'dt':
            dt = DecisionTreeClassifier(max_depth=len(comb))
            dt.fit(X, self.val_ground)
            return dt

        elif model == 'lr':
            lr = LogisticRegression()
            lr.fit(X, self.val_ground)
            return lr

        elif model == 'nn':
            nn = KNeighborsClassifier(algorithm='kd_tree')
            nn.fit(X, self.val_ground)
            return nn

        # Additional option to Snuba: LeNetWeakLabeler as labeling function
        # Randomization on bootstrapping data
        elif model == 'cnn':
            dict_training_param = {'learning_rate': self.lr, 'num_batches': self.n_batches, 'num_epochs': self.n_epochs}
            X_boot, y_boot = bootstrap_xy_balanced_class(self.val_primitive_matrix, self.val_ground,
                                                         size_per_class=self.bootstrap_size_per_class)
            if self.task == 'mnist' or self.task == 'fashion':
                cnn = LeNetWeakLabeler(in_dim_h=28, in_dim_w=28, in_dim_c=1, out_dim=2,
                                       dict_training_param=dict_training_param).to(DEVICE)
            elif self.task == 'cifar10':
                cnn = CifarWeakLabeler(in_dim_h=32, in_dim_w=32, in_dim_c=3, out_dim=2,
                                       dict_training_param=dict_training_param).to(DEVICE)
            elif self.task == 'mnist_5_way':
                cnn = LeNetWeakLabeler(in_dim_h=28, in_dim_w=28, in_dim_c=1, out_dim=5,
                                       dict_training_param=dict_training_param).to(DEVICE)
            elif self.task == 'cifar10_10_way':
                cnn = CifarWeakLabeler(in_dim_h=32, in_dim_w=32, in_dim_c=3, out_dim=10,
                                       dict_training_param=dict_training_param).to(DEVICE)
            elif self.task == 'cifar100_5_way':
                cnn = CifarWeakLabeler(in_dim_h=32, in_dim_w=32, in_dim_c=3, out_dim=10,
                                       dict_training_param=dict_training_param).to(DEVICE)
            else:
                raise NotImplementedError
            cnn.train()
            cnn.train_cnn(X_boot, y_boot)
            cnn.eval()
            return cnn

    def generate_heuristics(self, model, max_cardinality=1):
        """ 
        Generates heuristics over given feature cardinality

        model: fit logistic regression or a decision tree
        max_cardinality: max number of features each heuristic operates over
        """
        #have to make a dictionary?? or feature combinations here? or list of arrays?
        feature_combinations_final = []
        heuristics_final = []
        
        if not self.cnn:
            for cardinality in range(1, max_cardinality+1):
                feature_combinations = self.generate_feature_combinations(cardinality)
                heuristics = []
                for i, comb in enumerate(feature_combinations):
                    heuristics.append(self.fit_function(comb, model))
                feature_combinations_final.append(feature_combinations)
                heuristics_final.append(heuristics)
        else:  # 3. Use all pixels in image for cnn; no feature selection needed
            comb = range(self.val_primitive_matrix.shape[1])
            feature_combinations_final.append([comb])
            heuristics = []
            for i in range(self.num_cnn_labelers):
                heuristics.append(self.fit_function(comb, 'cnn'))
            heuristics_final.append(heuristics)

        return heuristics_final, feature_combinations_final

    def beta_optimizer(self, marginals, ground):
        """ 
        Returns the best beta parameter for abstain threshold given marginals
        Uses F1 score that maximizes the F1 score

        marginals: confidences for data from a single heuristic
        """	

        #Set the range of beta params
        #0.25 instead of 0.0 as a min makes controls coverage better
        beta_params = np.linspace(0.25,0.45,10)
        f1 = []
        for beta in beta_params:
            # 4. Binary version same as before
            # Multiclass version: as in binary version, assign -1 to abstain, 0 ... n_classes-1 to others
            # marginals in multiclass version is matrix of all prob
            if self.n_classes == 2:
                labels_cutoff = np.zeros(np.shape(marginals))
                labels_cutoff[marginals <= (self.b-beta)] = -1.
                labels_cutoff[marginals >= (self.b+beta)] = 1.
                f1.append(f1_score(ground, labels_cutoff, average='micro'))
            else:
                labels_cutoff = np.zeros(np.shape(marginals)[0])
                marginals_best = np.max(marginals, axis=1)
                # only higher than chance makes sense because of 3-way
                labels_cutoff[marginals_best >= (self.b + beta)] = np.argmax(marginals, axis=1)[marginals_best >= (
                            self.b + beta)]
                labels_cutoff[marginals_best < (self.b + beta)] = -1.
                f1.append(f1_score(ground, labels_cutoff, average='micro'))

        f1 = np.nan_to_num(f1)
        return beta_params[np.argsort(np.array(f1))[-1]]

    def find_optimal_beta(self, heuristics, X, feat_combos, ground):
        """ 
        Returns optimal beta for given heuristics

        heuristics: list of pre-trained logistic regression models
        X: primitive matrix
        feat_combos: feature indices to apply heuristics to
        ground: ground truth associated with X data
        """

        beta_opt = []
        for i, hf in enumerate(heuristics):
            if not self.cnn:
                # marginals = hf.predict_proba(X[:,feat_combos[i]])[:,1]
                marginals = hf.predict_proba(X[:, feat_combos[i]])
            else: # X is a float tensor for cnn
                if self.n_classes == 2:
                    marginals = hf.marginal(X.astype('float32'))
                else:
                    marginals = hf.prob_matrix(X.astype('float32'))
            # labels_cutoff = np.zeros(np.shape(marginals))
            beta_opt.append((self.beta_optimizer(marginals, ground)))
        return beta_opt



