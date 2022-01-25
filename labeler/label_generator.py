'''
Label generator that
(1) call Snuba to generate weak labelers,
(2) call Snorkel to generate strong labels from the weak labelers
(3) temperature scaling
'''

import numpy as np
import torch
from snorkel.labeling.model.label_model import LabelModel
from labeler.program_synthesis.heuristic_generator import HeuristicGenerator
from labeler.temp_scaling import tstorch_calibrate
from labeler.lenet_weak_labeler import LeNetWeakLabeler
from labeler.omniglot_weak_labeler import OmniglotWeakLabeler
from bootstrapping import bootstrap_xy_balanced_class

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"


# Label generation using LeNetWeakLabeler as labeling functions
# Param task can be 'mnist', 'fashion', 'cifar10' for different labeling function architectures
# Input X_u, y_u (for evaluation only), X_l, y_l
class LabelGenerator:

    def __init__(self, X_u, y_u, X_l, y_l, num_guesses=100, keep=10, max_labelers=25, min_labelers=5,
                 init_acc_threshold=0.8, bootstrap_size_per_class=25, task='mnist'):
        self.X_u = X_u
        self.y_u = y_u
        self.X_l = X_l
        self.y_l = y_l
        self.num_guesses = num_guesses
        self.keep = keep
        self.max_labelers = max_labelers
        self.min_labelers = min_labelers
        self.init_acc_threshold = init_acc_threshold
        self.bootstrap_size_per_class = bootstrap_size_per_class
        self.task = task
        if task in ['mnist', 'fashion', 'cifar10']:
            self.num_classes = 2
        elif task in ['mnist_5_way', 'cifar100_5_way']:
            self.num_classes = 5
        elif task in ['cifar10_10_way']:
            self.num_classes = 10
        else:
            raise NotImplementedError
        return

    # Search for best lr, n_batches and n_epochs, can be a major overhead on hard subsets of X_l
    def generate_hyperparams(self, idx=None, bad_configs=[]):

        # Check if a config is already troublesome
        def is_bad_config(bad_configs, good_config):
            for bad_config in bad_configs:
                if good_config[0] == bad_config[0] and good_config[1] == bad_config[1] \
                        and good_config[2] == bad_config[2]:
                    return True
            return False

        if self.task in ['mnist', 'fashion', 'cifar10']:
            lr_search_space = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
            n_batches_search_space = [5, 10]
            n_epochs_search_space = [30, 40, 50, 60, 70, 80, 90, 100]
        elif self.task in ['omniglot']:
            lr_search_space = [1e-3, 0.8e-3, 1.5e-3]
            n_batches_search_space = [20]
            n_epochs_search_space = [10, 15, 20, 25]
        elif self.task in ['mnist_5_way']:
            lr_search_space = [1e-3, 0.8e-3, 1.5e-3]
            n_batches_search_space = [10, 20]
            n_epochs_search_space = [180, 200, 220]
        else:
            raise NotImplementedError

        acc_threshold = self.init_acc_threshold
        good_configs = []

        if idx is None:
            X = self.X_l
            y = self.y_l
        else:
            X = self.X_l[idx]
            y = self.y_l[idx]

        # auto-tuning training hyperparams until 8 out of 10 weak labelers meet threshold
        # if threshold too high, decrease by 0.05 until 1/num_classes
        while acc_threshold >= 1 / self.num_classes:
            print("Searching hyperparameters with acc threshold = " + str(acc_threshold))
            for n_batches in n_batches_search_space:
                if len(good_configs) > 0:  # we only need good configs with smallest epochs
                    break
                for n_epochs in n_epochs_search_space:
                    for lr in lr_search_space:
                        print(str(lr) + ", " + str(n_batches) + ", " + str(n_epochs))
                        dict_training_params = {'learning_rate': lr, 'num_batches': n_batches, 'num_epochs': n_epochs}
                        if is_bad_config(bad_configs, [lr, n_batches, n_epochs]):
                            continue

                        if self.task in ['mnist', 'fashion', 'cifar10']:
                            model = LeNetWeakLabeler(in_dim_c=X.shape[1], in_dim_h=X.shape[2],
                                                 in_dim_w=X.shape[3], out_dim=self.num_classes,
                                                 dict_training_param=dict_training_params).to(DEVICE)
                        else:
                            raise NotImplementedError

                        accs = []
                        # conf_counts = []
                        # train the model under this parameter configuration on bootstrapping subsets
                        for i in range(10):
                            print("Training for hyperparameters search: iteration " + str(i))
                            X_l_boot, y_l_boot = bootstrap_xy_balanced_class(X, y,
                                                                             size_per_class=self.bootstrap_size_per_class)
                            model.train()
                            model.train_cnn(X_l_boot, y_l_boot)
                            model.eval()
                            prob_l = model.prob_matrix(X.astype('float32'))
                            y_l_hat = np.argmax(prob_l, axis=1)
                            acc = float(np.sum(y_l_hat == y)) / y.shape[0]
                            accs.append(acc)
                            print("Training result: " + str(acc))

                        # good config requirement: average accuracy on X_l greater than threshold
                        pass_acc_threshold = np.count_nonzero(np.array(accs) >= acc_threshold)
                        if pass_acc_threshold >= 8:
                        # if np.average(accs) >= acc_threshold:
                            new_config = [lr, n_batches, n_epochs]
                            if not is_bad_config(bad_configs, new_config):
                                good_configs.append(new_config)
                                break
                    if len(good_configs) > 0:
                        break
                if len(good_configs) > 0:
                    break
            # end searching if we find at least one good config
            if len(good_configs) > 0:
                print("Good config found: " + str(good_configs[0]))
                break
            acc_threshold -= 0.05

        # pick one good config and return
        return good_configs[0]

    # Compute the minimum number of data labeled in X_u by newly generated hfs
    def compute_n_labeled(self, new_lfs: list, gamma=0.05):
        n_labeled_list = []
        for lf in new_lfs:
            marginal_u = lf.prob_matrix(self.X_u.astype('float32'))
            marginal_u_best = np.amax(marginal_u, axis=1)
            n_labeled = np.count_nonzero(marginal_u_best - 1 / self.num_classes >= gamma)
            n_labeled_list.append(n_labeled)
        print("n_labeled computed: " + str(n_labeled_list))
        n_labeled_list = np.array(n_labeled_list)
        # find the non-zero labelers, drop the zero labelers
        if np.sum(n_labeled_list) == 0:
            return 0, None
        else:
            nonzero_idx = np.where(n_labeled_list != 0)[0]
            n_labeled_list_nonzero = n_labeled_list[nonzero_idx]
            return np.amin(n_labeled_list_nonzero), nonzero_idx

    # Check theoretical exit condition
    def check_theoretical_exit(self, prev_lfs: list, new_lfs: list, n_labeled, e_gamma=0.25, e_delta=0.2):
        M = len(prev_lfs) + len(new_lfs)
        log = np.log((2 * (M ** 2)) / e_delta)
        epsilon = e_gamma - np.sqrt((1 / (2 * n_labeled)) * log)

        # compute empirical accuracy
        empirical_accs = []
        for lf in new_lfs:
            prob_l = lf.prob_matrix(self.X_l.astype('float32'))
            y_l_prime = np.argmax(prob_l, axis=1)
            num_correct = np.sum(self.y_l == y_l_prime)
            empirical_acc = float(num_correct) / self.y_l.shape[0]
            empirical_accs.append(empirical_acc)
        max_empirical_acc = np.amax(empirical_accs)
        print("max empirical acc = " + str(max_empirical_acc))

        # compute learned accuracy from training a Snorkel generative model
        L_u_temp, L_l_temp = self.generate_label_matrices(prev_lfs + new_lfs)
        L_temp = np.concatenate((L_l_temp, L_u_temp), axis=0)
        snorkel_model_temp = LabelModel(cardinality=self.num_classes, verbose=False)
        snorkel_model_temp.fit(L_temp)
        learned_acc = snorkel_model_temp.score(L=L_l_temp, Y=self.y_l)['accuracy']
        print("learned acc = " + str(learned_acc))

        return np.abs(max_empirical_acc - learned_acc) > epsilon, max_empirical_acc, learned_acc

    # Call Snuba to generate weak labelers; this will be the major computational overhead
    # Compute theoretical exit condition and decide exit
    def generate_snuba_lfs(self, log_file):

        with open(log_file, 'a') as f:

            # Search for the best config of training hyperparams
            config = self.generate_hyperparams()
            f.write("Initial config: " + str(config[0]) + ", " + str(config[1]) + ", " + str(config[2]) + "\n")
            bad_configs = []  # prevent reusing of a bad config
            idx = None
            hg = HeuristicGenerator(self.X_u, self.X_l, self.y_l, self.y_u, b=1 / self.num_classes, cnn=True,
                                    num_cnn_labelers=self.num_guesses, task=self.task,
                                    lr=config[0], n_batches=config[1], n_epochs=config[2], n_classes=self.num_classes,
                                    bootstrap_size_per_class=self.bootstrap_size_per_class)

            # Inner loop of hg
            while len(hg.hf) < self.max_labelers:
                f.write("Snuba iteration =" + str(len(hg.hf)) + "\n")
                print("Snuba iteration = " + str(len(hg.hf)))

                first_iteration = len(hg.hf) == 0

                if first_iteration:  # first iteration, generate multiple labelers
                    hg.run_synthesizer(max_cardinality=1, idx=idx, keep=self.keep, model='cnn')
                else:  # following iterations, 1 at each time
                    hg.run_synthesizer(max_cardinality=1, idx=idx, keep=1, model='cnn')
                hg.run_verifier()

                # If not exit, check theoretical exit condition
                if first_iteration:
                    new_lf = hg.hf[-self.keep:]  # newly generated labelers
                    old_lf = hg.hf[0: len(hg.hf) - self.keep]  # previously generated labelers
                else:
                    new_lf = [hg.hf[-1]]
                    old_lf = hg.hf[0: -1]

                # Evaluation begins here
                hg.find_feedback()
                new_idx = hg.feedback_idx  # for confidence exit condition

                # Exit when very few idx remains unconfident, adjust this number for different task sequences
                if len(new_idx) <= 10:
                    f.write("Exit: new_idx is small\n")
                    break

                n_labeled, nonzero_idx = self.compute_n_labeled(new_lfs=new_lf)  # for theoretical exit condition
                f.write("n_labeled =" + str(n_labeled) + "\n")

                # Theoretical exit case 1: n_labeled == 0
                if n_labeled == 0:  # no more data points labeled
                    hg.hf = old_lf
                    bad_configs.append(config)
                    print("Updating bad configs to " + str(bad_configs))
                    f.write("Theoretical case 1: n_labeled == 0\n")
                    print("n_labeled == 0")
                    if len(old_lf) >= self.min_labelers:
                        f.write("Exit: sufficient labelers generated\n")
                        break
                    else:  # need to label more data points, keep generating
                        config = self.generate_hyperparams(idx=idx, bad_configs=bad_configs)
                        hg.update_training_configs(lr=config[0], n_batches=config[1], n_epochs=config[2])
                        f.write("Updated config: " + str(config[0]) + ", " + str(config[1]) + ", " + str(config[2]) + "\n")
                        continue

                new_lf = np.array(new_lf)[nonzero_idx].tolist()
                theoretical_exit, max_empirical_acc, learned_acc = \
                    self.check_theoretical_exit(prev_lfs=old_lf, new_lfs=new_lf, n_labeled=n_labeled)
                f.write("max empirical acc: " + str(max_empirical_acc) + "\n")
                f.write("learned acc: " + str(learned_acc) + "\n")

                # Theoretical exit case 2: n_labeled != 0 but check_theoretical_exit returns true
                if theoretical_exit:
                    hg.hf = old_lf
                    bad_configs.append(config)
                    print("Updating bad configs to " + str(bad_configs))
                    f.write("Theoretical case 2: checking returns true\n")
                    print("theoretical exit is true")
                    if len(old_lf) >= self.min_labelers:
                        f.write("Exit: sufficient labelers generated\n")
                        break
                    else:  # need to label more data points, keep generating
                        config = self.generate_hyperparams(idx=idx, bad_configs=bad_configs)
                        hg.update_training_configs(lr=config[0], n_batches=config[1], n_epochs=config[2])
                        f.write("Updated config: " + str(config[0]) + ", " + str(config[1]) + ", " + str(config[2]))
                        continue

                # Not meeting theoretical exit condition, continue with new training config
                else:
                    hg.hf = old_lf + new_lf
                    idx = new_idx
                    print("Updated: IDX:")
                    print(idx)
                    print(len(idx))
                    continue

            f.write("Num labelers = " + str(len(hg.hf)) + "\n\n")

        return hg.hf

    # Generate label matrices from Snuba weak labelers for Snorkel
    def generate_label_matrices(self, lfs):
        L_l = []
        L_u = []
        for lf in lfs:
            prob_u = lf.prob_matrix(self.X_u.astype('float32'))
            y_u_hat = np.argmax(prob_u, axis=1)
            L_u.append(y_u_hat)
            prob_l = lf.prob_matrix(self.X_l.astype('float32'))
            y_l_hat = np.argmax(prob_l, axis=1)
            L_l.append(y_l_hat)
        # pad with the last LF if num of LFs < 3, for Snorkel input
        if len(L_u) < 3:
            pad = 3 - len(L_u)
            prob = lfs[-1].prob_matrix(self.X_u.astype('float32'))
            y_u_hat = np.argmax(prob, axis=1)
            L_u = L_u + [y_u_hat] * pad
        if len(L_l) < 3:
            pad = 3 - len(L_l)
            prob = lfs[-1].prob_matrix(self.X_l.astype('float32'))
            y_l_hat = np.argmax(prob, axis=1)
            L_l = L_l + [y_l_hat] * pad
        return np.array(L_u).T, np.array(L_l).T

    # # main function to be called
    # def generate_labels(self, log_file):
    #
    #     # Step 1: generate label matrices from Snuba, track number of labelers generated
    #     lfs = self.generate_snuba_lfs(log_file)
    #     L_u, L_l = self.generate_label_matrices(lfs)
    #     L = np.concatenate((L_l, L_u), axis=0)
    #
    #     # Step 2: ensemble label matrices by training a Snorkel LabelModel
    #     # Here, need at least 3 weak labeling functions, so there is a padding in generate_label_matrices
    #     snorkel_model = LabelModel(cardinality=self.num_classes, verbose=False)
    #     snorkel_model.fit(L)
    #
    #     # Step 3: compute logits
    #     y_snorkel_u, logit_u = snorkel_model.predict(L_u, return_probs=True)
    #     y_snorkel_l, logit_l = snorkel_model.predict(L_l, return_probs=True)
    #
    #     # Step 4: temperature scaling to produce final y_u_prime
    #     logit_u_calibrated = tstorch_calibrate(val_logits=logit_l, val_ys=self.y_l.astype('int64'), logits=logit_u)
    #     y_u_prime = np.argmax(logit_u_calibrated, axis=1)
    #
    #     return logit_u_calibrated, y_u_prime, logit_l, y_snorkel_l
