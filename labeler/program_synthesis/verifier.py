import numpy as np
from scipy import sparse
from labeler.program_synthesis.label_aggregator import LabelAggregator

def odds_to_prob(l):
  """
  This is the inverse logit function logit^{-1}:
    l       = \log\frac{p}{1-p}
    \exp(l) = \frac{p}{1-p}
    p       = \frac{\exp(l)}{1 + \exp(l)}
  """
  return np.exp(l) / (1.0 + np.exp(l))

# No other difference except: 1. We use Snorkel for generative model since LabelAggregator does not support multiclass
# 2. b is 1/n_classes
class Verifier(object):
    """
    A class for the Snorkel Model Verifier
    """

    def __init__(self, L_train, L_val, val_ground, has_snorkel=True, n_classes=2):
        self.L_train = L_train.astype(int)
        self.L_val = L_val.astype(int)
        self.val_ground = val_ground

        self.has_snorkel = has_snorkel
        self.n_classes = n_classes

    def train_gen_model(self,deps=False,grid_search=False):
        """ 
        Calls appropriate generative model
        """
        if self.has_snorkel:
            #TODO: GridSearch
            from snorkel.labeling.model.label_model import LabelModel
            gen_model = LabelModel(cardinality=self.n_classes, verbose=False)
            gen_model.fit(self.L_train)
        else:
            gen_model = LabelAggregator()
            gen_model.train(self.L_train, rate=1e-3, mu=1e-6, verbose=False)
        self.gen_model = gen_model

    def assign_marginals(self):
        """ 
        Assigns probabilistic labels for train and val sets 
        """
        if not self.has_snorkel:
            self.train_marginals = self.gen_model.marginals(sparse.csr_matrix(self.L_train))
            self.val_marginals = self.gen_model.marginals(sparse.csr_matrix(self.L_val))
        else:
            y_train_prime, logit_train = self.gen_model.predict(self.L_train, return_probs=True)
            y_val_prime, logit_val = self.gen_model.predict(self.L_val, return_probs=True)
            self.train_marginals = logit_train
            self.val_marginals = logit_val
        # if gen_model doesn't converge, everything is abstain
        # self.train_marginals = np.nan_to_num(self.train_marginals, nan=0.5)
        # self.val_marginals = np.nan_to_num(self.val_marginals, nan=0.5)
        #print 'Learned Accuracies: ', odds_to_prob(self.gen_model.w)

    def find_vague_points(self, gamma=0.1, b=0.5):
        """ 
        Find val set indices where marginals are within thresh of b 
        """
        # val_idx = np.where(np.abs(self.val_marginals-b) <= gamma)
        val_idx = np.where(np.abs(np.max(self.val_marginals, axis=1) - b) <= gamma)
        return val_idx[0]

    def find_weighted_vague_points(self, gamma=None, b=None):
        """
        Return weights for the val set according to their confidence
        Confidences are in range [0,0.5], then normalized to sum to 1
        """
        weights = -np.abs(self.val_marginals - 0.5) + 0.5
        normalized_weights = weights / np.sum(weights)
        return len(normalized_weights) * normalized_weights

    def find_incorrect_points(self, b=0.5):
        """ Find val set indices where marginals are incorrect """
        val_labels = 2*(self.val_marginals > b)-1
        val_idx = np.where(val_labels != self.val_ground)
        return val_idx[0]