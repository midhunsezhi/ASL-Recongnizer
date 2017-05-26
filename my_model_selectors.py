import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict,
                 this_word: str, n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        this_word = self.this_word
        sequences = self.sequences

        if len(self.lengths) < 2:
            if self.verbose:
                print("not enough sequences to train this word = {}"
                      .format(this_word))
            return None

        hmm_models = {}

        best_score_overall = None
        best_i = None
        fold = 0

        n_fold_model_score = []
        for i in range(self.min_n_components, self.max_n_components):

            sum_logL = 0
            best_score = None
            best_model = None
            num_of_models = 0

            fold = 0
            kf = KFold(n_splits=min(3, len(self.lengths)))

            for cv_train_idx, cv_test_idx in kf.split(sequences):

                fold += 1

                X, lengths = combine_sequences(cv_train_idx, sequences)
                X2, lengths2 = combine_sequences(cv_test_idx, sequences)

                try:
                    model = GaussianHMM(n_components=i, covariance_type="diag",
                                        n_iter=1000,
                                        random_state=self.random_state,
                                        verbose=False).fit(X, lengths)

                    logL = model.score(X2, lengths2)
                    sum_logL += logL
                    num_of_models += 1
                    if best_score is None or best_score < logL:
                        best_score = logL
                        best_model = model

                    if self.verbose:
                        print("model created : word = {}, number of states = {},  fold = {}, \
                              score = {}".format(this_word, i, fold, logL))
                except ValueError:
                    if self.verbose:
                        print("model failed to create : word = {}, \
                               number of states = {}, score = {}"
                              .format(this_word, i, fold))

            if num_of_models == 0:
                if self.verbose:
                    print("no models generated for {} with {} states"
                          .format(this_word, i))
                continue

            hmm_models[i] = best_model
            # now average the scores for all the folds of this n
            avg_logL = sum_logL / num_of_models
            if self.verbose:
                print("best score for word = {} with num of states = {}, \
                      score = {}".format(this_word, i, best_score))

                print("average score for word = {} with num of states = {}, \
                      average score = {}".format(this_word, i, avg_logL))

            if best_score_overall is None or best_score_overall < avg_logL:
                best_score_overall = avg_logL
                best_i = i

        if self.verbose:
            print("best model for {}, number of states = {}"
                  .format(this_word, best_i))

        if best_i is None:
            return None
        return hmm_models[best_i]
