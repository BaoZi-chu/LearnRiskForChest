from data_process import similarity_based_feature as sbf
from common import config
import numpy as np
from . import tf_learn_weights as tflearn


cfg = config.Configuration(config.global_selection)


class RiskModel(object):
    def __init__(self):
        self.prob_interval_boundary_pts = sbf.get_equal_intervals(0.0, 1.0, cfg.interval_number_4_continuous_value)
        self.prob_dist_mean = None
        self.prob_dist_variance = None
        # parameters for risk model
        self.learn_weights = None
        self.learn_confidence = None
        self.learn_variances = None
        self.match_value = None
        self.unmatch_value = None
        self.func_params = None
        # train data
        self.train_data = None
        # validation data
        self.validation_data = None
        # test data
        self.test_data = None

    def train(self, train_machine_probs, valida_machine_probs):
        # use new classifier output probabilities.
        self.train_data.update_machine_info(train_machine_probs)
        self.validation_data.update_machine_info(valida_machine_probs)
        # update the distributions of probability features.
        prob_interval_2_ids = sbf.get_continuous_interval_to_ids(self.train_data.id_2_probs,
                                                                 [0],
                                                                 self.train_data.data_ids,
                                                                 self.prob_interval_boundary_pts)
        self.prob_dist_mean, self.prob_dist_variance = sbf.calculate_similarity_interval_distributions(
            prob_interval_2_ids,
            self.train_data.id_2_true_labels,
            cfg.minimum_observation_num)
        # update the probability feature of training data
        self.train_data.update_probability_feature(self.prob_interval_boundary_pts,
                                                   self.prob_dist_mean,
                                                   self.prob_dist_variance)
        # update the probability feature of validation data
        self.validation_data.update_probability_feature(self.prob_interval_boundary_pts,
                                                        self.prob_dist_mean,
                                                        self.prob_dist_variance)
        # -- sample mean --
        init_mu = np.concatenate((self.train_data.mu_vector,
                                  self.prob_dist_mean[0].reshape([1, -1])), axis=1).reshape([-1, 1])
        init_mu[np.where(init_mu == -1)] = 0.0

        # -- Learning feature weights. --
        # Note on 2019-03-31: the initial variance is abandoned, i.e., input is None.
        # Since we use relative standard deviation.
        parameters = tflearn.fit(self.validation_data.machine_labels.reshape([self.validation_data.data_len, 1]),
                                 self.validation_data.get_mean_x().tolil(),
                                 self.validation_data.get_variance_x().tolil(),
                                 self.validation_data.risk_labels.reshape([self.validation_data.data_len, 1]),
                                 self.validation_data.get_activation_matrix().tolil(),
                                 init_mu,
                                 init_variance=None)
        self.learn_weights = parameters[0]
        self.learn_confidence = parameters[1]
        self.learn_variances = parameters[2]
        self.match_value = parameters[3]
        self.unmatch_value = parameters[4]
        self.func_params = parameters[5]

    def predict(self, test_machine_probs):
        # use new classifier output probabilities.
        self.test_data.update_machine_info(test_machine_probs)
        # update the probability feature of training data
        self.test_data.update_probability_feature(self.prob_interval_boundary_pts,
                                                  self.prob_dist_mean,
                                                  self.prob_dist_variance)
        # -- Apply learned risk model to workload. --
        results = tflearn.predict(self.test_data.machine_labels.reshape([self.test_data.data_len, 1]),
                                  self.test_data.get_mean_x().tolil(),
                                  self.test_data.get_variance_x().tolil(),
                                  self.test_data.get_activation_matrix().tolil(),
                                  self.learn_weights,
                                  self.learn_confidence,
                                  self.learn_variances,
                                  self.match_value,
                                  self.unmatch_value,
                                  self.func_params)
        predict_probs = results[0].reshape(-1)
        pair_mus = results[1].reshape(-1)
        pair_sigmas = results[2].reshape(-1)
        self.test_data.risk_values = predict_probs
        self.test_data.pair_mus = pair_mus
        self.test_data.pair_sigmas = pair_sigmas

