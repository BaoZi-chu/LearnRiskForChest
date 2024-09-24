from __future__ import print_function, with_statement, division, absolute_import

import tensorflow as tf
# from tensorflow.distributions import Normal
import tensorflow_probability as tfp
from common import config
import numpy as np
from scipy import sparse as sp
from collections import Counter
import math
import logging
from tqdm import trange
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Deploying CPU setting.

tfd = tfp.distributions

cfg = config.Configuration(config.global_selection)

LEARN_VARIANCE = cfg.learn_variance
APPLY_WEIGHT_FUNC = cfg.apply_function_to_weight_classifier_output

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(module)s:%(levelname)s] - %(message)s")


def my_truncated_normal_ppf(confidence, a, b, mean, stddev):
    tf_norm = tfd.Normal(mean, stddev)
    _nb = tf_norm.cdf(b)
    _na = tf_norm.cdf(a)
    _sb = tf_norm.survival_function(b)
    _sa = tf_norm.survival_function(a)

    return tf.where(a > 0,
                    -tf_norm.quantile(confidence * _sb + _sa * (1.0 - confidence)),
                    tf_norm.quantile(confidence * _nb + _na * (1.0 - confidence)))


def gaussian_function(a, b, c, x):
    _part = - tf.math.divide(tf.square(x - b), tf.multiply(tf.constant(2.0, dtype=tf.double), tf.square(c)))
    # _f = tf.multiply(a, tf.exp(_part))
    _f = - tf.exp(_part) + a + 1.0
    return _f


def fit(machine_results, _mus_X, _sigmas_X, _y, _feature_activation_matrix, init_mu, init_variance=None):
    """
    All input parameters are numpy array type.
    :param machine_results: shape: (n, 1), the machine results.
    :param _mus_X: shape: (n, m), the means of m risk features. !!! Scipy Sparse Matrix (lil_matrix) !!!
    :param _sigmas_X: shape: (n, m), the variances of m risk features. !!! Scipy Sparse Matrix (lil_matrix) !!!
    :param _y: shape: (n, 1), the risk labels of n training data.
    :param _feature_activation_matrix: shape (n, m), indicate which features are used in each data.
                                        !!! Scipy Sparse Matrix (lil_matrix) !!!
    :param init_mu: shape (m, 1)
    :param init_variance: shape (m, 1)
    :return:
    """

    tf.reset_default_graph()
    m = _mus_X.shape[1]

    if not APPLY_WEIGHT_FUNC:
        # -- feature weights --
        # continuous_m = config.get_interval_number_4_continuous_value()  # number of continuous features
        continuous_m = 0
        if continuous_m > 0:
            print("- One weight for multiple intervals.")
            one_conti_feature = 1
        else:
            print("- Different weights for different intervals.")
            one_conti_feature = 0
        discrete_m = m - continuous_m  # number of discrete features
    else:
        print("- Different weights for different probabilities (use Gaussian function).")
        continuous_m = cfg.interval_number_4_continuous_value
        discrete_m = m - continuous_m
        one_conti_feature = 0

    # -- feature variances --
    # continuous_m_var = config.get_interval_number_4_continuous_value()
    continuous_m_var = 0
    if continuous_m_var > 0:
        print("- One variance for multiple intervals.")
        one_conti_feature_var = 1
    else:
        print("- Different variances for different intervals.")
        one_conti_feature_var = 0
    discrete_m_var = m - continuous_m_var

    data_len = len(_y)  # number of data
    np.random.seed(2019)
    tf.set_random_seed(2019)
    # The mean number of activated rule features.
    rule_mean_number = np.mean(np.array(np.sum(_feature_activation_matrix, axis=1))) - 1.0
    max_w = 1.0 / np.max(np.sum(_feature_activation_matrix, axis=1))
    class_count = Counter(_y.reshape(-1))
    # print("- Max number of features: {}, 1/max={}.".format(1.0/max_w, max_w))
    # -- Set class weight w.r.t class size. --
    # risky_weight = 1.0 * class_count.get(0) / class_count.get(1)
    # -- or set to 1.0 --
    # risky_weight = 1.0
    # print("- Set risky label weight = {}. [{}]".format(risky_weight, class_count))
    learning_rate = 0.001
    print("- Set learning rate = {}".format(learning_rate))

    with tf.name_scope('constants'):
        alpha = tf.constant(cfg.risk_confidence, dtype=tf.double)
        a = tf.constant(0.0, dtype=tf.double)
        b = tf.constant(1.0, dtype=tf.double)
        l1 = tf.constant(0.001, dtype=tf.double)
        l2 = tf.constant(0.001, dtype=tf.double)
        label_match_value = tf.constant(1.0, dtype=tf.double)
        label_unmatch_value = tf.constant(0.0, dtype=tf.double)
        # print("- Manually set parameters of Gaussian function.")
        # weight_func_a = tf.constant(0.5, dtype=tf.double)
        weight_func_b = tf.constant(0.5, dtype=tf.double)
        # weight_func_c = tf.constant(1.0, dtype=tf.double)
        variance_initializer = None
        if init_variance is not None:
            variance_initializer = np.array(init_variance).reshape([-1, 1])[:discrete_m_var + one_conti_feature_var, 0]
            variance_initializer = tf.constant(variance_initializer,
                                               dtype=tf.double,
                                               shape=[discrete_m_var + one_conti_feature_var, 1])

    with tf.name_scope('inputs'):
        # Variables (Vectors)
        machine_label = tf.placeholder(tf.int8, name='ML')  # (n, 1)
        risk_y = tf.placeholder(tf.int64, name='y')  # (n, 1)
        mus = tf.placeholder(tf.double, name='mu')  # (n, m)
        sigmas = tf.placeholder(tf.double, name='sigma')  # (n, m)
        feature_matrix = tf.placeholder(tf.double, name='featureMatrix')  # (n, m)
        # parameters for learning to rank
        pairwise_risky_values = tf.placeholder(tf.double, name='pairwise_values')
        pairwise_risky_labels = tf.placeholder(tf.double, name='pairwise_labels')

        # Discrete feature weights and one probability-based feature weight.
        discrete_w = tf.get_variable(name='discrete_w', shape=[discrete_m + one_conti_feature, 1],
                                     dtype=tf.double,
                                     initializer=tf.random_uniform_initializer(0., max_w),
                                     regularizer=tf.contrib.layers.l1_l2_regularizer(l1, l2),
                                     constraint=lambda t: tf.abs(t)
                                     )  # (m ,1)

        # Note: for continuous features, different intervals have their own variances.
        if variance_initializer is None:
            # This variable is treated as relative standard deviation (RSD), rsd = sd / mu * 100%.
            # For simplicity, we do not change its name here.
            discrete_variances = tf.get_variable(name='discrete_variances',
                                                 shape=[discrete_m_var + one_conti_feature_var, 1],
                                                 dtype=tf.double,
                                                 initializer=tf.random_uniform_initializer(0.1, 0.9),
                                                 constraint=lambda t: tf.clip_by_value(t, 0.1, 0.9)
                                                 )
        else:
            pass
            # discrete_variances = tf.get_variable(name='discrete_variances',
            #                                      dtype=tf.double,
            #                                      initializer=variance_initializer,
            #                                      # constraint=lambda t: tf.abs(t),
            #                                      constraint=lambda t: tf.clip_by_value(t, 0., 0.5)
            #                                      )

        learn2rank_sigma = tf.get_variable(name='learning_to_rank_sigma',
                                           initializer=tf.constant(1.0, dtype=tf.double),
                                           regularizer=tf.contrib.layers.l1_l2_regularizer(l1, l2))

        weight_func_a = tf.get_variable(name='weight_function_w',
                                        initializer=tf.constant(1.0, dtype=tf.double),
                                        regularizer=tf.contrib.layers.l1_l2_regularizer(l1, l2),
                                        constraint=lambda t: tf.abs(t))

        weight_func_c = tf.get_variable(name='weight_function_variance',
                                        initializer=tf.constant(0.5, dtype=tf.double),
                                        regularizer=tf.contrib.layers.l1_l2_regularizer(l1, l2),
                                        constraint=lambda t: tf.abs(t))

    with tf.name_scope('prediction'):
        # Handling Feature Weights.
        if one_conti_feature == 1:
            # 1.1 Continuous feature weights.
            contin_w = tf.convert_to_tensor([discrete_w[discrete_m]] * (continuous_m - 1))
            # 1.2 Concat both weights.
            w = tf.concat([discrete_w, contin_w], axis=0)
        else:
            w = discrete_w

        # Handling Feature Variances.
        if one_conti_feature_var == 1:
            # 2.1 Continuous feature variances.
            contin_variances = tf.convert_to_tensor([discrete_variances[discrete_m]] * (continuous_m_var - 1))
            # 2.2 Concat both weiths.
            variances = tf.concat([discrete_variances, contin_variances], axis=0)
        else:
            variances = discrete_variances

        # In the newest solution, the above 'variances' is actually Relative Standard Deviation.
        # Here we transform it to real variances.
        new_init_mu = init_mu.copy()
        # If mu is 0.0, then the variance will be 0.0 in any case. So we set the 0.0 to 0.1 for non-zero variances.
        new_init_mu[np.where(new_init_mu == 0.0)] = 0.1
        standard_deviation = variances * new_init_mu
        variances = tf.square(standard_deviation)

        if not APPLY_WEIGHT_FUNC:
            #  Note: In practice, big_mu and big_sigma can be zero, and when calculates gradients,
            #        the f(big_mu, big_sigma) can be zero and as the denominator at the same time.
            #        So here we add a small number 1e-10.
            big_mu = tf.matmul(mus, w) + 1e-10  # (n, m) * (m, 1) -> (n, 1)

            if not LEARN_VARIANCE:
                # -- ** 1. use pre-set variances. ** --
                print("- No learning variances.")
                big_sigma = tf.matmul(sigmas, tf.square(w)) + 1e-10  # (n, m) * (m, 1) -> (n, 1)
            else:
                # -- ** 2. learn variances. ** --
                print("- Learning variances.")
                learn_sigmas_X = feature_matrix * tf.reshape(variances, [1, m])
                big_sigma = tf.matmul(learn_sigmas_X, tf.square(w)) + 1e-10

            # Normalize the weights of features in each pair.
            weight_vector = tf.matmul(feature_matrix, w) + 1e-10  # (n, m) * (m, 1) -> (n, 1)
            big_mu = big_mu / weight_vector
            big_sigma = big_sigma / (tf.square(weight_vector))
        else:
            # Need to calculate weights for each probability.
            rule_mus = tf.slice(mus, [0, 0], [-1, discrete_m])
            # sparse matrix, only one position has value.
            machine_mus = tf.slice(mus, [0, discrete_m], [-1, continuous_m])
            machine_mus_vector = tf.reshape(tf.reduce_sum(machine_mus, axis=1), [-1, 1])
            machine_weights = gaussian_function(weight_func_a, weight_func_b, weight_func_c, machine_mus_vector)
            machine_weights = tf.reshape(machine_weights, [-1, 1])

            big_mu = tf.matmul(rule_mus, discrete_w) + tf.multiply(machine_mus_vector, machine_weights) + 1e-10

            if not LEARN_VARIANCE:
                # -- ** 1. use pre-set variances. ** --
                print("- No learning variances.")
                use_sigmas = sigmas
            else:
                # -- ** 2. learn variances. ** --
                print("- Learning variances.")
                learn_sigmas_X = feature_matrix * tf.reshape(variances, [1, m])
                use_sigmas = learn_sigmas_X
            rule_sigmas = tf.slice(use_sigmas, [0, 0], [-1, discrete_m])
            machine_sigmas = tf.slice(use_sigmas, [0, discrete_m], [-1, continuous_m])
            machine_sigmas_vector = tf.reshape(tf.reduce_sum(machine_sigmas, axis=1), [-1, 1])

            big_sigma = tf.matmul(rule_sigmas, tf.square(discrete_w)) + tf.multiply(machine_sigmas_vector,
                                                                                    tf.square(machine_weights)) + 1e-10

            # Normalize the weights of features in each pair.
            rule_activate_matrix = tf.slice(feature_matrix, [0, 0], [-1, discrete_m])
            weight_vector = tf.matmul(rule_activate_matrix, discrete_w) + machine_weights + 1e-10
            big_mu = big_mu / weight_vector
            big_sigma = big_sigma / (tf.square(weight_vector))

        # Truncated normal distribution.
        Fr_alpha = my_truncated_normal_ppf(alpha, a, b, big_mu, tf.sqrt(big_sigma))
        Fr_alpha_bar = my_truncated_normal_ppf(1 - alpha, a, b, big_mu, tf.sqrt(big_sigma))

        machine_label = tf.cast(machine_label, tf.double)

        prob = Fr_alpha * (tf.ones_like(machine_label) - machine_label) + (
                tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * machine_label

        prob_2_label = tf.concat([prob, tf.cast(risk_y, tf.double)], axis=1)
        descent_order = tf.gather(prob_2_label, tf.nn.top_k(-prob_2_label[:, 0], k=tf.shape(risk_y)[0]).indices)
        order_risky_label = tf.slice(descent_order, [0, 1], [-1, 1])
        risky_index = tf.where(tf.equal(order_risky_label, 1))
        risky_row_index = tf.slice(risky_index, [0, 0], [-1, 1]) + 1

        acc = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.math.greater(prob, 0.5), tf.int64), risk_y), tf.int32))

        # Average Precision
        risky_count = tf.reshape(tf.where(tf.equal(risky_row_index, risky_row_index))[:, 0], [-1, 1]) + 1
        cut_off_k_precisions = tf.math.divide(tf.cast(risky_count, tf.double), tf.cast(risky_row_index, tf.double))
        avg_precision = tf.reduce_mean(cut_off_k_precisions)

    with tf.name_scope('loss'):
        # -- Pairwise loss. --
        '''
        * cross entropy cost function.
        loss_function = - p'_ij * log(p_ij) - (1 - p'_ij) * log(1 - p_ij)
        where,
        p'_ij = 0.5 * (1 + higher_rank(xi, xj));
        p_ij = e^(o_ij) / (1 + e^(o_ij)), o_ij = f(xi) - f(xj);
        higher_rank(xi, xj) = xi_risky_label - xj_risky_label;
        Concise Form:
        loss_function = - p'_ij * o_ij + log(1 + e^(o_ij))
        Ref: Burges C, Shaked T, Renshaw E, et al. Learning to rank using gradient descent[C].
        Proceedings of the 22nd International Conference on Machine learning (ICML-05). 2005: 89-96.
        '''

        def get_pairwise_combinations(input_data, out_result):
            start_index = tf.constant(0)

            def while_condition(_i, *args):
                return tf.less(_i, tf.shape(input_data)[0] - 1)

            def body(_i, data, result):
                # do something here which you want to do in your loop
                # increment i
                result = tf.concat(
                    [result, tf.reshape(tf.stack(tf.meshgrid(data[_i], data[_i + 1:]), axis=-1), [-1, 2])], axis=0)
                return [tf.add(_i, 1), data, result]

            # do the loop:
            r = tf.while_loop(while_condition, body, [start_index, input_data, out_result])[2][1:]
            return r
        pairwise_probs = get_pairwise_combinations(prob, pairwise_risky_values)
        pairwise_labels = get_pairwise_combinations(tf.cast(risk_y, tf.double), pairwise_risky_labels)
        p_target_ij = 0.5 * (1.0 + pairwise_labels[:, 0] - pairwise_labels[:, 1])
        o_ij = pairwise_probs[:, 0] - pairwise_probs[:, 1]
        # -- Remove pairs that have same labels directly.
        diff_label_indices = tf.where(tf.not_equal(p_target_ij, 0.5))  # indices of pairs have different labels.
        new_p_target_ij = tf.gather(p_target_ij, diff_label_indices)
        new_o_ij = tf.gather(o_ij, diff_label_indices) * learn2rank_sigma
        cost = tf.reduce_sum(
            - new_p_target_ij * new_o_ij + tf.log(1.0 + tf.exp(new_o_ij))) + tf.losses.get_regularization_loss()

    with tf.name_scope('optimization'):
        global_step = tf.Variable(0, name="tr_global_step", trainable=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost,
                                                                                            global_step=global_step)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # mainloop
        ep = cfg.risk_training_epochs
        shuffle_data = True
        # bs = np.maximum(int(data_len * 0.05), 2)
        bs = 512
        batch_num = data_len // bs + (1 if data_len % bs else 0)
        print("\n- Set number of training epochs={}.".format(ep))
        print("- The train data batch size={}, batch number={}.".format(bs, batch_num))
        print("- Shuffle data each epoch: {}".format(shuffle_data))
        # if batch_num > 1:
        #     print("- The average precision is approximated! Cause there exists multiple batches and "
        #           "the rank positions of risky data are partially evaluated.")

        for epoch in trange(ep, desc="Train Risk Model", ascii=True):
            if shuffle_data:
                # reshuffle training data
                index = np.random.permutation(np.arange(data_len))
            else:
                index = np.arange(data_len)  # non shuffle version
            accuracy = 0
            average_precision = []
            loss = 0.

            # mini_batch training
            for i in range(batch_num):
                machine_results_batch = machine_results[index][bs * i:bs * i + bs]
                y_batch = _y[index][bs * i:bs * i + bs]
                mus_batch = _mus_X[index][bs * i:bs * i + bs]
                sigmas_batch = _sigmas_X[index][bs * i:bs * i + bs]
                activate_features_batch = _feature_activation_matrix[index][bs * i:bs * i + bs]
                return_values = sess.run(
                    [optimizer, global_step, prob, w, cost, acc, alpha, variances, label_match_value,
                     label_unmatch_value, weight_func_a, weight_func_b, weight_func_c, avg_precision, learn2rank_sigma],
                    feed_dict={machine_label: machine_results_batch,
                               risk_y: y_batch,
                               mus: mus_batch.todense(),
                               sigmas: sigmas_batch.todense(),
                               feature_matrix: activate_features_batch.todense(),
                               pairwise_risky_values: [[0., 0.]],  # This fist element will be removed.
                               pairwise_risky_labels: [[0., 0.]]})
                _ = return_values[0]
                step = return_values[1]
                prob_ = return_values[2]
                w_ = return_values[3]
                cost_ = return_values[4]
                acc_ = return_values[5]
                alpha_ = return_values[6]
                variances_ = return_values[7]
                label_match_value_ = return_values[8]
                label_unmatch_value_ = return_values[9]
                _func_a = return_values[10]
                _func_b = return_values[11]
                _func_c = return_values[12]
                _avg_precision = return_values[13]
                _l2rank_sigma = return_values[14]
                accuracy += acc_
                loss += cost_
                if not math.isnan(_avg_precision):
                    average_precision.append(_avg_precision)
        return w_, alpha_, variances_, label_match_value_, label_unmatch_value_, [_func_a, _func_b, _func_c]


def predict(machine_results, _mus_X, _sigmas_X, _feature_activation_matrix, _w, _alpha, _variances, _match_value,
            _unmatch_value, func_parameters, apply_learn_v=True):
    """
    All input parameters are numpy array type.
    :param machine_results: shape: (n, 1), the machine results.
    :param _mus_X: shape: (n, m), the means of m risk features. !!! Scipy Sparse Matrix (lil_matrix) !!!
    :param _sigmas_X: shape: (n, m), the variances of m risk features. !!! Scipy Sparse Matrix (lil_matrix) !!!
    :param _feature_activation_matrix: shape (n, m), indicate which features are used in each data.
                                        !!! Scipy Sparse Matrix (lil_matrix) !!!
    :param _w: shape: (m, 1), the learned feature weights.
    :param _alpha: the confidence for risk analysis.
    :param _variances: shape: (m, 1), the learned variances.
    :param _match_value:
    :param _unmatch_value:
    :param func_parameters: the parameters for gaussian function
    :param apply_learn_v: If False, use _sigma_X instead of _variances.
    :return:
    """
    tf.reset_default_graph()
    m = _mus_X.shape[1]  # number of features
    continuous_m = cfg.interval_number_4_continuous_value
    discrete_m = m - continuous_m

    with tf.name_scope('constants'):
        a = tf.constant(0.0, dtype=tf.double)
        b = tf.constant(1.0, dtype=tf.double)

    with tf.name_scope('inputs'):
        # Variables (Vectors)
        machine_label = tf.placeholder(tf.int8, name='ML')  # (n, 1)
        mus = tf.placeholder(tf.double, name='mu')  # (n, m)
        sigmas = tf.placeholder(tf.double, name='sigma')  # (n, m)
        w = tf.placeholder(tf.double, name='w')  # (m ,1)
        alpha = tf.placeholder(tf.double, name='alpha')
        feature_matrix = tf.placeholder(tf.double, name='featureMatrix')  # (n, m)
        variances = tf.placeholder(tf.double, name='variance')
        match_value = tf.placeholder(tf.double, name='match_value')
        unmatch_value = tf.placeholder(tf.double, name='unmatch_value')
        weight_func_a = tf.placeholder(tf.double, name='weight_function_w')
        weight_func_b = tf.placeholder(tf.double, name='weight_function_mean')
        weight_func_c = tf.placeholder(tf.double, name='weight_function_variance')

    with tf.name_scope('prediction'):
        if not APPLY_WEIGHT_FUNC or not apply_learn_v:
            big_mu = tf.matmul(mus, w) + 1e-10  # (n, m) * (m, 1) -> (n, 1)
            # -- 1. use pre-set variances. --
            if not apply_learn_v or not LEARN_VARIANCE:
                print("- Apply pre-set variances.")
                big_sigma = tf.matmul(sigmas, tf.square(w)) + 1e-10  # (n, m) * (m, 1) -> (n, 1)
            else:
                # -- 2. learn variances. --
                print("- Apply learned variances.")
                learn_sigmas_X = feature_matrix * tf.reshape(variances, [1, m])
                big_sigma = tf.matmul(learn_sigmas_X, tf.square(w)) + 1e-10

            # Normalize the weights of features in each pair.
            weight_vector = tf.matmul(feature_matrix, w)  # (n, m) * (m, 1) -> (n, 1)
            big_mu = big_mu / weight_vector
            big_sigma = big_sigma / (tf.square(weight_vector))
        else:
            # Need to calculate weights for each probability.
            rule_mus = tf.slice(mus, [0, 0], [-1, discrete_m])
            discrete_w = tf.slice(w, [0, 0], [discrete_m, 1])
            # sparse matrix, only one position has value.
            machine_mus = tf.slice(mus, [0, discrete_m], [-1, continuous_m])
            machine_mus_vector = tf.reshape(tf.reduce_sum(machine_mus, axis=1), [-1, 1])
            machine_weights = gaussian_function(weight_func_a, weight_func_b, weight_func_c, machine_mus_vector)
            machine_weights = tf.reshape(machine_weights, [-1, 1])

            big_mu = tf.matmul(rule_mus, discrete_w) + tf.multiply(machine_mus_vector, machine_weights) + 1e-10

            if not LEARN_VARIANCE:
                # -- ** 1. use pre-set variances. ** --
                print("- Apply pre-set variances.")
                use_sigmas = sigmas
            else:
                # -- ** 2. learn variances. ** --
                print("- Apply learned variances.")
                learn_sigmas_X = feature_matrix * tf.reshape(variances, [1, m])
                use_sigmas = learn_sigmas_X
            rule_sigmas = tf.slice(use_sigmas, [0, 0], [-1, discrete_m])
            machine_sigmas = tf.slice(use_sigmas, [0, discrete_m], [-1, continuous_m])
            machine_sigmas_vector = tf.reshape(tf.reduce_sum(machine_sigmas, axis=1), [-1, 1])

            big_sigma = tf.matmul(rule_sigmas, tf.square(discrete_w)) + tf.multiply(machine_sigmas_vector,
                                                                                    tf.square(machine_weights)) + 1e-10

            # Normalize the weights of features in each pair.
            rule_activate_matrix = tf.slice(feature_matrix, [0, 0], [-1, discrete_m])
            weight_vector = tf.matmul(rule_activate_matrix, discrete_w) + machine_weights + 1e-10
            big_mu = big_mu / weight_vector
            big_sigma = big_sigma / (tf.square(weight_vector))

        # Truncated normal distribution.
        Fr_alpha = my_truncated_normal_ppf(alpha, a, b, big_mu, tf.sqrt(big_sigma))
        Fr_alpha_bar = my_truncated_normal_ppf(1 - alpha, a, b, big_mu, tf.sqrt(big_sigma))

        machine_label = tf.cast(machine_label, tf.double)

        prob = Fr_alpha * (tf.ones_like(machine_label) - machine_label) + (
                tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * machine_label

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # To handle ResourceExhaustedError (see above for traceback):
        # OOM when allocating tensor with shape[24850,3836]..., use batch prediction.
        data_len = _mus_X.shape[0]
        # bs = np.maximum(int(data_len * 0.05), 2)
        bs = 1024
        batch_num = data_len // bs + (1 if data_len % bs else 0)

        print("- The test data batch size={}, batch number={}".format(bs, batch_num))

        prob1 = tf.constant([[0.]], dtype=tf.double)
        big_mu1 = tf.constant([[0.]], dtype=tf.double)
        big_sigma1 = tf.constant([[0.]], dtype=tf.double)

        # mini_batch
        for i in range(batch_num):
            machine_results_batch = machine_results[bs * i:bs * i + bs]
            mus_batch = _mus_X[bs * i:bs * i + bs]
            sigmas_batch = _sigmas_X[bs * i:bs * i + bs]
            activate_features_batch = _feature_activation_matrix[bs * i:bs * i + bs]
            prob_b, big_mu_b, big_sigma_b = sess.run([prob, big_mu, big_sigma],
                                                     feed_dict={machine_label: machine_results_batch,
                                                                mus: mus_batch.todense(),
                                                                w: _w,
                                                                alpha: _alpha,
                                                                feature_matrix: activate_features_batch.todense(),
                                                                sigmas: sigmas_batch.todense(),
                                                                variances: _variances,
                                                                match_value: _match_value,
                                                                unmatch_value: _unmatch_value,
                                                                weight_func_a: func_parameters[0],
                                                                weight_func_b: func_parameters[1],
                                                                weight_func_c: func_parameters[2]})
            prob1 = tf.concat([prob1, prob_b], 0)
            big_mu1 = tf.concat([big_mu1, big_mu_b], 0)
            big_sigma1 = tf.concat([big_sigma1, big_sigma_b], 0)
            prob1, big_mu1, big_sigma1 = sess.run([prob1, big_mu1, big_sigma1])
        # Remove the meaningless first one.
        prob1 = prob1[1:]
        big_mu1 = big_mu1[1:]
        big_sigma1 = big_sigma1[1:]
        return prob1, big_mu1, big_sigma1


def testf():
    m_results = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0]).reshape([-1, 1])
    train_X = np.array([[1, 1, 1, 1]] * 10)
    train_y = np.array([0, 1, 0, 1, 1, 1, 0, 0, 1, 1]).reshape([-1, 1])
    mus = np.array([0.2, 0.8, 0.4, 0.9])
    # sigmas = (np.array([0.1, 0.15, 0.12, 0.05]) ** 2)
    sigmas = np.array([0.1, 0.1, 0.1, 0.1])
    mus_X = train_X * mus
    sigmas_X = train_X * sigmas
    weights, alpha, variances, match_value, unmatch_value, func_params = fit(m_results, sp.lil_matrix(mus_X),
                                                                             sp.lil_matrix(sigmas_X), train_y,
                                                                             sp.lil_matrix(train_X),
                                                                             mus.reshape([-1, 1]),
                                                                             None)
    print(weights.shape)
    print("Discrete feature weights:", weights)
    print("Function parameters:", func_params)
    print("Match value: {}, Unmatch value: {}".format(match_value, unmatch_value))

    print("Initial variances: {}".format(sigmas))
    print("Learned variances: {}".format(variances))

    # weights, alpha, variances, match_value, unmatch_value, func_params = fit(m_results, sp.lil_matrix(mus_X),
    #                                                                          sp.lil_matrix(sigmas_X), train_y,
    #                                                                          sp.lil_matrix(train_X))

    test_X = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1]])
    test_m_results = np.array([0, 0, 1, 1, 0]).reshape([-1, 1])
    test_mu_X = test_X * mus
    test_sigmas_X = test_X * sigmas
    probs, pair_mus, pair_sigmas = predict(test_m_results, sp.lil_matrix(test_mu_X), sp.lil_matrix(test_sigmas_X),
                                           weights, alpha, sp.lil_matrix(test_X), variances, match_value, unmatch_value,
                                           func_params)
    print(probs)
    print(probs.reshape(-1))
    print(pair_mus.reshape(-1))
    print(pair_sigmas.reshape(-1))


if __name__ == '__main__':
    testf()
