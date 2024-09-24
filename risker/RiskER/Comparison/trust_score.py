import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from common import utils
import time
import socket
from common import config
from sklearn.metrics import confusion_matrix
from Comparison import trustscore
import numpy as np
from Comparison import dl_embed_features as dlf
import deepmatcher as dm
import collections

cfg = config.Configuration(config.global_selection)

APPLY_DEEP_LEARNING_REPRESENTATION = False

op_dir = curPath + '/results/' + cfg.get_parent_path().split('/')[-3] + '/' + cfg.get_parent_path().split('/')[
    -2] + '/'
deep_learning_data_path = cfg.get_parent_path().rstrip('/')  # Remove the last '/'.


if not os.path.exists(op_dir):
    os.mkdir(op_dir)


def get_dl_representation():
    if not os.path.exists(deep_learning_data_path):
        print("Warning: Data source is not found!")
        sys.exit()

    train, validation, test = dm.data.process(path=deep_learning_data_path,
                                              train='train.csv',
                                              validation='validation.csv',
                                              test='test.csv',
                                              ignore_columns=['left_id', 'right_id'],
                                              check_cached_data=False,
                                              auto_rebuild_cache=False,
                                              embeddings='glove.6B.300d',
                                              embeddings_cache_path='D:/vector_cache')
    model = dm.MatchingModel()
    model.initialize(train)
    model.load_state(deep_learning_data_path + '/hybrid_model.pth')
    train_results = dlf.get_attr_sim_representation(model, train)
    test_results = dlf.get_attr_sim_representation(model, test)
    dl_feature_info = collections.namedtuple('dl_feature_info',
                                             ['train_ids', 'train_features', 'test_ids', 'test_features'])
    return dl_feature_info(train_results[0], train_results[1], test_results[0], test_results[1])


def cal_trust_score(clf_data):
    id_2_pair_info = clf_data['pair_info']

    # training data
    train_ids = clf_data['train'].ids
    train_true_y = np.array(clf_data['train'].true_y)
    # test data
    test_ids = clf_data['test'].ids
    test_predict_y = np.array(clf_data['test'].predict_y)

    if APPLY_DEEP_LEARNING_REPRESENTATION:
        dl_feature_info = get_dl_representation()
        if np.array_equal(train_ids, dl_feature_info.train_ids):
            train_X = dl_feature_info.train_features
        else:
            print("ERROR: The input train data is not consistent with the existence one!")
        if np.array_equal(test_ids, dl_feature_info.test_ids):
            test_X = dl_feature_info.test_features
        else:
            print("ERROR: The input train data is not consistent with the existence one!")
    else:
        train_info = []
        for i in range(len(train_ids)):
            train_info.append(id_2_pair_info.get(train_ids[i]))
        train_X = np.array(train_info)[:, 2:]
        test_info = []
        for i in range(len(test_ids)):
            test_info.append(id_2_pair_info.get(test_ids[i]))
        test_X = np.array(test_info)[:, 2:]

    # Default
    trust_model = trustscore.TrustScore()

    # Filter out alpha (0 < alpha < 1) proportion of the training points with
    # lowest k-NN density when computing trust score.
    # trust_model = trustscore.TrustScore(k=10, alpha=0.1, filtering="density")

    # Filter out alpha (0 < alpha < 1) proportion of the training points with
    # highest label disagreement amongst its k-NN neighbors.
    # trust_model = trustscore.TrustScore(k=10, alpha=0.1, filtering="disagreement")

    trust_model.fit(train_X, train_true_y)

    trust_score = trust_model.get_score(test_X, test_predict_y)

    id_2_trust = dict()

    # To draw ROC curve, the scores need to be in descending order.
    for i in range(len(test_ids)):
        id_2_trust[test_ids[i]] = - trust_score[i]  # risk_score = - trust_score
    id_trust_scores = sorted(id_2_trust.items(), key=lambda item: item[1], reverse=True)

    # for i in range(len(test_ids)):
    #     id_2_trust[test_ids[i]] = trust_score[i]
    # id_trust_scores = sorted(id_2_trust.items(), key=lambda item: item[1], reverse=False)
    return id_trust_scores


def main():
    clf_data = utils.load_results(cfg)
    print('Apply Deep Learning Representation: {}'.format(APPLY_DEEP_LEARNING_REPRESENTATION))
    risk_score = cal_trust_score(clf_data)
    workload_ids = clf_data['test'].ids
    workload_predict_y = clf_data['test'].predict_y
    workload_y = clf_data['test'].true_y
    workload_probs = clf_data['test'].probs
    id_2_label_index = dict()
    for i in range(len(workload_ids)):
        id_2_label_index[workload_ids[i]] = i

    # -- Expectation based risk analysis. --
    id_2_expectation_based_risk = utils.expectation_based_risk(workload_ids,
                                                               workload_predict_y,
                                                               workload_probs)

    # -- Improve the machine results according to the risk analysis. --
    tn, fp, fn, tp = confusion_matrix(workload_y, workload_predict_y).ravel()
    improved_space = fp + fn
    budgets = cfg.get_budgets()
    budget_2_correction_num = dict()
    budget_2_quality = dict()
    budgets.append(improved_space)
    budgets = sorted(budgets)
    for budget in budgets:
        correction_num_list = []
        quality_list = []
        # -- Expectation based revision. --
        correction_num, eva_info = utils.revise_results(workload_y,
                                                        workload_predict_y,
                                                        budget,
                                                        id_2_expectation_based_risk,
                                                        id_2_label_index)
        correction_num_list.append(correction_num)
        quality_list.append(eva_info)

        # -- Risk analysis based revision. --
        correction_num, eva_info = utils.revise_results(workload_y,
                                                        workload_predict_y,
                                                        budget,
                                                        risk_score,
                                                        id_2_label_index)
        correction_num_list.append(correction_num)
        quality_list.append(eva_info)

        budget_2_correction_num[budget] = correction_num_list
        budget_2_quality[budget] = quality_list
    # summary
    print("\n------ Summary ------")
    print("tn: {}, fp: {}, fn: {}, tp: {}".format(tn, fp, fn, tp))
    print("Total pairs can be revised: {}".format(improved_space))
    print("%-10s\t%-13s\t%-13s\t%-13s" % ('Budget', 'Expect', 'TrustScore', 'Improvement1(%)'))
    budget_2_correction_num = sorted(budget_2_correction_num.items())
    for k, v in budget_2_correction_num:
        print("%-10s\t%-13s\t%-13s\t%-13s" % (str(k), str(v[0]), str(v[1]),
                                              str(round(100.0 * (v[1] - v[0]) / v[0], 5))))
    print("### The End. ###\n")

    # save uncertainty scores.
    op_file = open(op_dir + '_trust_score.txt', 'w', 1, encoding='utf-8')
    for i in range(len(risk_score)):
        _id = risk_score[i][0]
        _risk = risk_score[i][1]
        _label_index = id_2_label_index.get(_id)
        _str = "{}, {}, {}, {}".format(workload_y[_label_index],
                                       workload_predict_y[_label_index],
                                       _risk,
                                       _id)
        op_file.write(_str + '\n')
    op_file.flush()
    op_file.close()

    op_file2 = open(op_dir + '_expect_score.txt', 'w', 1, encoding='utf-8')
    for i in range(len(id_2_expectation_based_risk)):
        _id = id_2_expectation_based_risk[i][0]
        _risk = id_2_expectation_based_risk[i][1]
        _label_index = id_2_label_index.get(_id)
        _str = "{}, {}, {}, {}".format(workload_y[_label_index],
                                       workload_predict_y[_label_index],
                                       _risk,
                                       _id)
        op_file2.write(_str + '\n')
    op_file2.flush()
    op_file2.close()


if __name__ == '__main__':
    file_name_ = time.strftime('%Y-%m-%d-', time.localtime(time.time())) + cfg.get_parent_path().split('/')[
        -3] + '-' + cfg.get_parent_path().split('/')[-2] + '-' + str(
        socket.gethostname()) + '_' + '_trust.txt'
    print(file_name_)
    OUT_FIlE = True
    if OUT_FIlE:
        break_point = sys.stdout
        buffering_size = 1  # line buffering (ref: https://docs.python.org/3/library/functions.html#open)
        out_file = open(op_dir + file_name_, 'a+', buffering_size, encoding='utf-8')
        sys.stdout = out_file
        for run_count in range(1):
            main()
        out_file.flush()
        out_file.close()
        sys.stdout = break_point
    else:
        break_point = sys.stdout
        out_file = break_point
        main()
