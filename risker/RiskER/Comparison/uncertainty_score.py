from common import utils
import time
import sys
import socket
from common import config
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

cfg = config.Configuration(config.global_selection)

op_dir = 'results/' + cfg.get_parent_path().split('/')[-3] + '/' + cfg.get_parent_path().split('/')[-2] + '/'


if not os.path.exists(op_dir):
    os.mkdir(op_dir)

NUMBER_OF_MODELS = 20


def cal_bootstrap_prob(clf_data):
    result_middle_name = 'test_prediction_'
    id_2_predict_labels = dict()
    test_ids = clf_data['test'].ids
    for i in range(len(test_ids)):
        id_2_predict_labels[test_ids[i]] = []
    for i in range(NUMBER_OF_MODELS):
        predcit_result_path = cfg.get_parent_path() + result_middle_name + str(i + 1) + '.csv'
        workload_info = pd.read_csv(predcit_result_path).values
        workload_probs = workload_info[:, 1]
        workload_predict_y = utils.get_predict_label(workload_probs)
        workload_ids = workload_info[:, 0].astype(str)
        for j in range(len(workload_ids)):
            id_2_predict_labels.get(workload_ids[j]).append(workload_predict_y[j])
    id_2_match_prob = dict()
    for k, v in id_2_predict_labels.items():
        id_2_match_prob[k] = np.average(v)
    return id_2_match_prob


def cal_uncertainty_score(clf_data):
    test_ids = clf_data['test'].ids
    id_2_match_prob = cal_bootstrap_prob(clf_data)
    id_2_uncertainty = dict()
    for i in range(len(test_ids)):
        test_prob = id_2_match_prob.get(test_ids[i])
        id_2_uncertainty[test_ids[i]] = test_prob * (1 - test_prob)  # score = p * (1 - p)
    id_uncertainty_scores = sorted(id_2_uncertainty.items(), key=lambda item: item[1], reverse=True)
    return id_uncertainty_scores


def main():
    clf_data = utils.load_results(cfg)
    risk_score = cal_uncertainty_score(clf_data)
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
    print("%-10s\t%-13s\t%-13s\t%-13s" % ('Budget', 'Expect', 'Uncertainty', 'Improvement1(%)'))
    budget_2_correction_num = sorted(budget_2_correction_num.items())
    for k, v in budget_2_correction_num:
        print("%-10s\t%-13s\t%-13s\t%-13s" % (str(k), str(v[0]), str(v[1]),
                                              str(round(100.0 * (v[1] - v[0]) / v[0], 5))))
    print("### The End. ###\n")

    # save uncertainty scores.
    op_file = open(op_dir + time.strftime('%Y-%m-%d', time.localtime(time.time())) +
                   '_uncertain_score.txt', 'w', 1, encoding='utf-8')
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


if __name__ == '__main__':
    file_name_ = time.strftime('%Y-%m-%d-', time.localtime(time.time())) + cfg.get_parent_path().split('/')[
        -3] + '-' + cfg.get_parent_path().split('/')[-2] + '-' + str(
        socket.gethostname()) + '_uncertainty.txt'
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

