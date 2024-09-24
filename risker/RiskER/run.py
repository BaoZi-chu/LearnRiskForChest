import datetime
from common import config
import scipy
from data_process import risk_dataset
from risker import risk_model
import pandas as pd
from sklearn.metrics import confusion_matrix
import time
import draw_results
import socket
import sys
import os
import argparse


def prepare_data_4_risk_model(cfg):
    print("\n\n### {} ###".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("- Start running: {}.".format(__file__.split("/")[-1]))
    rule_file = cfg.get_parent_path() + 'decision_tree_rules_info.txt'
    print("- Load rules from: {}".format(rule_file))
    print("------ Configuration of input rules -------")
    rule_cfg = open(rule_file, 'r')
    for line in rule_cfg.readlines():
        print('|' + line.strip('\n'))
        if "(After cleaning)" in line:
            break
    train_data, validation_data, test_data = risk_dataset.load_data(cfg)
    rm = risk_model.RiskModel()
    rm.train_data = train_data
    rm.validation_data = validation_data
    rm.test_data = test_data
    return rm


# Write results to independent files for future analysis, e.g., calculate AUROC.
def output_risk_scores(file_path, id_2_scores, label_index, ground_truth_y, predict_y):
    op_file = open(file_path, 'w', 1, encoding='utf-8')
    for i in range(len(id_2_scores)):
        _id = id_2_scores[i][0]
        _risk = id_2_scores[i][1]
        _label_index = label_index.get(_id)
        _str = "{}, {}, {}, {}".format(ground_truth_y[_label_index],
                                       predict_y[_label_index],
                                       _risk,
                                       _id)
        op_file.write(_str + '\n')
    op_file.flush()
    op_file.close()
    return True


# The Comparison method (Baseline): measuring the risk based on the classifier's output probability.
def expectation_based_risk(_ids, _machine_labels, _expectations):
    id_2_risk = []
    for i in range(len(_ids)):
        expectation = _expectations[i]
        m_label = _machine_labels[i]
        if m_label == 1:
            label_value = 1.0
        else:
            label_value = 0.0
        id_2_risk.append([_ids[i], scipy.absolute(label_value - expectation)])  # Larger difference, Higher risk.
    # In descending order, i.e., pairs with higher risk rank ahead.
    id_2_risk_desc = sorted(id_2_risk, key=lambda item: item[1], reverse=True)
    return id_2_risk_desc


def revise_results(_true_labels, _machine_labels, budget, id_2_risk_desc, id_2_index):
    correct_number = 0
    i = 0
    while i < budget and i < len(id_2_risk_desc):
        pair_id = id_2_risk_desc[i][0]
        _index = id_2_index.get(pair_id)
        if _true_labels[_index] != _machine_labels[_index]:
            # -- The ground-truth label does not equal to classifier label. --
            correct_number += 1
        i += 1
    return correct_number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, help='Specify data set.'
                                                    '0: DBLP-Scholar, 1: Abt-Buy, 2: Amazon-Google,'
                                                    '3: songs, 4: DA2DS, 5: AB2AG.')
    parser.add_argument('--split', type=int,
                        help='Specify data split. The ratio of Training:Validation:Test.'
                             '0: (1:2:7), 1: (2:2:6), 2: (3:2:5).')
    parser.add_argument('--epoch', type=int,
                        help='Specify the number of epochs for risk model training.')
    parser.add_argument('--dl-epoch', type=int,
                        help='Specify the number of epochs for DeepMatcher training.')
    args = parser.parse_args()

    cfg = config.Configuration(config.global_selection)

    if args.dataset is not None:
        cfg.data_selection = args.dataset
    if args.split is not None:
        cfg.tvt_selection = args.split
    if args.epoch is not None:
        cfg.risk_training_epochs = args.epoch
    if args.dl_epoch is not None:
        cfg.deepmatcher_epochs = args.dl_epoch

    OUT_FIlE_FLAG = True  # Print out the information onto a file.
    out_file = None
    if OUT_FIlE_FLAG:
        curPath = os.path.abspath(os.path.dirname(__file__))
        file_name_ = time.strftime('%Y-%m-%d-', time.localtime(time.time())) + cfg.get_parent_path().split('/')[
            -3] + '-' + str(socket.gethostname()) + '-' + cfg.train_valida_test_ratio.get(cfg.tvt_selection) + '-' + str(
            __file__.split("/")[-1].split('.')[0]) + '.txt'
        break_point = sys.stdout
        buffering_size = 1  # line buffering (ref: https://docs.python.org/3/library/functions.html#open)
        out_file = open(curPath + "/results/" + file_name_, 'a+', buffering_size, encoding='utf-8')
        sys.stdout = out_file
        print("Log the information on {}.".format(curPath + "/results/" + file_name_))

    # -- The Start of Risk Analysis --

    # Create a risk model and also load the data information.
    my_risk_model = prepare_data_4_risk_model(cfg)

    # Get the classifier's output probabilities.
    train_info = pd.read_csv(cfg.get_parent_path() + 'train_prediction.csv').values
    train_predicts = train_info[:, 1]
    valida_info = pd.read_csv(cfg.get_parent_path() + 'validation_prediction.csv').values
    valida_predicts = valida_info[:, 1]
    test_info = pd.read_csv(cfg.get_parent_path() + 'test_prediction.csv').values
    test_predicts = test_info[:, 1]

    # Train the risk model, i.e., learn the risk features' weights and variances.
    my_risk_model.train(train_predicts, valida_predicts)

    # Apply the trained risk model on the test data.
    # To rank the data based on their risks of being mislabeled.
    my_risk_model.predict(test_predicts)

    risk_scores = my_risk_model.test_data.risk_values.reshape(-1)

    # -- The End of Risk Analysis --

    # Just for evaluating the performance.
    test_num = my_risk_model.test_data.data_len
    test_ids = my_risk_model.test_data.data_ids
    test_predict_y = my_risk_model.test_data.machine_labels

    id_2_label_index = dict()
    id_2_VaR_risk = []
    for index in range(test_num):
        id_2_VaR_risk.append([test_ids[index], risk_scores[index]])
        id_2_label_index[test_ids[index]] = index
    id_2_VaR_risk = sorted(id_2_VaR_risk, key=lambda item: item[1], reverse=True)

    # -- Expectation based risk analysis. --
    id_2_expectation_based_risk = expectation_based_risk(test_ids, test_predict_y, test_predicts)

    # -- Improve the machine results according to the risk analysis. --
    test_ground_truth_y = my_risk_model.test_data.true_labels
    # tn, fp, fn, tp = confusion_matrix(test_ground_truth_y, test_predict_y).ravel()
    # improved_space = fp + fn
    # budgets = cfg.get_budgets()
    # budget_2_correction_num = dict()
    # budgets.append(improved_space)
    # budgets = sorted(budgets)
    # for budget in budgets:
    #     correction_num_list = []
    #     # quality_list = []
    #     # -- Expectation based revision. --
    #     correction_num = revise_results(test_ground_truth_y,
    #                                     test_predict_y,
    #                                     budget,
    #                                     id_2_expectation_based_risk,
    #                                     id_2_label_index)
    #     correction_num_list.append(correction_num)
    #
    #     # -- Risk analysis based revision. --
    #     correction_num = revise_results(test_ground_truth_y,
    #                                     test_predict_y,
    #                                     budget,
    #                                     id_2_VaR_risk,
    #                                     id_2_label_index)
    #     correction_num_list.append(correction_num)
    #
    #     budget_2_correction_num[budget] = correction_num_list

    # Summary
    # print("\n------ Summary ------")
    # print("tn: {}, fp: {}, fn: {}, tp: {}".format(tn, fp, fn, tp))
    # print("Total pairs can be revised: {}".format(improved_space))
    # print("%-10s\t%-13s\t%-13s" % ('Budget', 'Expect', 'VaR'))
    # budget_2_correction_num = sorted(budget_2_correction_num.items())
    # for k, v in budget_2_correction_num:
    #     print("%-10s\t%-13s\t%-13s" % (str(k), str(v[0]), str(v[1])))
    # print("### The End. ###\n")

    file_pth = cfg.get_parent_path() + time.strftime('%Y-%m-%d_', time.localtime(
        time.time())) + str(cfg.train_valida_test_ratio.get(cfg.tvt_selection)) + '_risk_score.txt'
    print("- The corresponding risk score path (Risk Model): {}".format(file_pth))
    output_risk_scores(file_pth, id_2_VaR_risk, id_2_label_index, test_ground_truth_y, test_predict_y)

    file_pth = cfg.get_parent_path() + time.strftime('%Y-%m-%d_', time.localtime(
        time.time())) + str(cfg.train_valida_test_ratio.get(cfg.tvt_selection)) + '_baseline_score.txt'
    print("- The corresponding risk score path (Baseline): {}".format(file_pth))
    output_risk_scores(file_pth, id_2_expectation_based_risk, id_2_label_index, test_ground_truth_y, test_predict_y)

    # Just visualize the comparison AUROC results.
    draw_results.plot(cfg.get_parent_path(),
                      target_file_names=['baseline_score.txt', 'risk_score.txt'],
                      method_names=['Baseline', 'LearnRisk'],
                      colors=['slateblue', 'limegreen'],
                      lines=['--', '-'])

    if OUT_FIlE_FLAG and out_file is not None:
        out_file.flush()
        out_file.close()


if __name__ == '__main__':
    main()
