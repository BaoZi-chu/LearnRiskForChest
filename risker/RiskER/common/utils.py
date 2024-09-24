import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import scipy
import collections
import pandas as pd


def get_predict_label(_probs):
    prob_temp = np.array(_probs)
    _label = (prob_temp >= 0.5).astype(int)
    return _label


def get_true_label(_ids, id_2_pinfo):
    _y = []
    for _id in _ids:
        _y.append(id_2_pinfo.get(_id)[1])
    return _y


def calculate_feature_mu_sigma(ob_ids, ob_labels):
    """

    :param ob_ids: observed data ids.
    :param ob_labels: observed data labels.
    :return: _mu: mean (sample mean)
              _sigma: variance (the second central moment of samples)
    """
    _labels = []
    for _id in ob_ids:
        _labels.append(ob_labels.get(_id))
    _labels = np.array(_labels)
    _mu = np.average(_labels)
    _delta = (_labels - _mu) ** 2
    _sum = np.sum(_delta)
    _sigma = _sum / np.maximum(len(_labels) - 1, 1)
    # - Select the discriminative features.
    # if 0.1 < _mu < 0.9:
    #     return None
    # else:
    #     return [_mu, _sigma]
    # - No selection.
    return [_mu, _sigma]


def revise_results(_true_labels, _machine_labels, budget, id_2_risk_desc, id_2_index):
    revised_machine_labels = _machine_labels.copy()
    correct_number = 0
    i = 0
    while i < budget and i < len(id_2_risk_desc):
        pair_id = id_2_risk_desc[i][0]
        _index = id_2_index.get(pair_id)
        if _true_labels[_index] != _machine_labels[_index]:
            # -- The ground-truth label does not equal to classifier label. --
            correct_number += 1
        revised_machine_labels[_index] = _true_labels[_index]
        i += 1

    precision = precision_score(_true_labels, revised_machine_labels)
    recall = recall_score(_true_labels, revised_machine_labels)
    f_one = f1_score(_true_labels, revised_machine_labels)
    eva_info = [precision, recall, f_one]
    return correct_number, eva_info


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


def load_results(config):
    print("-------------------------------------------")
    print("- Data source: {}.".format(config.get_raw_data_path()))
    df = pd.read_csv(config.get_raw_data_path())
    pairs = df.values
    print("- # of data: {}.".format(len(pairs)))
    id_2_pair_info = dict()
    for elem in pairs:
        id_2_pair_info[elem[0]] = elem  # <id, info>

    if config.use_other_domain_workload():
        df2 = pd.read_csv(config.get_shift_raw_data_path())
        pairs2 = df2.values
        print("- Shift data source: {}.".format(config.get_shift_raw_data_path()))
        print("- # of shift data: {}.".format(len(pairs2)))
        for elem in pairs2:
            id_2_pair_info[elem[0]] = elem  # <id, info>

    # -- Using deep learning model as the classifier. --
    # -- Train data --
    train_info = pd.read_csv(config.get_parent_path() + 'train_prediction.csv').values
    train_probs = train_info[:, 1]
    train_predict_y = get_predict_label(train_probs)
    train_ids = train_info[:, 0].astype(str)
    train_y = get_true_label(train_ids, id_2_pair_info)
    train_precision = precision_score(train_y, train_predict_y)
    train_recall = recall_score(train_y, train_predict_y)
    train_f1 = f1_score(train_y, train_predict_y)
    print("- Train data's Precision: {}, Recall: {}, F1-Score: {}.".format(train_precision, train_recall, train_f1))

    # -- Validation data --
    valida_info = pd.read_csv(config.get_parent_path() + 'validation_prediction.csv').values
    valida_probs = valida_info[:, 1]
    valida_predict_y = get_predict_label(valida_probs)
    valida_ids = valida_info[:, 0].astype(str)
    valida_y = get_true_label(valida_ids, id_2_pair_info)
    valida_precision = precision_score(valida_y, valida_predict_y)
    valida_recall = recall_score(valida_y, valida_predict_y)
    valida_f1 = f1_score(valida_y, valida_predict_y)
    print("- Validation data's Precision: {}, Recall: {}, F1-Score: {}.".format(valida_precision, valida_recall,
                                                                                valida_f1))

    # -- Workload --
    workload_info = pd.read_csv(config.get_parent_path() + 'test_prediction.csv').values
    workload_probs = workload_info[:, 1]
    workload_predict_y = get_predict_label(workload_probs)
    workload_ids = workload_info[:, 0].astype(str)
    workload_y = get_true_label(workload_ids, id_2_pair_info)
    workload_precision = precision_score(workload_y, workload_predict_y)
    workload_recall = recall_score(workload_y, workload_predict_y)
    workload_f1 = f1_score(workload_y, workload_predict_y)
    print("- Test data's Precision: {}, Recall: {}, F1-Score: {}.".format(workload_precision, workload_recall,
                                                                          workload_f1))

    dl_model_data_len = len(train_ids) + len(valida_ids) + len(workload_ids)
    print("- [Train : Validation : Test] = [{:.1f} : {:.1f} : {:.1f}]".format(1.0 * len(train_ids) / dl_model_data_len,
                                                                              1.0 * len(valida_ids) / dl_model_data_len,
                                                                              1.0 * len(
                                                                                  workload_ids) / dl_model_data_len))

    clf_info = collections.namedtuple('clf_info', ['ids', 'probs', 'true_y', 'predict_y'])
    clf_train = clf_info(train_ids, train_probs, train_y, train_predict_y)
    clf_validation = clf_info(valida_ids, valida_probs, valida_y, valida_predict_y)
    clf_test = clf_info(workload_ids, workload_probs, workload_y, workload_predict_y)

    clf_summary = dict()
    clf_summary['train'] = clf_train
    clf_summary['validation'] = clf_validation
    clf_summary['test'] = clf_test
    clf_summary['pair_info'] = id_2_pair_info
    return clf_summary
