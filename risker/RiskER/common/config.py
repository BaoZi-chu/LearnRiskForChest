import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

"""
Data sets selection:
0: DBLP-Scholar
1: Abt-Buy
2: Amazon-Google
3: songs
4: DA2DS
5: AB2AG
"""

global_selection = 0


class Configuration(object):
    def __init__(self, selection):
        self.data_selection = selection
        self.train_valida_test_ratio = {0: '127', 1: '226', 2: '325'}
        self.path_dict = {0: rootPath + '/input_data/DBLP-Scholar/',
                          1: rootPath + '/input_data/Abt-Buy/',
                          2: rootPath + '/input_data/Amazon-Google/',
                          3: rootPath + '/input_data/songs/',
                          4: rootPath + '/input_data/DA2DS/',
                          5: rootPath + '/input_data/AB2AG/'}

        self.data_source1_dict = {0: self.path_dict.get(0) + 'DBLP1.csv',
                                  1: self.path_dict.get(1) + 'Abt.csv',
                                  2: self.path_dict.get(2) + 'Amazon.csv',
                                  3: self.path_dict.get(3) + 'msd.csv',
                                  4: self.path_dict.get(4) + 'DBLP1.csv',
                                  5: self.path_dict.get(5) + 'Amazon.csv'}

        self.data_source2_dict = {0: self.path_dict.get(0) + 'Scholar.csv',
                                  1: self.path_dict.get(1) + 'Buy_new.csv',
                                  2: self.path_dict.get(2) + 'GoogleProducts.csv',
                                  3: None,
                                  4: self.path_dict.get(4) + 'Scholar.csv',
                                  5: self.path_dict.get(5) + 'GoogleProducts.csv'}

        self.tvt_selection = 2
        self.risk_training_size = 20
        self.random_select_risk_training = False
        self.risk_confidence = 0.9
        self.minimum_observation_num = 5
        self.budget_levels = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                              1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
                              2500, 3000, 3500, 4000, 4500, 5000]
        self.interval_number_4_continuous_value = 50
        self.learn_variance = True
        self.apply_function_to_weight_classifier_output = True
        self.deepmatcher_epochs = 10
        self.risk_training_epochs = 1000

    def get_parent_path(self):
        return self.path_dict.get(self.data_selection) + self.train_valida_test_ratio.get(self.tvt_selection) + '/'

    def get_data_source_1(self):
        return self.data_source1_dict.get(self.data_selection)

    def get_data_source_2(self):
        return self.data_source2_dict.get(self.data_selection)

    def get_raw_data_path(self):
        return self.path_dict.get(self.data_selection) + 'pair_info_more.csv'

    def get_shift_raw_data_path(self):
        return self.path_dict.get(self.data_selection) + 'pair_info_more_2.csv'

    def get_raw_decision_tree_rules_path(self):
        return self.get_parent_path() + 'decision_tree_rules_raw.txt'

    def get_decision_tree_rules_path(self):
        return self.get_parent_path() + 'decision_tree_rules_clean.txt'

    def use_other_domain_workload(self):
        return self.data_selection in {4, 5}

    @staticmethod
    def get_budgets():
        budget_levels = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                         1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
                         2500, 3000, 3500, 4000, 4500, 5000]
        return budget_levels
