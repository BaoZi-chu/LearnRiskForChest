import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import datetime
import time
from common import config
import deepmatcher as dm


train_pred_path = '/train_prediction.csv'
valid_pred_path = '/validation_prediction.csv'
test_pred_path = '/test_prediction.csv'

cfg = config.Configuration(config.global_selection)


def main():
    _dir = cfg.get_parent_path().rstrip('/')  # Remove the last '/'.
    if not os.path.exists(_dir):
        print("Warning: Data source is not found!")
        sys.exit()

    print("\n--- {} ---\n".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    _time_start = time.time()
    train, validation, test = dm.data.process(path=_dir,
                                              train='train.csv',
                                              validation='validation.csv',
                                              test='test.csv',
                                              ignore_columns=['left_id', 'right_id'],
                                              embeddings='fasttext.en.bin',
                                              embeddings_cache_path='E:/vector_cache')
    '''
    attr_summarizer: 
    * sif: This model considers the words present in each attribute value pair to determine a match or non-match. 
           It does not take word order into account.
    * rnn: This model considers the sequences of words present in each attribute value pair to determine a match 
           or non-match.
    * attention: This model considers the alignment of words present in each attribute value pair to determine a match 
                 or non-match. It does not take word order into account.
    * hybrid: This model considers the alignment of sequences of words present in each attribute value pair to 
              determine a match or non-match. This is the default.
    '''
    model = dm.MatchingModel(attr_summarizer='hybrid')
    model.run_train(train,
                    validation,
                    epochs=10,
                    best_save_path=_dir + '/hybrid_model.pth',
                    test_data=test)
    train_predictions = model.run_prediction(train)
    validation_predictions = model.run_prediction(validation)
    test_predictions = model.run_prediction(test)
    # -- Save results to files. --
    train_predictions.to_csv(_dir + train_pred_path)
    validation_predictions.to_csv(_dir + valid_pred_path)
    test_predictions.to_csv(_dir + test_pred_path)
    print("running time: {:.2f}h.".format(1.0 * (time.time() - _time_start) / 3600))
    print("\n--- {} ---\n".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


if __name__ == '__main__':
    main()
