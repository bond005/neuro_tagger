from argparse import ArgumentParser
import os
import pickle
import sys

import numpy as np
from sklearn.model_selection import ShuffleSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer

try:
    from neuro_tagger.neuro_tagger import NeuroTagger
    from neuro_tagger.dataset_loading import load_dataset_from_brat, tokenize_all_by_sentences
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from neuro_tagger.neuro_tagger import NeuroTagger
    from neuro_tagger.dataset_loading import load_dataset_from_brat, tokenize_all_by_sentences


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The binary file with the best neural network model.')
    parser.add_argument('-d', '--data', dest='data_name', type=str, required=True,
                        help='The data directory with source texts and their Brat annotations.')
    parser.add_argument('--n_calls', dest='number_of_calls', type=int, required=False, default=50,
                        help='The total number of evaluations.')
    parser.add_argument('--cv', dest='cv', type=int, required=False, default=5,
                        help='The folds number for cross-validation.')
    parser.add_argument('--batch', dest='batch_size', type=int, required=False, default=16,
                        help='Size of mini-batch.')
    parser.add_argument('--elmo', dest='elmo_name', type=str, required=True, help='The ELMo model name.')
    parser.add_argument('--lang', dest='language', type=str, required=False, default='english',
                        help='The language of all texts for the sentence tokenizer.')
    args = parser.parse_args()

    model_name = os.path.join(args.model_name)
    model_dir = os.path.dirname(model_name)
    if len(model_dir) > 0:
        assert os.path.isdir(model_dir), 'The directory `{0}` does not exist!'.format(model_dir)
    data_name = os.path.join(args.data_name)
    assert os.path.isdir(data_name), 'The directory `{0}` does not exist!'.format(data_name)
    n_calls = args.number_of_calls
    assert n_calls > 10, 'The total number of evaluations must be a positive integer value greater than 10!'
    cv = args.cv
    assert cv > 1, 'The folds number for cross-validation must be greater than 1.'
    elmo_name = args.elmo_name.strip()
    assert len(elmo_name) > 0, 'The ELMo model name is empty!'
    batch_size = args.batch_size
    assert batch_size > 0, 'The mini-batch size must be a positive value!'

    texts, labels = load_dataset_from_brat(data_name)
    texts, labels = tokenize_all_by_sentences(texts, labels)
    texts = np.array(texts, dtype=object)
    labels = np.array(labels, dtype=object)
    print('Number of texts is {0}.'.format(len(texts)))
    print('')
    rs = ShuffleSplit(n_splits=cv)
    indices_for_cv = [(train_index, test_index) for train_index, test_index in rs.split(texts)]
    del rs

    cls = NeuroTagger(elmo_name=elmo_name, use_crf=True, use_lstm=False, verbose=True, batch_size=batch_size,
                      cached=True)
    opt = BayesSearchCV(
        cls,
        {'l2_kernel': Real(1e-6, 1e+6, prior='log-uniform'), 'l2_chain': Real(1e-6, 1e+6, prior='log-uniform')},
        cv=indices_for_cv,
        refit=True,
        n_iter=n_calls,
        verbose=True,
        n_jobs=1
    )
    opt.fit(texts, labels)
    with open(model_name, 'rb') as fp:
        pickle.dump(opt.best_estimator_, fp)
    print('')
    print('====================')
    print('CRF:')
    print('  - best score is {0:.6f};'.format(opt.best_score_))
    print('  - best parameters are {0}.'.format(opt.best_params_))
    print('====================')
    print('')
    best_score = opt.best_score_
    del cls, opt

    cls = NeuroTagger(elmo_name=elmo_name, use_crf=False, use_lstm=True, verbose=True, batch_size=batch_size,
                      cached=True)
    opt = BayesSearchCV(
        cls,
        {'dropout': Real(0.0, 0.8, prior='uniform'), 'recurrent_dropout': Real(0.0, 0.8, prior='uniform'),
         'n_units': Integer(8, 1024)},
        cv=indices_for_cv,
        refit=True,
        n_iter=n_calls,
        verbose=True,
        n_jobs=1
    )
    opt.fit(texts, labels)
    if best_score < opt.best_score_:
        with open(model_name, 'rb') as fp:
            pickle.dump(opt.best_estimator_, fp)
        best_score = opt.best_score_
    print('')
    print('====================')
    print('BiLSTM:')
    print('  - best score is {0:.6f};'.format(opt.best_score_))
    print('  - best parameters are {0}.'.format(opt.best_params_))
    print('====================')
    print('')
    del cls, opt

    cls = NeuroTagger(elmo_name=elmo_name, use_crf=True, use_lstm=True, verbose=True, batch_size=batch_size,
                      cached=True)
    opt = BayesSearchCV(
        cls,
        {'l2_kernel': Real(1e-6, 1e+6, prior='log-uniform'), 'l2_chain': Real(1e-6, 1e+6, prior='log-uniform'),
         'dropout': Real(0.0, 0.8, prior='uniform'), 'recurrent_dropout': Real(0.0, 0.8, prior='uniform'),
         'n_units': Integer(8, 1024)},
        cv=indices_for_cv,
        refit=True,
        n_iter=n_calls,
        verbose=True,
        n_jobs=1
    )
    opt.fit(texts, labels)
    if best_score < opt.best_score_:
        with open(model_name, 'rb') as fp:
            pickle.dump(opt.best_estimator_, fp)
    print('')
    print('====================')
    print('BiLSTM-CRF:')
    print('  - best score is {0:.6f};'.format(opt.best_score_))
    print('  - best parameters are {0}.'.format(opt.best_params_))
    print('====================')


if __name__ == '__main__':
    main()
