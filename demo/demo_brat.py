from argparse import ArgumentParser
import codecs
import json
import os
import pickle
import sys
from typing import Union, Tuple

import numpy as np
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer

try:
    from neuro_tagger.neuro_tagger import NeuroTagger
    from neuro_tagger.dataset_loading import load_dataset_from_brat, tokenize_all_by_sentences
    from neuro_tagger.tokenizer import DefaultTokenizer
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from neuro_tagger.neuro_tagger import NeuroTagger
    from neuro_tagger.dataset_loading import load_dataset_from_brat, tokenize_all_by_sentences
    from neuro_tagger.tokenizer import DefaultTokenizer


def parse_config(file_name: str) -> Tuple[str, Union[Tuple[float, float], Tuple[int, float, float],
                                                     Tuple[int, float, float, float, float]]]:
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        config = json.load(fp)
    if 'ner_type' not in config:
        raise ValueError('The key `ner_type` is not found in the configuration data from file `{0}`.'.format(file_name))
    ner_type = config['ner_type']
    if ner_type not in {'crf', 'lstm', 'lstm-crf'}:
        raise ValueError('`{0}` is inadmissible value for the `ner_type`!'.format(config['ner_type']))
    if ner_type in {'crf', 'lstm-crf'}:
        if 'l2-chain' not in config:
            raise ValueError('The key `l2-chain` is not found in the configuration data from file '
                             '`{0}`.'.format(file_name))
        if 'l2-kernel' not in config:
            raise ValueError('The key `l2-kernel` is not found in the configuration data from file '
                             '`{0}`.'.format(file_name))
        try:
            l2_chain = float(config['l2-chain'])
        except:
            l2_chain = None
        if l2_chain is None:
            raise ValueError('{0} is inadmissible value for the `l2-chain`!'.format(config['l2-chain']))
        elif l2_chain < 0.0:
            raise ValueError('{0} is inadmissible value for the `l2-chain`!'.format(config['l2-chain']))
        try:
            l2_kernel = float(config['l2-kernel'])
        except:
            l2_kernel = None
        if l2_kernel is None:
            raise ValueError('{0} is inadmissible value for the `l2-kernel`!'.format(config['l2-kernel']))
        elif l2_kernel < 0.0:
            raise ValueError('{0} is inadmissible value for the `l2-kernel`!'.format(config['l2-kernel']))
    else:
        l2_kernel = None
        l2_chain = None
    if ner_type in {'lstm', 'lstm-crf'}:
        if 'n_units' not in config:
            raise ValueError('The key `n_units` is not found in the configuration data from file '
                             '`{0}`.'.format(file_name))
        if 'dropout' not in config:
            raise ValueError('The key `dropout` is not found in the configuration data from file '
                             '`{0}`.'.format(file_name))
        if 'recurrent_dropout' not in config:
            raise ValueError('The key `recurrent_dropout` is not found in the configuration data from file '
                             '`{0}`.'.format(file_name))
        try:
            n_units = int(config['n_units'])
        except:
            n_units = None
        if n_units is None:
            raise ValueError('{0} is inadmissible value for the `n_units`!'.format(config['n_units']))
        elif n_units <= 1:
            raise ValueError('{0} is inadmissible value for the `n_units`!'.format(config['n_units']))
        try:
            dropout = float(config['dropout'])
        except:
            dropout = None
        if dropout is None:
            raise ValueError('{0} is inadmissible value for the `dropout`!'.format(config['dropout']))
        elif (dropout < 0.0) or (dropout >= 1.0):
            raise ValueError('{0} is inadmissible value for the `dropout`!'.format(config['dropout']))
        try:
            recurrent_dropout = float(config['recurrent_dropout'])
        except:
            recurrent_dropout = None
        if recurrent_dropout is None:
            raise ValueError('{0} is inadmissible value for the `recurrent_dropout`!'.format(
                config['recurrent_dropout']))
        elif (recurrent_dropout < 0.0) or (recurrent_dropout >= 1.0):
            raise ValueError('{0} is inadmissible value for the `recurrent_dropout`!'.format(
                config['recurrent_dropout']))
    else:
        n_units = None
        dropout = None
        recurrent_dropout = None
    if ner_type == 'crf':
        return ner_type, (l2_kernel, l2_chain)
    if ner_type == 'lstm':
        return ner_type, (n_units, dropout, recurrent_dropout)
    return ner_type, (n_units, dropout, recurrent_dropout, l2_kernel, l2_chain)


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
    parser.add_argument('--config', dest='config_file', type=str, required=False, default=None,
                        help='The JSON file with NER configuration (if it is not specified, then optimal NER '
                             'configuration is selected automatically).')
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
    config_file_name = args.config_file
    if config_file_name is None:
        config = None
    else:
        config_file_name = os.path.normpath(config_file_name)
        assert os.path.isfile(config_file_name), 'The file `{0}` does not exist!'.format(config_file_name)
        config = parse_config(config_file_name)
    texts, labels = load_dataset_from_brat(data_name)
    texts, labels = tokenize_all_by_sentences(texts, labels)
    lengths_of_texts = np.array([len(cur.split()) for cur in texts], dtype=np.int32)
    print('Total number of texts is {0}.'.format(len(lengths_of_texts)))
    print('Maximal number of tokens in text is {0}.'.format(lengths_of_texts.max()))
    print('Mean number of tokens in text is {0} +- {1}.'.format(lengths_of_texts.mean(), lengths_of_texts.std()))
    print('')
    NeuroTagger.print_info_about_labels(labels)
    texts = np.array(texts, dtype=object)
    labels = np.array(labels, dtype=object)
    indices_for_cv = NeuroTagger.stratified_kfold(texts, labels, cv)
    if config is not None:
        ner_type = config[0]
        if ner_type == 'crf':
            cls = NeuroTagger(elmo_name=elmo_name, use_crf=True, use_lstm=False, verbose=True, batch_size=batch_size,
                              cached=True, n_epochs=1000, tokenizer=DefaultTokenizer(), l2_kernel=config[1][0],
                              l2_chain=config[1][1])
        elif ner_type == 'lstm':
            cls = NeuroTagger(elmo_name=elmo_name, use_crf=False, use_lstm=True, verbose=True, batch_size=batch_size,
                              cached=True, n_epochs=1000, tokenizer=DefaultTokenizer(), n_units=config[1][0],
                              dropout=config[1][1], recurrent_dropout=config[1][2])
        else:
            cls = NeuroTagger(elmo_name=elmo_name, use_crf=True, use_lstm=True, verbose=True, batch_size=batch_size,
                              cached=True, n_epochs=1000, tokenizer=DefaultTokenizer(), n_units=config[1][0],
                              dropout=config[1][1], recurrent_dropout=config[1][2], l2_kernel=config[1][3],
                              l2_chain=config[1][4])
        f1 = cross_val_score(cls, X=texts, y=labels, cv=indices_for_cv, n_jobs=1)
        print('')
        print('F1-score is {0:.6f}.'.format(f1))
        return
    cls = NeuroTagger(elmo_name=elmo_name, use_crf=True, use_lstm=False, verbose=True, batch_size=batch_size,
                      cached=True, n_epochs=1000, tokenizer=DefaultTokenizer())
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
    with open(model_name, 'wb') as fp:
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
                      cached=True, n_epochs=1000, tokenizer=DefaultTokenizer())
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
        with open(model_name, 'wb') as fp:
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

    cls = NeuroTagger(elmo_name=elmo_name, use_crf=True, use_lstm=True, verbose=2, batch_size=batch_size,
                      cached=True, n_epochs=1000, tokenizer=DefaultTokenizer())
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
        with open(model_name, 'wb') as fp:
            pickle.dump(opt.best_estimator_, fp)
    print('')
    print('====================')
    print('BiLSTM-CRF:')
    print('  - best score is {0:.6f};'.format(opt.best_score_))
    print('  - best parameters are {0}.'.format(opt.best_params_))
    print('====================')


if __name__ == '__main__':
    main()
