from argparse import ArgumentParser
import codecs
import os
import pickle
import sys

from skopt import BayesSearchCV
from skopt.space import Real, Integer

try:
    from neuro_tagger.neuro_tagger import NeuroTagger
    from neuro_tagger.dataset_loading import load_dataset_from_factrueval2016
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from neuro_tagger.neuro_tagger import NeuroTagger
    from neuro_tagger.dataset_loading import load_dataset_from_factrueval2016


def main():
    POSSIBLE_MODEL_TYPES = {'crf', 'lstm', 'lstm-crf'}
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The binary file with the best neural network model.')
    parser.add_argument('-d', '--data', dest='data_name', type=str, required=True,
                        help='Path to the FactRuEval-2016 repository.')
    parser.add_argument('-r', '--result', dest='result_name', type=str, required=True,
                        help='The directory into which all recognized named entity labels will be saved.')
    parser.add_argument('--n_calls', dest='number_of_calls', type=int, required=False, default=50,
                        help='The total number of evaluations.')
    parser.add_argument('--cv', dest='cv', type=int, required=False, default=5,
                        help='The folds number for cross-validation.')
    parser.add_argument('--batch', dest='batch_size', type=int, required=False, default=16,
                        help='Size of mini-batch.')
    parser.add_argument('--elmo', dest='elmo_name', type=str, required=True, help='The ELMo model name.')
    parser.add_argument('--type', dest='type_of_model', type=str, required=False, default='crf',
                        help='The type of model (crf, lstm or lstm-crf).')
    args = parser.parse_args()

    model_name = os.path.join(args.model_name)
    model_dir = os.path.dirname(model_name)
    if len(model_dir) > 0:
        assert os.path.isdir(model_dir), 'The directory `{0}` does not exist!'.format(model_dir)
    factrueval_name = os.path.join(args.data_name)
    assert os.path.isdir(factrueval_name), 'The directory `{0}` does not exist!'.format(factrueval_name)
    result_dir_name = os.path.join(args.result_name)
    assert os.path.isdir(result_dir_name), 'The directory `{0}` does not exist!'.format(result_dir_name)
    n_calls = args.number_of_calls
    assert n_calls > 10, 'The total number of evaluations must be a positive integer value greater than 10!'
    cv = args.cv
    assert cv > 1, 'The folds number for cross-validation must be greater than 1.'
    elmo_name = args.elmo_name.strip()
    assert len(elmo_name) > 0, 'The ELMo model name is empty!'
    batch_size = args.batch_size
    assert batch_size > 0, 'The mini-batch size must be a positive value!'
    model_type = args.type_of_model
    assert model_type in POSSIBLE_MODEL_TYPES, '`{0}` is unknown model type!'.format(model_type)

    if os.path.isfile(model_name):
        with open(model_name, 'rb') as fp:
            model_for_factrueval = pickle.load(fp)
        assert isinstance(model_for_factrueval, NeuroTagger), \
            'Classifier for the FactRuEval-2016 cannot be loaded from the file `{0}`.'.format(model_name)
        print('Classifier for the FactRuEval-2016 has been loaded...')
    else:
        texts_for_training, labels_for_training, _ = load_dataset_from_factrueval2016(
            os.path.join(factrueval_name, 'devset'))
        print('Data for training have been loaded...')
        NeuroTagger.print_info_about_labels(labels_for_training)
        indices_for_cv = NeuroTagger.stratified_kfold(texts_for_training, labels_for_training, cv)
        if model_type == 'crf':
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
        elif model_type == 'lstm':
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
        else:
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
        opt.fit(texts_for_training, labels_for_training)
        model_for_factrueval = opt.best_estimator_
        with open(model_name, 'wb') as fp:
            pickle.dump(model_for_factrueval, fp)
        print('Classifier for the FactRuEval-2016 has been trained and saved...')
        print('Best parameters are: {0}.'.format(opt.best_params_))
        print('Best score is {0:.6f}.'.format(opt.best_score_))
    print('')
    texts_for_testing, labels_for_testing, book_names = load_dataset_from_factrueval2016(
        os.path.join(factrueval_name, 'testset'))
    print('Data for testing have been loaded...')
    NeuroTagger.print_info_about_labels(labels_for_testing)
    predicted_labels = model_for_factrueval.predict(texts_for_testing)
    cur_book_name = ''
    fp = None
    start_pos = 0
    try:
        for text_idx in range(len(texts_for_testing)):
            if cur_book_name != book_names[text_idx]:
                if fp is not None:
                    fp.close()
                start_pos = 0
                cur_book_name = book_names[text_idx]
                fp = codecs.open(os.path.join(result_dir_name, cur_book_name + '.task1'), mode='w', encoding='utf-8')
            if len(predicted_labels[text_idx]) > 0:
                for cur_ne in predicted_labels[text_idx]:
                    fp.write('{0} {1} {2}\n'.format(cur_ne[0].lower(), cur_ne[1] + start_pos, cur_ne[2]))
                fp.write('\n')
            start_pos += len(texts_for_testing[text_idx])
    finally:
        if fp is not None:
            fp.close()


if __name__ == '__main__':
    main()
