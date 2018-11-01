import copy
import math
import os
import re
import tempfile
from typing import List, Set, Tuple, Union

from keras_contrib.layers import CRF
import keras.backend as K
from keras.layers import Input, LSTM, Masking, Bidirectional, TimeDistributed, Dense
from keras.models import Model
from keras.regularizers import l2
from keras.utils import print_summary
from nltk.tokenize.nist import NISTTokenizer
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf
import tensorflow_hub as hub


class NeuroTagger(ClassifierMixin, BaseEstimator):
    CACHE = None

    def __init__(self, elmo_name: str, n_units: int=512, dropout: float=0.7, recurrent_dropout: float=0.0,
                 l2_kernel: float=1e-3, l2_chain: float=1e-6, n_epochs: int=100, validation_part: float=0.2,
                 batch_size: int=32, use_lstm: bool=False, use_crf: bool=True, verbose: Union[int, bool]=True,
                 cached: bool=False):
        self.elmo_name = elmo_name
        self.n_units = n_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.l2_kernel = l2_kernel
        self.l2_chain = l2_chain
        self.n_epochs = n_epochs
        self.validation_part = validation_part
        self.verbose = verbose
        self.use_lstm = use_lstm
        self.use_crf = use_crf
        self.batch_size = batch_size
        self.cached = cached

    def __del__(self):
        if (hasattr(self, 'elmo_') or hasattr(self, 'classifier_')):
            if hasattr(self, 'elmo_'):
                del self.elmo_
            if hasattr(self, 'classifier_'):
                del self.classifier_
            K.clear_session()
        if hasattr(self, 'tokenizer_'):
            del self.tokenizer_

    def fit(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray]):
        self.check_params(**self.get_params(deep=False))
        self.check_X(X, 'X')
        self.named_entities_ = self.check_y(y, [len(X[idx]) for idx in range(len(X))], 'y')
        self.update_elmo()
        X_tokenized = self.tokenize(X)
        self.max_text_len_ = max(map(lambda idx: len(X_tokenized[idx]), range(len(X_tokenized))))
        if self.verbose:
            print('Maximal length of text is {0}.'.format(self.max_text_len_))
        K.get_session()
        X_ = self.texts_to_X(X, X_tokenized, self.max_text_len_)
        y_ = self.labels_to_y(X, y, X_tokenized, self.max_text_len_)
        if self.verbose:
            print('All data have been prepared using ELMo.')
        train_index, test_index = self.stratified_split(X, y, self.validation_part)
        X_train = X_[train_index]
        y_train = y_[train_index]
        X_test = X_[test_index]
        y_test = y_[test_index]
        X_tokenized_test = []
        for idx in test_index:
            X_tokenized_test.append(X_tokenized[idx])
        del X_
        del y_
        self.embedding_size_ = X_train.shape[2]
        nn_input = Input(shape=(self.max_text_len_, self.embedding_size_), name='input_of_tagger')
        nn_output = Masking(mask_value=0.0, input_shape=(self.max_text_len_, self.embedding_size_),
                            name='masking_layer')(nn_input)
        if self.use_lstm:
            nn_output = Bidirectional(LSTM(self.n_units, return_sequences=True, dropout=self.dropout,
                                           recurrent_dropout=self.recurrent_dropout, name='lstm_layer'),
                                      merge_mode='ave', name='BiLSTM_layer')(nn_output)
            if self.use_crf:
                crf = CRF(units=len(self.named_entities_) * 2 + 1, learn_mode='join', test_mode='viterbi',
                          kernel_regularizer=(l2(self.l2_kernel) if self.l2_kernel > 0.0 else None),
                          chain_regularizer=(l2(self.l2_chain) if self.l2_chain > 0.0 else None), name='crf_layer')
                nn_output = crf(nn_output)
                self.classifier_ = Model(nn_input, nn_output)
                self.classifier_.compile(optimizer='rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])
            else:
                nn_output = TimeDistributed(Dense(len(self.named_entities_) * 2 + 1, activation='softmax',
                                                  name='dense_layer'), name='time_distr')(nn_output)
                self.classifier_ = Model(nn_input, nn_output)
                self.classifier_.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                                         metrics=['categorical_accuracy'])
        else:
            crf = CRF(units=len(self.named_entities_) * 2 + 1, learn_mode='join', test_mode='viterbi',
                      kernel_regularizer=(l2(self.l2_kernel) if self.l2_kernel > 0.0 else None),
                      chain_regularizer=(l2(self.l2_chain) if self.l2_chain > 0.0 else None), name='crf_layer')
            nn_output = crf(nn_output)
            self.classifier_ = Model(nn_input, nn_output)
            self.classifier_.compile(optimizer='rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])
        if self.verbose:
            print_summary(self.classifier_)
            print('')
            print('Training is started...')
        tmp_file_name = self.get_temp_name()
        try:
            epoch_idx = 0
            best_f1 = None
            best_accuracy = None
            patience = 0
            MAX_PATIENCE = 5
            column_widths = [max(len(str(self.n_epochs)), len('Epoch')), max(8, len('F1-macro')),
                             max(8, len('Accuracy'))]
            if self.verbose:
                print('{0:>{1}}  {2:>{3}}  {4:>{5}}'.format('Epoch', column_widths[0], 'F1-macro', column_widths[1],
                                                            'Accuracy', column_widths[2]))
            lengths_of_texts_for_testing = [len(X_tokenized_test[idx]) for idx in range(len(X_tokenized_test))]
            while epoch_idx < self.n_epochs:
                for X_batch, y_batch in self.generate_batches(X_train, y_train, self.batch_size):
                    self.classifier_.train_on_batch(X_batch, y_batch)
                    del X_batch, y_batch
                epoch_idx += 1
                y_pred = None
                for X_batch in self.generate_batches(X_test, None, self.batch_size, shuffle=False):
                    y_batch = self.classifier_.predict_on_batch(X_batch)
                    if y_pred is None:
                        y_pred = y_batch.copy()
                    else:
                        y_pred = np.vstack((y_pred, y_batch))
                    del X_batch, y_batch
                cur_f1 = self.f1_macro(y_test, y_pred[:y_test.shape[0]], lengths_of_texts_for_testing)
                cur_accuracy = self.accuracy(y_test, y_pred[:y_test.shape[0]], lengths_of_texts_for_testing)
                del y_pred
                if best_f1 is None:
                    best_f1 = cur_f1
                    best_accuracy = cur_accuracy
                    self.classifier_.save_weights(tmp_file_name)
                    patience = 0
                elif cur_f1 > best_f1:
                    best_f1 = cur_f1
                    if best_accuracy < cur_accuracy:
                        best_accuracy = cur_accuracy
                    self.classifier_.save_weights(tmp_file_name)
                    patience = 0
                elif (abs(cur_f1 - best_f1) < K.epsilon()) and (cur_accuracy > best_accuracy):
                    best_f1 = cur_f1
                    if best_accuracy < cur_accuracy:
                        best_accuracy = cur_accuracy
                    self.classifier_.save_weights(tmp_file_name)
                    patience = 0
                else:
                    if best_accuracy < cur_accuracy:
                        best_accuracy = cur_accuracy
                    patience += 1
                if self.verbose:
                    print('{0:>{1}}  {2:>{3}.6f}  {4:>{5}.6f}'.format(epoch_idx, column_widths[0], cur_f1,
                                                                      column_widths[1], cur_accuracy, column_widths[2]))
                if patience >= MAX_PATIENCE:
                    if self.verbose:
                        print('Early stopping!')
                    break
            if patience < MAX_PATIENCE:
                if self.verbose:
                    print('Epochs number is exceeded.')
            if os.path.isfile(tmp_file_name):
                self.classifier_.load_weights(tmp_file_name)
        finally:
            if os.path.isfile(tmp_file_name):
                os.remove(tmp_file_name)
        if self.verbose:
            print('Training is finished...')
        return self

    def predict(self, X: Union[list, tuple, np.ndarray]) -> Union[list, tuple, np.ndarray]:
        self.check_params(**self.get_params(deep=False))
        self.check_X(X, 'X')
        check_is_fitted(self, ['classifier_', 'named_entities_', 'max_text_len_', 'embedding_size_'])
        self.update_elmo()
        X_tokenized = self.tokenize(X)
        y = None
        for X_batch in self.generate_batches(self.texts_to_X(X, X_tokenized, self.max_text_len_), None,
                                             self.batch_size, shuffle=False):
            y_batch = self.classifier_.predict_on_batch(X_batch)
            if y is None:
                y = y_batch.copy()
            else:
                y = np.vstack((y, y_batch))
            del X_batch, y_batch
        y = np.argmax(y[:len(X)], axis=-1)
        res = []
        for text_idx in range(y.shape[0]):
            ne_token_start = -1
            ne_type = None
            entities_in_text = []
            for token_idx in range(min(len(X_tokenized[text_idx]), y.shape[1])):
                if y[text_idx, token_idx] > 0:
                    ne_idx = (int(y[text_idx, token_idx]) - 1) // 2
                    if ne_type is None:
                        ne_type = self.named_entities_[ne_idx]
                        ne_token_start = token_idx
                    else:
                        if ne_type != self.named_entities_[ne_idx]:
                            start_char_idx = X_tokenized[text_idx][ne_token_start][0]
                            end_char_idx = X_tokenized[text_idx][token_idx - 1][0] + \
                                           X_tokenized[text_idx][token_idx - 1][1]
                            entities_in_text.append((ne_type, start_char_idx, end_char_idx - start_char_idx))
                            ne_type = self.named_entities_[ne_idx]
                            ne_token_start = token_idx
                else:
                    if ne_type is not None:
                        start_char_idx = X_tokenized[text_idx][ne_token_start][0]
                        end_char_idx = X_tokenized[text_idx][token_idx - 1][0] + X_tokenized[text_idx][token_idx - 1][1]
                        entities_in_text.append((ne_type, start_char_idx, end_char_idx - start_char_idx))
                        ne_token_start = -1
                        ne_type = None
            if ne_type is not None:
                token_idx = min(len(X_tokenized[text_idx]), y.shape[1])
                start_char_idx = X_tokenized[text_idx][ne_token_start][0]
                end_char_idx = X_tokenized[text_idx][token_idx - 1][0] + X_tokenized[text_idx][token_idx - 1][1]
                entities_in_text.append((ne_type, start_char_idx, end_char_idx - start_char_idx))
            res.append(tuple(entities_in_text))
        del y
        if isinstance(X, tuple):
            return tuple(res)
        if isinstance(X, np.ndarray):
            return np.array(res, dtype=object)
        return res

    def score(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray], sample_weight=None):
        self.check_params(**self.get_params(deep=False))
        check_is_fitted(self, ['classifier_', 'named_entities_', 'max_text_len_', 'embedding_size_'])
        self.check_X(X, 'X')
        named_entities = self.check_y(y, [len(X[idx]) for idx in range(len(X))], 'y')
        if not (set(named_entities) <= set(self.named_entities_)):
            raise ValueError('`y` is wrong! These entities are unknown: [{0}].'.format(
                ', '.join(sorted(list(set(named_entities) - set(self.named_entities_))))))
        self.update_elmo()
        X_tokenized = self.tokenize(X)
        y_true = self.labels_to_y(X, y, X_tokenized, self.max_text_len_)
        y_pred = None
        for X_batch in self.generate_batches(self.texts_to_X(X, X_tokenized, self.max_text_len_), None, self.batch_size,
                                             shuffle=False):
            y_batch = self.classifier_.predict_on_batch(X_batch)
            if y_pred is None:
                y_pred = y_batch.copy()
            else:
                y_pred = np.vstack((y_pred, y_batch))
            del X_batch, y_batch
        return self.f1_macro(y_true, y_pred[:y_true.shape[0]],
                             [len(X_tokenized[idx]) for idx in range(len(X_tokenized))])

    def tokenize(self, X: Union[list, tuple, np.ndarray]) -> List[tuple]:
        self.update_tokenizer()
        token_bounds_in_texts = []
        doc_idx = 0
        for idx in range(len(X)):
            token_bounds_in_cur_text = []
            start_pos = 0
            for cur_token in self.tokenizer_.international_tokenize(X[idx]):
                found_pos = X[idx].find(cur_token, start_pos)
                if found_pos < 0:
                    raise ValueError('Text `{0}` cannot be tokenized!'.format(X[idx]))
                if not (set(cur_token) <= {'_'}):
                    search_res = self.re_for_token_.search(cur_token)
                    if search_res is not None:
                        if (search_res.start() >= 0) and (search_res.end() >= 0):
                            token_bounds_in_cur_text.append((found_pos, len(cur_token)))
                start_pos = found_pos + len(cur_token)
            token_bounds_in_texts.append(tuple(token_bounds_in_cur_text))
            del token_bounds_in_cur_text
            doc_idx += 1
        return token_bounds_in_texts

    def texts_to_X(self, texts: Union[list, tuple, np.ndarray], token_bounds_in_all_texts: List[tuple],
                   max_text_len: int) -> np.ndarray:
        if self.cached:
            if self.CACHE is None:
                X = None
            else:
                if isinstance(texts, np.ndarray):
                    key = tuple(texts.tolist())
                elif not isinstance(texts, tuple):
                    key = tuple(texts)
                else:
                    key = texts
                if key in self.CACHE:
                    X = self.CACHE[key]
                else:
                    X = None
        else:
            X = None
        if X is None:
            embedding_size = None
            n_batches = int(math.ceil(len(texts) / float(self.batch_size)))
            sess = K.get_session()
            tokens_ph = tf.placeholder(shape=(None, None), dtype=tf.string, name='tokens')
            tokens_length_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='tokens_length')
            embeddings_of_texts = self.elmo_(
                inputs={
                    'tokens': tokens_ph,
                    'sequence_len': tokens_length_ph
                },
                signature='tokens',
                as_dict=True
            )['elmo']
            for batch_idx in range(n_batches):
                start_pos = batch_idx * self.batch_size
                end_pos = min(len(texts), (batch_idx + 1) * self.batch_size)
                texts_in_batch = []
                lengths_of_texts = []
                for text_idx in range(start_pos, end_pos):
                    new_text = [texts[text_idx][token_bounds[0]:(token_bounds[0] + token_bounds[1])]
                                for token_bounds in token_bounds_in_all_texts[text_idx]]
                    if len(new_text) > max_text_len:
                        new_text = new_text[:max_text_len]
                    lengths_of_texts.append(len(new_text))
                    while len(new_text) < max_text_len:
                        new_text.append('')
                    texts_in_batch.append(new_text)
                embeddings_of_texts_as_numpy = sess.run(
                    embeddings_of_texts,
                    feed_dict={
                        tokens_ph: texts_in_batch,
                        tokens_length_ph: lengths_of_texts
                    }
                )
                if embedding_size is None:
                    embedding_size = embeddings_of_texts_as_numpy.shape[2]
                if X is None:
                    X = np.zeros((len(texts), max_text_len, embedding_size), dtype=np.float32)
                for idx in range(end_pos - start_pos):
                    text_idx = start_pos + idx
                    for token_idx in range(min(max_text_len, len(token_bounds_in_all_texts[text_idx]))):
                        X[text_idx][token_idx] = embeddings_of_texts_as_numpy[idx][token_idx]
                del embeddings_of_texts_as_numpy, texts_in_batch, lengths_of_texts
            if self.cached:
                if isinstance(texts, np.ndarray):
                    key = tuple(texts.tolist())
                elif not isinstance(texts, tuple):
                    key = tuple(texts)
                else:
                    key = texts
                if self.CACHE is None:
                    self.CACHE = {key: X}
                else:
                    self.CACHE[key] = X
        return X

    def labels_to_y(self, texts: Union[list, tuple, np.ndarray], labels: Union[list, tuple, np.ndarray],
                    token_bounds_in_all_texts: List[tuple], max_text_len: int) -> np.ndarray:
        y = np.zeros((len(texts), max_text_len, len(self.named_entities_) * 2 + 1), dtype=np.float32)
        for text_idx in range(len(texts)):
            for token_idx in range(max_text_len):
                y[text_idx, token_idx, 0] = 1.0
            for ne_type, ne_start, ne_end in self.prepare_labels(len(texts[text_idx]),
                                                                 token_bounds_in_all_texts[text_idx], labels[text_idx]):
                if ne_start >= max_text_len:
                    continue
                if ne_end > max_text_len:
                    ne_end = max_text_len
                try:
                    ne_idx = self.named_entities_.index(ne_type)
                except ValueError as e:
                    raise ValueError(str(e) + ' `{0}` is unknown entity!'.format(ne_type))
                y[text_idx, ne_start, ne_idx * 2 + 1] = 1.0
                y[text_idx, ne_start, 0] = 0.0
                for token_idx in range(ne_start + 1, ne_end):
                    y[text_idx, token_idx, ne_idx * 2 + 2] = 1.0
                    y[text_idx, token_idx, 0] = 0.0
        return y

    def update_tokenizer(self):
        if not hasattr(self, 'tokenizer_'):
            self.tokenizer_ = NISTTokenizer()
        if not hasattr(self, 're_for_token_'):
            self.re_for_token_ = re.compile('[\w]+', re.U)

    def update_elmo(self):
        if not hasattr(self, 'elmo_'):
            self.elmo_ = hub.Module(self.elmo_name, trainable=True)

    def get_params(self, deep=True):
        return {'elmo_name': self.elmo_name, 'n_units': self.n_units, 'dropout': self.dropout,
                'recurrent_dropout': self.recurrent_dropout, 'l2_kernel': self.l2_kernel, 'l2_chain': self.l2_chain,
                'n_epochs': self.n_epochs, 'validation_part': self.validation_part, 'verbose': self.verbose,
                'batch_size': self.batch_size, 'use_lstm': self.use_lstm, 'use_crf': self.use_crf,
                'cached': self.cached}

    def set_params(self, **params):
        for parameter, value in params.items():
            self.__setattr__(parameter, value)
        return self

    def load_weights(self, weights_of_classifier: Union[bytearray, bytes]):
        if (not isinstance(weights_of_classifier, bytearray)) and (not isinstance(weights_of_classifier, bytes)):
            raise ValueError('`weights_of_classifier` is wrong! Expected an array of bytes, bot got a `{0}`!'.format(
                type(weights_of_classifier)))
        tmp_weights_name = self.get_temp_name()
        try:
            with open(tmp_weights_name, 'wb') as fp:
                fp.write(weights_of_classifier)
            nn_input = Input(shape=(self.max_text_len_, self.embedding_size_), name='input_of_tagger')
            nn_output = Masking(mask_value=0.0, input_shape=(self.max_text_len_, self.embedding_size_),
                                name='masking_layer')(nn_input)
            if self.use_lstm:
                nn_output = Bidirectional(LSTM(self.n_units, return_sequences=True, dropout=self.dropout,
                                               recurrent_dropout=self.recurrent_dropout, name='lstm_layer'),
                                          merge_mode='ave', name='BiLSTM_layer')(nn_output)
                if self.use_crf:
                    crf = CRF(units=len(self.named_entities_) * 2 + 1, learn_mode='join', test_mode='viterbi',
                              kernel_regularizer=(l2(self.l2_kernel) if self.l2_kernel > 0.0 else None),
                              chain_regularizer=(l2(self.l2_chain) if self.l2_chain > 0.0 else None),
                              name='crf_layer')
                    nn_output = crf(nn_output)
                    self.classifier_ = Model(nn_input, nn_output)
                    self.classifier_.load_weights(tmp_weights_name)
                    self.classifier_.compile(optimizer='rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])
                else:
                    nn_output = TimeDistributed(Dense(len(self.named_entities_) * 2 + 1,
                                                      activation='softmax', name='dense_layer'),
                                                name='time_distr')(nn_output)
                    self.classifier_ = Model(nn_input, nn_output)
                    self.classifier_.load_weights(tmp_weights_name)
                    self.classifier_.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                                             metrics=['categorical_accuracy'])
            else:
                crf = CRF(units=len(self.named_entities_) * 2 + 1, learn_mode='join', test_mode='viterbi',
                          kernel_regularizer=(l2(self.l2_kernel) if self.l2_kernel > 0.0 else None),
                          chain_regularizer=(l2(self.l2_chain) if self.l2_chain > 0.0 else None), name='crf_layer')
                nn_output = crf(nn_output)
                self.classifier_ = Model(nn_input, nn_output)
                self.classifier_.load_weights(tmp_weights_name)
                self.classifier_.compile(optimizer='rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])
            self.classifier_._make_predict_function()
            self.classifier_._make_test_function()
            self.classifier_._make_train_function()
        finally:
            if os.path.isfile(tmp_weights_name):
                os.remove(tmp_weights_name)

    def dump_weights(self) -> Union[bytearray, bytes]:
        check_is_fitted(self, ['classifier_', 'named_entities_', 'max_text_len_', 'embedding_size_'])
        tmp_weights_name = self.get_temp_name()
        try:
            if os.path.isfile(tmp_weights_name):
                os.remove(tmp_weights_name)
            self.classifier_.save_weights(tmp_weights_name)
            with open(tmp_weights_name, 'rb') as fp:
                weights_of_classifier = fp.read()
            os.remove(tmp_weights_name)
        finally:
            if os.path.isfile(tmp_weights_name):
                os.remove(tmp_weights_name)
        return weights_of_classifier

    def dump_all(self):
        try:
            check_is_fitted(self, ['classifier_', 'named_entities_', 'max_text_len_', 'embedding_size_'])
            is_trained = True
        except:
            is_trained = False
        params = self.get_params(True)
        if is_trained:
            params['named_entities_'] = copy.deepcopy(self.named_entities_)
            params['max_text_len_'] = self.max_text_len_
            params['embedding_size_'] = self.embedding_size_
            params['weights'] = self.dump_weights()
        return params

    def load_all(self, new_params):
        if not isinstance(new_params, dict):
            raise ValueError('`new_params` is wrong! Expected `{0}`, got `{1}`.'.format(type({0: 1}), type(new_params)))
        self.check_params(**new_params)
        expected_param_keys = {'elmo_name', 'n_units', 'dropout', 'recurrent_dropout', 'l2_kernel', 'l2_chain',
                               'n_epochs', 'validation_part', 'verbose', 'batch_size', 'use_crf', 'use_lstm', 'cached'}
        params_after_training = {'weights', 'named_entities_', 'max_text_len_', 'embedding_size_'}
        is_fitted = len(set(new_params.keys())) > len(expected_param_keys)
        if is_fitted:
            if set(new_params.keys()) != (expected_param_keys | params_after_training):
                raise ValueError('`new_params` does not contain all expected keys!')
        self.batch_size = new_params['batch_size']
        self.elmo_name = new_params['elmo_name']
        self.n_units = new_params['n_units']
        self.dropout = new_params['dropout']
        self.recurrent_dropout = new_params['recurrent_dropout']
        self.l2_kernel = new_params['l2_kernel']
        self.l2_chain = new_params['l2_chain']
        self.n_epochs = new_params['n_epochs']
        self.validation_part = new_params['validation_part']
        self.verbose = new_params['verbose']
        self.use_crf = new_params['use_crf']
        self.use_lstm = new_params['use_lstm']
        self.cached = new_params['cached']
        if is_fitted:
            if not isinstance(new_params['named_entities_'], tuple):
                raise ValueError('`named_entities_` is wrong! Expected `{0}`, got `{1}`.'.format(
                    type((1, 2)), type(new_params['named_entities_'])))
            if len(new_params['named_entities_']) < 1:
                raise ValueError('`named_entities_` is empty!')
            if not isinstance(new_params['max_text_len_'], int):
                raise ValueError('`max_text_len_` is wrong! Expected `{0}`, got `{1}`.'.format(
                    type(2), type(new_params['max_text_len_'])))
            if new_params['max_text_len_'] < 1:
                raise ValueError('`max_text_len_` is wrong! Expected a positive integer value, but {0} is not '
                                 'positive.'.format(new_params['max_text_len_']))
            if not isinstance(new_params['embedding_size_'], int):
                raise ValueError('`embedding_size_` is wrong! Expected `{0}`, got `{1}`.'.format(
                    type(2), type(new_params['embedding_size_'])))
            if new_params['embedding_size_'] < 1:
                raise ValueError('`embedding_size_` is wrong! Expected a positive integer value, but {0} is not '
                                 'positive.'.format(new_params['embedding_size_']))
            self.named_entities_ = new_params['named_entities_']
            self.max_text_len_ = new_params['max_text_len_']
            self.embedding_size_ = new_params['embedding_size_']
            K.get_session()
            self.load_weights(new_params['weights'])
        return self

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.load_all(self.dump_all())
        return result

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.load_all(self.dump_all())
        return result

    def __getstate__(self):
        return self.dump_all()

    def __setstate__(self, state):
        self.load_all(state)

    @staticmethod
    def prepare_labels(text_len: int, token_bounds: tuple,
                       labels_for_text: Union[list, tuple]) -> List[Tuple[str, int, int]]:
        indices = np.full(shape=(text_len,), fill_value=-1, dtype=np.int32)
        token_counter = 0
        for cur in token_bounds:
            for idx in range(cur[0], cur[0] + cur[1]):
                indices[idx] = token_counter
            token_counter += 1
        prepared = []
        for cur_label in labels_for_text:
            idx = cur_label[1]
            while idx < text_len:
                if indices[idx] >= 0:
                    break
                idx += 1
            if idx < (cur_label[1] + cur_label[2]):
                start_token_idx = indices[idx]
                idx = cur_label[1] + cur_label[2] - 1
                while idx > cur_label[1]:
                    if indices[idx] >= 0:
                        break
                    idx -= 1
                end_token_idx = indices[idx]
                prepared.append((cur_label[0].upper(), start_token_idx, end_token_idx + 1))
        return prepared

    @staticmethod
    def strip_token_bounds(source_text, token_start, token_len) -> Union[Tuple[int, int], None]:
        n = len(source_text)
        token_start_ = token_start
        token_end_ = token_start + token_len - 1
        while token_start_ < n:
            if not source_text[token_start_].isspace():
                break
            token_start_ += 1
        if (token_start_ >= n) or (token_start_ > token_end_):
            return None
        while token_end_ > token_start_:
            if not source_text[token_end_].isspace():
                break
            token_end_ -= 1
        return token_start_, token_end_ - token_start_ + 1

    @staticmethod
    def f1_macro(y_true: np.ndarray, y_pred: np.ndarray, lengths_of_texts: List[int]) -> float:
        if not isinstance(y_true, np.ndarray):
            raise ValueError('`y_true` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(np.array([1, 2])), type(y_true)))
        if not isinstance(y_pred, np.ndarray):
            raise ValueError('`y_pred` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(np.array([1, 2])), type(y_pred)))
        if y_true.ndim != 3:
            raise ValueError('`y_true` is wrong! Expected a 3-D array, but got a {0}-D one.'.format(y_true.ndim))
        if y_true.shape != y_pred.shape:
            raise ValueError('`y_pred` does not correspond to `y_true`. {0} != {1}.'.format(y_pred.shape, y_true.shape))
        if len(lengths_of_texts) != y_true.shape[0]:
            raise ValueError('`lengths_of_texts` does not correspond to `y_true`. {0} != {1}.'.format(
                len(lengths_of_texts), y_true.shape[0]))
        y_true_ = np.argmax(y_true[0], axis=-1)[0:lengths_of_texts[0]]
        y_pred_ = np.argmax(y_pred[0], axis=-1)[0:lengths_of_texts[0]]
        for text_idx in range(1, len(lengths_of_texts)):
            y_true_ = np.concatenate((y_true_, np.argmax(y_true[text_idx], axis=-1)[0:lengths_of_texts[text_idx]]))
            y_pred_ = np.concatenate((y_pred_, np.argmax(y_pred[text_idx], axis=-1)[0:lengths_of_texts[text_idx]]))
        f1 = 0.0
        n = 0
        for class_idx in range(1, y_true.shape[2]):
            y_true_bin = (y_true_ == class_idx).astype(np.int32)
            y_pred_bin = (y_pred_ == class_idx).astype(np.int32)
            if (y_true_bin.max() > 0) or (y_pred_bin.max() > 0):
                f1 += f1_score(y_true_bin, y_pred_bin, average='binary')
                n += 1
        if n == 0:
            return 0.0
        return f1 / float(n)

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray, lengths_of_texts: List[int]) -> float:
        if not isinstance(y_true, np.ndarray):
            raise ValueError('`y_true` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(np.array([1, 2])), type(y_true)))
        if not isinstance(y_pred, np.ndarray):
            raise ValueError('`y_pred` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(np.array([1, 2])), type(y_pred)))
        if y_true.ndim != 3:
            raise ValueError('`y_true` is wrong! Expected a 3-D array, but got a {0}-D one.'.format(y_true.ndim))
        if y_true.shape != y_pred.shape:
            raise ValueError('`y_pred` does not correspond to `y_true`. {0} != {1}.'.format(y_pred.shape, y_true.shape))
        if len(lengths_of_texts) != y_true.shape[0]:
            raise ValueError('`lengths_of_texts` does not correspond to `y_true`. {0} != {1}.'.format(
                len(lengths_of_texts), y_true.shape[0]))
        n = 0
        n_correct = 0
        for sample_idx in range(len(lengths_of_texts)):
            for token_idx in range(lengths_of_texts[sample_idx]):
                n += 1
                if y_true[sample_idx][token_idx].argmax() == y_pred[sample_idx][token_idx].argmax():
                    n_correct += 1
        if n == 0:
            return 0.0
        return n_correct / float(n)

    @staticmethod
    def generate_batches(X: np.ndarray, y: Union[np.ndarray, None], batch_size: int, shuffle: bool=True):
        n_batches = math.ceil(X.shape[0] / float(batch_size))
        indices = np.arange(0, X.shape[0], dtype=np.int32)
        if X.shape[0] < (n_batches * batch_size):
            indices = np.concatenate((indices, np.random.choice(indices, n_batches * batch_size - X.shape[0])))
        if shuffle:
            np.random.shuffle(indices)
        for batch_idx in range(n_batches):
            indices_in_batch = indices[(batch_idx * batch_size):((batch_idx + 1) * batch_size)]
            if y is None:
                yield X[indices_in_batch]
            else:
                yield X[indices_in_batch], y[indices_in_batch]
            del indices_in_batch
        del indices

    @staticmethod
    def check_X(X: Union[list, tuple, np.ndarray], name_X: str='X'):
        if (not isinstance(X, list)) and (not isinstance(X, tuple)) and (not isinstance(X, np.ndarray)):
            raise ValueError('`{0}` is wrong! Expected `{1}`, `{2}` or `{3}`, but got `{4}`.'.format(
                name_X, type([1, 2]), type((1, 2)), type(np.array([1, 2])), type(X)))
        if isinstance(X, np.ndarray):
            if X.ndim != 1:
                raise ValueError('`{0}` is wrong! Expected 1-D array, but got {1}-D one.'.format(name_X, X.ndim))

    @staticmethod
    def check_y(y: Union[list, tuple, np.ndarray], lengths_of_texts: Union[list, tuple], name_y: str='y') -> tuple:
        if (not isinstance(y, list)) and (not isinstance(y, tuple)) and (not isinstance(y, np.ndarray)):
            raise ValueError('`{0}` is wrong! Expected `{1}`, `{2}` or `{3}`, but got `{4}`.'.format(
                name_y, type([1, 2]), type((1, 2)), type(np.array([1, 2])), type(y)))
        if isinstance(y, np.ndarray):
            if y.ndim != 1:
                raise ValueError('`{0}` is wrong! Expected 1-D array, but got {1}-D one.'.format(name_y, y.ndim))
        if len(y) != len(lengths_of_texts):
            raise ValueError('`{0}` does not correspond to sequence of input texts. Number of input texts is {1}, but '
                             'length of `{0}` is {2}.'.format(name_y, len(lengths_of_texts), len(y)))
        named_entities = set()
        for sample_idx in range(len(y)):
            if (not isinstance(y[sample_idx], list)) and (not isinstance(y[sample_idx], tuple)):
                raise ValueError('Labels for sample {0} are wrong! Expected `{1}` or `{2}`, but got `{3}`.'.format(
                    sample_idx, type([1, 2]), type((1, 2)), type(y[sample_idx])))
            for ne_idx in range(len(y[sample_idx])):
                if (not isinstance(y[sample_idx][ne_idx], list)) and (not isinstance(y[sample_idx][ne_idx], tuple)):
                    raise ValueError('Description of tag {0} for sample {1} is wrong! Expected `{2}` or `{3}`, but '
                                     'got `{4}`.'.format(ne_idx, sample_idx, type([1, 2]), type((1, 2)),
                                                         type(y[sample_idx][ne_idx])))
                if len(y[sample_idx][ne_idx]) != 3:
                    raise ValueError('Description of tag {0} for sample {1} is wrong! Expected a 3-element sequence, '
                                     'but got a {2}-element one.'.format(ne_idx, sample_idx,
                                                                         len(y[sample_idx][ne_idx])))
                if (not hasattr(y[sample_idx][ne_idx][0], 'upper')) or (not hasattr(y[sample_idx][ne_idx][0], 'strip')):
                    raise ValueError('Description of tag {0} for sample {1} is wrong! First element (entity type) must'
                                     ' be a string object, but it is `{2}`.'.format(ne_idx, sample_idx,
                                                                                    type(y[sample_idx][ne_idx][0])))
                ne_type = y[sample_idx][ne_idx][0].strip().upper()
                if (len(ne_type) == 0) or (ne_type == 'O'):
                    raise ValueError('Description of tag {0} for sample {1} is wrong! First element (entity type) is '
                                     'incorrect.'.format(ne_idx, sample_idx))
                named_entities.add(ne_type)
                if not isinstance(y[sample_idx][ne_idx][1], int):
                    raise ValueError('Description of tag {0} for sample {1} is wrong! Second element (entity start) '
                                     'must be `{2}`, but it is `{3}`.'.format(ne_idx, sample_idx, type(3),
                                                                              type(y[sample_idx][ne_idx][1])))
                if not isinstance(y[sample_idx][ne_idx][2], int):
                    raise ValueError('Description of tag {0} for sample {1} is wrong! Third element (entity length) '
                                     'must be `{2}`, but it is `{3}`.'.format(ne_idx, sample_idx, type(3),
                                                                              type(y[sample_idx][ne_idx][2])))
                if (y[sample_idx][ne_idx][1] < 0) or \
                        ((y[sample_idx][ne_idx][1] + y[sample_idx][ne_idx][2]) > lengths_of_texts[sample_idx]) or \
                        (y[sample_idx][ne_idx][2] < 1):
                    raise ValueError('Description of tag {0} for sample {1} is wrong! ({2}, {3}) is inadmissible '
                                     'value of entity bounds.'.format(ne_idx, sample_idx, y[sample_idx][ne_idx][1],
                                                                      y[sample_idx][ne_idx][2]))
        return tuple(sorted(list(named_entities)))

    @staticmethod
    def check_params(**kwargs):
        if 'batch_size' not in kwargs:
            raise ValueError('`batch_size` is not found!')
        if np.dtype(type(kwargs['batch_size'])).kind != 'i':
            raise ValueError('`batch_size` must be `{0}`, not `{1}`.'.format(type(10), type(kwargs['batch_size'])))
        if kwargs['batch_size'] < 1:
            raise ValueError('`batch_size` must be a positive number! {0} is not positive.'.format(
                kwargs['batch_size']))
        if 'n_units' not in kwargs:
            raise ValueError('`n_units` is not found!')
        if np.dtype(type(kwargs['n_units'])).kind != 'i':
            raise ValueError('`n_units` must be `{0}`, not `{1}`.'.format(type(10), type(kwargs['n_units'])))
        if kwargs['n_units'] < 1:
            raise ValueError('`n_units` must be a positive number! {0} is not positive.'.format(kwargs['n_units']))
        if 'n_epochs' not in kwargs:
            raise ValueError('`n_epochs` is not found!')
        if np.dtype(type(kwargs['n_epochs'])).kind != 'i':
            raise ValueError('`n_epochs` must be `{0}`, not `{1}`.'.format(type(10), type(kwargs['n_epochs'])))
        if kwargs['n_epochs'] < 1:
            raise ValueError('`n_epochs` must be a positive number! {0} is not positive.'.format(kwargs['n_epochs']))
        if 'dropout' not in kwargs:
            raise ValueError('`dropout` is not found!')
        if np.dtype(type(kwargs['dropout'])).kind not in {'f', 'i'}:
            raise ValueError('`dropout` must be `{0}`, not `{1}`.'.format(type(10.3), type(kwargs['dropout'])))
        if (kwargs['dropout'] < 0.0) or (kwargs['dropout'] >= 1.0):
            raise ValueError('`dropout` must be a floating-point number in the interval [0.0, 1.0)! {0} is not in '
                             'this interval'.format(kwargs['dropout']))
        if 'recurrent_dropout' not in kwargs:
            raise ValueError('`recurrent_dropout` is not found!')
        if np.dtype(type(kwargs['recurrent_dropout'])).kind not in {'f', 'i'}:
            raise ValueError('`recurrent_dropout` must be `{0}`, not `{1}`.'.format(
                type(10.3), type(kwargs['recurrent_dropout'])))
        if (kwargs['recurrent_dropout'] < 0.0) or (kwargs['recurrent_dropout'] >= 1.0):
            raise ValueError('`recurrent_dropout` must be a floating-point number in the interval [0.0, 1.0)! {0} is '
                             'not in this interval'.format(kwargs['recurrent_dropout']))
        if 'l2_kernel' not in kwargs:
            raise ValueError('`l2_kernel` is not found!')
        if np.dtype(type(kwargs['l2_kernel'])).kind not in {'f', 'i'}:
            raise ValueError('`l2_kernel` must be `{0}`, not `{1}`.'.format(type(10.3), type(kwargs['l2_kernel'])))
        if kwargs['l2_kernel'] < 0.0:
            raise ValueError('`l2_kernel` must be a non-negative floating-point number! {0} is negative number'.format(
                kwargs['l2_kernel']))
        if 'l2_chain' not in kwargs:
            raise ValueError('`l2_chain` is not found!')
        if np.dtype(type(kwargs['l2_chain'])).kind not in {'f', 'i'}:
            raise ValueError('`l2_chain` must be `{0}`, not `{1}`.'.format(type(10.3), type(kwargs['l2_chain'])))
        if kwargs['l2_chain'] < 0.0:
            raise ValueError('`l2_chain` must be a non-negative floating-point number! {0} is negative number'.format(
                kwargs['l2_chain']))
        if 'validation_part' not in kwargs:
            raise ValueError('`validation_part` is not found!')
        if np.dtype(type(kwargs['validation_part'])).kind not in {'f', 'i'}:
            raise ValueError('`validation_part` must be `{0}`, not `{1}`.'.format(
                type(10.3), type(kwargs['validation_part'])))
        if (kwargs['validation_part'] <= 0.0) or (kwargs['validation_part'] >= 1.0):
            raise ValueError('`validation_part` must be a floating-point number in the interval (0.0, 1.0)! {0} is '
                             'not in this interval'.format(kwargs['validation_part']))
        if 'verbose' not in kwargs:
            raise ValueError('`verbose` is not found!')
        if np.dtype(type(kwargs['verbose'])).kind not in {'b', 'i'}:
            raise ValueError('`verbose` must be `{0}`, not `{1}`.'.format(type(True), type(kwargs['verbose'])))
        if 'cached' not in kwargs:
            raise ValueError('`cached` is not found!')
        if np.dtype(type(kwargs['cached'])).kind not in {'b', 'i'}:
            raise ValueError('`cached` must be `{0}`, not `{1}`.'.format(type(True), type(kwargs['cached'])))
        if 'use_crf' not in kwargs:
            raise ValueError('`use_crf` is not found!')
        if np.dtype(type(kwargs['use_crf'])).kind not in {'b', 'i'}:
            raise ValueError('`use_crf` must be `{0}`, not `{1}`.'.format(type(True), type(kwargs['use_crf'])))
        if 'use_lstm' not in kwargs:
            raise ValueError('`use_lstm` is not found!')
        if np.dtype(type(kwargs['use_lstm'])).kind not in {'b', 'i'}:
            raise ValueError('`use_lstm` must be `{0}`, not `{1}`.'.format(type(True), type(kwargs['use_lstm'])))
        if (not kwargs['use_lstm']) and (not kwargs['use_crf']):
            raise ValueError('`use_lstm` or `use_crf` must be True.')

    @staticmethod
    def get_temp_name() -> str:
        fp = tempfile.NamedTemporaryFile(delete=True)
        file_name = fp.name
        fp.close()
        del fp
        return file_name

    @staticmethod
    def bag_of_named_entities(labels_of_text: tuple, possible_named_entities: set) -> str:
        return ' '.join(sorted(list(set(filter(lambda it: it in possible_named_entities,
                                               map(lambda cur_label: cur_label[0].upper(), labels_of_text))))))

    @staticmethod
    def select_main_named_entities(y: Union[list, tuple, np.ndarray], min_freq: int) -> Set[str]:
        frequencies_of_ne = dict()
        for sample_idx in range(len(y)):
            for cur_ne in y[sample_idx]:
                cur_ne_type = cur_ne[0].upper()
                frequencies_of_ne[cur_ne_type] = frequencies_of_ne.get(cur_ne_type, 0) + 1
        min_freq_ = min_freq
        possible_named_entities = set(filter(lambda it: frequencies_of_ne[it] >= min_freq_, frequencies_of_ne.keys()))
        while len(possible_named_entities) > 0:
            frequencies_of_ne_bag = dict()
            for sample_idx in range(len(y)):
                ne_bag = NeuroTagger.bag_of_named_entities(y[sample_idx], possible_named_entities)
                frequencies_of_ne_bag[ne_bag] = frequencies_of_ne_bag.get(ne_bag, 0) + 1
            if all(map(lambda ne_bag: frequencies_of_ne_bag[ne_bag] >= min_freq, frequencies_of_ne_bag.keys())):
                break
            min_freq_ += 1
            possible_named_entities = set(filter(lambda it: frequencies_of_ne[it] >= min_freq_,
                                                 frequencies_of_ne.keys()))
        return possible_named_entities

    @staticmethod
    def stratified_kfold(X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray],
                         k: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        possible_named_entities = NeuroTagger.select_main_named_entities(y, k)
        if len(possible_named_entities) == 0:
            raise ValueError('The specified data cannot be splitted!')
        skf = StratifiedKFold(n_splits=k, shuffle=True)
        indices_for_cv = [
            (train_index, test_index) for train_index, test_index in skf.split(
                X, list(map(lambda it: NeuroTagger.bag_of_named_entities(it, possible_named_entities), y))
            )
        ]
        return indices_for_cv

    @staticmethod
    def stratified_split(X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray],
                         validation_part: float) -> Tuple[np.ndarray, np.ndarray]:
        possible_named_entities = NeuroTagger.select_main_named_entities(y, 2)
        if len(possible_named_entities) == 0:
            raise ValueError('The specified data cannot be splitted!')
        sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_part)
        splits = [
            (train_index, test_index) for train_index, test_index in sss.split(
                X, list(map(lambda it: NeuroTagger.bag_of_named_entities(it, possible_named_entities), y))
            )
        ]
        train_index = splits[0][0]
        test_index = splits[0][1]
        return train_index, test_index

    @staticmethod
    def print_info_about_labels(labels: List[tuple]):
        print('Number of texts is {0}.'.format(len(labels)))
        frequencies_of_named_entities = dict()
        for cur_sample in labels:
            for cur_ne in cur_sample:
                cur_ne_type = cur_ne[0].upper()
                frequencies_of_named_entities[cur_ne_type] = frequencies_of_named_entities.get(cur_ne_type, 0) + 1
        named_entities = sorted(list(frequencies_of_named_entities.keys()))
        max_width_of_ne = len(named_entities[0])
        max_width_of_value = len(str(frequencies_of_named_entities[named_entities[0]]))
        for cur_ne_type in named_entities:
            if len(cur_ne_type) > max_width_of_ne:
                max_width_of_ne = len(cur_ne_type)
            value = str(frequencies_of_named_entities[cur_ne_type])
            if len(value) > max_width_of_value:
                max_width_of_value = len(value)
        print('Named entities:')
        for cur_ne_type in named_entities:
            print(
                '  {0:>{1}}\t{2:>{3}}'.format(cur_ne_type, max_width_of_ne, frequencies_of_named_entities[cur_ne_type],
                                              max_width_of_value))
        print('')
