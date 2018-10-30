import codecs
import os
import pickle
import re
from requests import get
import sys
import tarfile
from typing import List, Tuple
import unittest

import numpy as np

try:
    from neuro_tagger.neuro_tagger import NeuroTagger
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from neuro_tagger.neuro_tagger import NeuroTagger


class TestNeuroTagger(unittest.TestCase):
    elmo_model_name = None

    @classmethod
    def setUpClass(cls):
        cls.elmo_model_name = TestNeuroTagger.load_russian_elmo(
            url='http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz',
            target_dir=os.path.join(os.path.dirname(__file__), 'testdata', 'elmo_ru')
        )

    def setUp(self):
        self.data_set_name = os.path.join(os.path.dirname(__file__), 'testdata', 'labeled_data_for_testing.txt')
        self.model_name = os.path.join(os.path.dirname(__file__), 'testdata', 'russian_ner.pkl')
        self.tagger = NeuroTagger(
            elmo_name=self.elmo_model_name,
            n_units=16, dropout=0.5, recurrent_dropout=0.0, l2_kernel=1e-3, l2_chain=0.0, n_epochs=10,
            validation_part=(1.0 / 6.0), batch_size=2, verbose=True, use_crf=True, use_lstm=True
        )
        self.texts, self.labels = self.load_dataset(
            os.path.join(os.path.dirname(__file__), 'testdata', 'labeled_data_for_testing.txt')
        )

    def tearDown(self):
        if os.path.isfile(self.model_name):
            os.remove(self.model_name)
        if hasattr(self, 'tagger'):
            del self.tagger

    def test_create(self):
        self.assertIsInstance(self.tagger, NeuroTagger)
        self.assertTrue(hasattr(self.tagger, 'elmo_name'))
        self.assertTrue(hasattr(self.tagger, 'n_units'))
        self.assertTrue(hasattr(self.tagger, 'dropout'))
        self.assertTrue(hasattr(self.tagger, 'recurrent_dropout'))
        self.assertTrue(hasattr(self.tagger, 'l2_kernel'))
        self.assertTrue(hasattr(self.tagger, 'l2_chain'))
        self.assertTrue(hasattr(self.tagger, 'n_epochs'))
        self.assertTrue(hasattr(self.tagger, 'validation_part'))
        self.assertTrue(hasattr(self.tagger, 'batch_size'))
        self.assertTrue(hasattr(self.tagger, 'verbose'))
        self.assertTrue(hasattr(self.tagger, 'use_crf'))
        self.assertTrue(hasattr(self.tagger, 'use_lstm'))
        self.assertEqual(self.tagger.elmo_name, self.elmo_model_name)
        self.assertEqual(self.tagger.n_units, 16)
        self.assertAlmostEqual(self.tagger.dropout, 0.5)
        self.assertAlmostEqual(self.tagger.recurrent_dropout, 0.0)
        self.assertAlmostEqual(self.tagger.l2_kernel, 1e-3)
        self.assertAlmostEqual(self.tagger.l2_chain, 0.0)
        self.assertEqual(self.tagger.n_epochs, 10)
        self.assertAlmostEqual(self.tagger.validation_part, 1.0 / 6.0)
        self.assertEqual(self.tagger.batch_size, 2)
        self.assertTrue(self.tagger.verbose)
        self.assertTrue(self.tagger.use_crf)
        self.assertTrue(self.tagger.use_lstm)
        self.assertFalse(hasattr(self.tagger, 'tokenizer_'))
        self.assertFalse(hasattr(self.tagger, 'elmo_'))
        self.assertFalse(hasattr(self.tagger, 'classifier_'))
        self.assertFalse(hasattr(self.tagger, 'named_entities_'))
        self.assertFalse(hasattr(self.tagger, 'max_text_len_'))
        self.assertFalse(hasattr(self.tagger, 'embedding_size_'))

    def test_serialize_unfitted(self):
        with open(self.model_name, 'wb') as fp:
            pickle.dump(self.tagger, fp)
        with open(self.model_name, 'rb') as fp:
            other_tagger = pickle.load(fp)
        self.assertIsInstance(other_tagger, NeuroTagger)
        self.assertTrue(hasattr(other_tagger, 'elmo_name'))
        self.assertTrue(hasattr(other_tagger, 'n_units'))
        self.assertTrue(hasattr(other_tagger, 'dropout'))
        self.assertTrue(hasattr(other_tagger, 'recurrent_dropout'))
        self.assertTrue(hasattr(other_tagger, 'l2_kernel'))
        self.assertTrue(hasattr(other_tagger, 'l2_chain'))
        self.assertTrue(hasattr(other_tagger, 'n_epochs'))
        self.assertTrue(hasattr(other_tagger, 'validation_part'))
        self.assertTrue(hasattr(other_tagger, 'batch_size'))
        self.assertTrue(hasattr(other_tagger, 'verbose'))
        self.assertTrue(hasattr(other_tagger, 'use_crf'))
        self.assertTrue(hasattr(other_tagger, 'use_lstm'))
        self.assertEqual(other_tagger.elmo_name, self.elmo_model_name)
        self.assertEqual(other_tagger.n_units, 16)
        self.assertAlmostEqual(other_tagger.dropout, 0.5)
        self.assertAlmostEqual(other_tagger.recurrent_dropout, 0.0)
        self.assertAlmostEqual(other_tagger.l2_kernel, 1e-3)
        self.assertAlmostEqual(other_tagger.l2_chain, 0.0)
        self.assertEqual(other_tagger.n_epochs, 10)
        self.assertAlmostEqual(other_tagger.validation_part, 1.0 / 6.0)
        self.assertEqual(other_tagger.batch_size, 2)
        self.assertTrue(other_tagger.verbose)
        self.assertTrue(other_tagger.use_crf)
        self.assertTrue(other_tagger.use_lstm)
        self.assertFalse(hasattr(other_tagger, 'tokenizer_'))
        self.assertFalse(hasattr(other_tagger, 'elmo_'))
        self.assertFalse(hasattr(other_tagger, 'classifier_'))
        self.assertFalse(hasattr(other_tagger, 'named_entities_'))
        self.assertFalse(hasattr(other_tagger, 'max_text_len_'))
        self.assertFalse(hasattr(other_tagger, 'embedding_size_'))

    def test_tokenize_1(self):
        s = ['Мама мыла раму.', 'Папа мыл синхрофазотрон!']
        true_tokens = [
            ((0, 4), (5, 4), (10, 4)),
            ((0, 4), (5, 3), (9, 14))
        ]
        predicted_tokens = self.tagger.tokenize(s)
        self.assertIsInstance(predicted_tokens, list)
        self.assertEqual(true_tokens, predicted_tokens)
        self.assertTrue(hasattr(self.tagger, 'tokenizer_'))

    def test_tokenize_2(self):
        s = ('Мама мыла раму.', 'Папа мыл синхрофазотрон!')
        true_tokens = [
            ((0, 4), (5, 4), (10, 4)),
            ((0, 4), (5, 3), (9, 14))
        ]
        predicted_tokens = self.tagger.tokenize(s)
        self.assertIsInstance(predicted_tokens, list)
        self.assertEqual(true_tokens, predicted_tokens)
        self.assertTrue(hasattr(self.tagger, 'tokenizer_'))

    def test_tokenize_3(self):
        s = np.array(['Мама мыла раму.', 'Папа мыл синхрофазотрон!'], dtype=object)
        true_tokens = [
            ((0, 4), (5, 4), (10, 4)),
            ((0, 4), (5, 3), (9, 14))
        ]
        predicted_tokens = self.tagger.tokenize(s)
        self.assertIsInstance(predicted_tokens, list)
        self.assertEqual(true_tokens, predicted_tokens)
        self.assertTrue(hasattr(self.tagger, 'tokenizer_'))

    def test_tokenize_4(self):
        s = []
        true_tokens = []
        predicted_tokens = self.tagger.tokenize(s)
        self.assertIsInstance(predicted_tokens, list)
        self.assertEqual(true_tokens, predicted_tokens)
        self.assertTrue(hasattr(self.tagger, 'tokenizer_'))

    def test_texts_to_X_1(self):
        source_texts = [
            'Мама мыла раму.',
            'Папа мыл синхрофазотрон!',
            'Скажи-ка, дядя, ведь недаром Москва?',
            'инопланетяне скоро захватят нашу планету!!!'
        ]
        token_bounds = [
            ((0, 4), (5, 4), (10, 4)),
            ((0, 4), (5, 3), (9, 14)),
            ((0, 8), (10, 4), (16, 4), (21, 7), (29, 6)),
            ((0, 12), (13, 5), (19, 8), (28, 4), (33, 7))
        ]
        n = 10
        self.tagger.update_elmo()
        X = self.tagger.texts_to_X(source_texts, token_bounds, n)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.ndim, 3)
        self.assertEqual(X.shape[0], 4)
        self.assertEqual(X.shape[1], n)
        self.assertEqual(X.shape[2], 1024)
        for text_idx in range(len(source_texts)):
            n_tokens = len(token_bounds[text_idx])
            for token_idx in range(min(n_tokens, n)):
                self.assertGreater(np.linalg.norm(X[text_idx][token_idx]), 0.0)
            if n_tokens < n:
                for token_idx in range(n - n_tokens):
                    self.assertAlmostEqual(0.0, np.linalg.norm(X[text_idx][token_idx + n_tokens]))

    def test_texts_to_X_2(self):
        source_texts = (
            'Мама мыла раму.',
            'Папа мыл синхрофазотрон!',
            'Скажи-ка, дядя, ведь недаром Москва?',
            'инопланетяне скоро захватят нашу планету!!!'
        )
        token_bounds = [
            ((0, 4), (5, 4), (10, 4)),
            ((0, 4), (5, 3), (9, 14)),
            ((0, 8), (10, 4), (16, 4), (21, 7), (29, 6)),
            ((0, 12), (13, 5), (19, 8), (28, 4), (33, 7))
        ]
        n = 4
        self.tagger.update_elmo()
        X = self.tagger.texts_to_X(source_texts, token_bounds, n)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.ndim, 3)
        self.assertEqual(X.shape[0], 4)
        self.assertEqual(X.shape[1], n)
        self.assertEqual(X.shape[2], 1024)
        for text_idx in range(len(source_texts)):
            n_tokens = len(token_bounds[text_idx])
            for token_idx in range(min(n_tokens, n)):
                self.assertGreater(np.linalg.norm(X[text_idx][token_idx]), 0.0)
            if n_tokens < n:
                for token_idx in range(n - n_tokens):
                    self.assertAlmostEqual(0.0, np.linalg.norm(X[text_idx][token_idx + n_tokens]))

    def test_labels_to_y(self):
        source_texts = (
            'Мама мыла раму.',
            'Папа мыл синхрофазотрон!',
            'Скажи-ка, дядя, ведь недаром Москва?',
            'инопланетяне скоро захватят нашу планету!!!'
        )
        source_labels = [
            (('PER', 0, 4),),
            (('ORG', 8, 15),),
            (('LOC', 29, 7),),
            (('PER', 0, 12), ('LOC', 28, 12))
        ]
        token_bounds = [
            ((0, 4), (5, 4), (10, 4)),
            ((0, 4), (5, 3), (9, 14)),
            ((0, 8), (10, 4), (16, 4), (21, 7), (29, 6)),
            ((0, 12), (13, 5), (19, 8), (28, 4), (33, 7))
        ]
        max_text_len = 5
        self.tagger.named_entities_ = ('LOC', 'ORG', 'PER')
        true_labels = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0]
                ],
                [
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0]
                ],
                [
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0]
                ],
                [
                    [0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0]
                ]
            ],
            dtype=np.float32
        )
        predicted_labels = self.tagger.labels_to_y(source_texts, source_labels, token_bounds, max_text_len)
        self.assertIsInstance(predicted_labels, np.ndarray)
        self.assertEqual(true_labels.shape, predicted_labels.shape)
        for text_idx in range(true_labels.shape[0]):
            for token_idx in range(true_labels.shape[1]):
                for class_idx in range(true_labels.shape[2]):
                    self.assertAlmostEqual(true_labels[text_idx][token_idx][class_idx],
                                           predicted_labels[text_idx][token_idx][class_idx])

    def test_strip_token_bounds_1(self):
        src_text = 'Здравствуй, мир! И тебе привет, пришелец!'
        src_bounds = (18, 6)
        true_bounds = (19, 4)
        self.assertEqual(true_bounds, NeuroTagger.strip_token_bounds(src_text, src_bounds[0], src_bounds[1]))

    def test_strip_token_bounds_2(self):
        src_text = 'Здравствуй, мир! И тебе привет, пришелец!'
        src_bounds = (19, 4)
        true_bounds = (19, 4)
        self.assertEqual(true_bounds, NeuroTagger.strip_token_bounds(src_text, src_bounds[0], src_bounds[1]))

    def test_prepare_labels(self):
        source_labels = (('PER', 0, 12), ('LOC', 28, 12))
        token_bounds = ((0, 12), (13, 5), (19, 8), (28, 4), (33, 7), (40, 1), (41, 1), (42, 1))
        true_labels = [('PER', 0, 1), ('LOC', 3, 5)]
        self.assertEqual(true_labels, NeuroTagger.prepare_labels(43, token_bounds, source_labels))

    def test_fit_predict_score(self):
        res = self.tagger.fit(self.texts, self.labels)
        self.assertIsInstance(res, NeuroTagger)
        self.assertTrue(hasattr(res, 'tokenizer_'))
        self.assertTrue(hasattr(res, 'elmo_'))
        self.assertTrue(hasattr(res, 'classifier_'))
        self.assertTrue(hasattr(res, 'named_entities_'))
        self.assertTrue(hasattr(res, 'max_text_len_'))
        self.assertTrue(hasattr(res, 'embedding_size_'))
        self.assertEqual(self.tagger.named_entities_, ('LOC', 'ORG', 'PER'))
        predicted = res.predict(self.texts)
        self.assertIsInstance(predicted, list)
        self.assertEqual(len(self.labels), len(predicted))
        for text_idx in range(len(predicted)):
            self.assertIsInstance(predicted[text_idx], tuple)
            self.assertGreater(len(predicted[text_idx]), 0)
        f1 = self.tagger.score(self.texts, self.labels)
        self.assertGreater(f1, 0.0)
        self.assertLessEqual(f1, 1.0)

    def test_serialize_fitted(self):
        self.tagger.fit(self.texts, self.labels)
        predicted_1 = tuple(self.tagger.predict(self.texts))
        with open(self.model_name, 'wb') as fp:
            pickle.dump(self.tagger, fp)
        true_named_entities = self.tagger.named_entities_
        true_max_text_len = self.tagger.max_text_len_
        true_embedding_size = self.tagger.embedding_size_
        del self.tagger
        with open(self.model_name, 'rb') as fp:
            other_tagger = pickle.load(fp)
        self.assertIsInstance(other_tagger, NeuroTagger)
        self.assertTrue(hasattr(other_tagger, 'elmo_name'))
        self.assertTrue(hasattr(other_tagger, 'n_units'))
        self.assertTrue(hasattr(other_tagger, 'dropout'))
        self.assertTrue(hasattr(other_tagger, 'recurrent_dropout'))
        self.assertTrue(hasattr(other_tagger, 'l2_kernel'))
        self.assertTrue(hasattr(other_tagger, 'l2_chain'))
        self.assertTrue(hasattr(other_tagger, 'n_epochs'))
        self.assertTrue(hasattr(other_tagger, 'validation_part'))
        self.assertTrue(hasattr(other_tagger, 'batch_size'))
        self.assertTrue(hasattr(other_tagger, 'verbose'))
        self.assertTrue(hasattr(other_tagger, 'use_crf'))
        self.assertTrue(hasattr(other_tagger, 'use_lstm'))
        self.assertEqual(other_tagger.elmo_name, self.elmo_model_name)
        self.assertEqual(other_tagger.n_units, 16)
        self.assertAlmostEqual(other_tagger.dropout, 0.5)
        self.assertAlmostEqual(other_tagger.recurrent_dropout, 0.0)
        self.assertAlmostEqual(other_tagger.l2_kernel, 1e-3)
        self.assertAlmostEqual(other_tagger.l2_chain, 0.0)
        self.assertEqual(other_tagger.n_epochs, 10)
        self.assertAlmostEqual(other_tagger.validation_part, 1.0 / 6.0)
        self.assertEqual(other_tagger.batch_size, 2)
        self.assertTrue(other_tagger.verbose)
        self.assertTrue(other_tagger.use_crf)
        self.assertTrue(other_tagger.use_lstm)
        self.assertFalse(hasattr(other_tagger, 'tokenizer_'))
        self.assertFalse(hasattr(other_tagger, 'elmo_'))
        self.assertTrue(hasattr(other_tagger, 'classifier_'))
        self.assertTrue(hasattr(other_tagger, 'named_entities_'))
        self.assertTrue(hasattr(other_tagger, 'max_text_len_'))
        self.assertTrue(hasattr(other_tagger, 'embedding_size_'))
        self.assertEqual(other_tagger.named_entities_, true_named_entities)
        self.assertEqual(other_tagger.max_text_len_, true_max_text_len)
        self.assertEqual(other_tagger.embedding_size_, true_embedding_size)
        predicted_2 = tuple(other_tagger.predict(self.texts))
        self.assertEqual(predicted_1, predicted_2)

    def test_check_X_1(self):
        X = {1, 2, 3}
        true_err_msg = re.escape('`X_train` is wrong! Expected `{0}`, `{1}` or `{2}`, but got `{3}`.'.format(
            type([1, 2]), type((1, 2)), type(np.array([1, 2])), type(X)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.tagger.check_X(X, 'X_train')

    def test_check_X_2(self):
        X = np.random.uniform(0.0, 1.0, size=(10, 3))
        true_err_msg = re.escape('`X_test` is wrong! Expected 1-D array, but got {0}-D one.'.format(X.ndim))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.tagger.check_X(X, 'X_test')

    def test_check_y_1(self):
        source_labels = [
            (('ORG', 0, 4), ('PER', 10, 3)),
            (('LOC', 6, 10),),
            (('PER', 0, 9),)
        ]
        lengths_of_texts = [45, 78, 17]
        true_named_entities = ('LOC', 'ORG', 'PER')
        self.assertEqual(true_named_entities, self.tagger.check_y(source_labels, lengths_of_texts))

    def test_check_y_2(self):
        source_labels = {
            (('ORG', 0, 4), ('PER', 10, 3)),
            (('LOC', 6, 10),),
            (('PER', 0, 9),)
        }
        lengths_of_texts = [45, 78, 17]
        true_err_msg = re.escape('`yyy` is wrong! Expected `{0}`, `{1}` or `{2}`, but got `{3}`.'.format(
            type([1, 2]), type((1, 2)), type(np.array([1, 2])), type(source_labels)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = self.tagger.check_y(source_labels, lengths_of_texts, 'yyy')

    def test_check_y_3(self):
        source_labels = [
            (('ORG', 0, 4), ('PER', 10, 3)),
            (('LOC', 6, 10),),
            (('PER', 0, 9),)
        ]
        lengths_of_texts = [45, 78, 17, 2]
        true_err_msg = re.escape('`yyy` does not correspond to sequence of input texts. Number of input texts is 4, '
                                 'but length of `yyy` is 3.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = self.tagger.check_y(source_labels, lengths_of_texts, 'yyy')

    def test_check_y_4(self):
        source_labels = [
            {('ORG', 0, 4), ('PER', 10, 3)},
            (('LOC', 6, 10),),
            (('PER', 0, 9),)
        ]
        lengths_of_texts = [45, 78, 17]
        true_err_msg = re.escape('Labels for sample {0} are wrong! Expected `{1}` or `{2}`, but got `{3}`.'.format(
            0, type([1, 2]), type((1, 2)), type({1, 2})))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = self.tagger.check_y(source_labels, lengths_of_texts, 'yyy')

    def test_check_y_5(self):
        source_labels = [
            (('ORG', 0, 4), ('PER', 10, 3)),
            ({'LOC', 6, 10, 1},),
            (('PER', 0, 9),)
        ]
        lengths_of_texts = [45, 78, 17]
        true_err_msg = re.escape('Description of tag {0} for sample {1} is wrong! Expected `{2}` or `{3}`, but '
                                 'got `{4}`.'.format(0, 1, type([1, 2]), type((1, 2)), type({1, 2})))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = self.tagger.check_y(source_labels, lengths_of_texts, 'yyy')

    def test_check_y_6(self):
        source_labels = [
            (('ORG', 0, 4), ('PER', 10, 3)),
            (['LOC', 6, 10, 1],),
            (('PER', 0, 9),)
        ]
        lengths_of_texts = [45, 78, 17]
        true_err_msg = re.escape('Description of tag {0} for sample {1} is wrong! Expected a 3-element sequence, but '
                                 'got a {2}-element one.'.format(0, 1, 4))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = self.tagger.check_y(source_labels, lengths_of_texts, 'yyy')

    def test_check_y_7(self):
        source_labels = [
            (('ORG', 0, 4), (-2, 10, 3)),
            (('LOC', 6, 10),),
            (('PER', 0, 9),)
        ]
        lengths_of_texts = [45, 78, 17]
        true_err_msg = re.escape('Description of tag {0} for sample {1} is wrong! First element (entity type) must be '
                                 'a string object, but it is `{2}`.'.format(1, 0, type(-2)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = self.tagger.check_y(source_labels, lengths_of_texts, 'yyy')

    def test_check_y_8(self):
        source_labels = [
            (('ORG', 0, 4), ('O', 10, 3)),
            (('LOC', 6, 10),),
            (('PER', 0, 9),)
        ]
        lengths_of_texts = [45, 78, 17]
        true_err_msg = re.escape('Description of tag 1 for sample 0 is wrong! First element (entity type) is '
                                 'incorrect.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = self.tagger.check_y(source_labels, lengths_of_texts, 'yyy')

    def test_check_y_9(self):
        source_labels = [
            (('ORG', 0, 4), ('PER', '10', 3)),
            (('LOC', 6, 10),),
            (('PER', 0, 9),)
        ]
        lengths_of_texts = [45, 78, 17]
        true_err_msg = re.escape('Description of tag {0} for sample {1} is wrong! Second element (entity start) must '
                                 'be `{2}`, but it is `{3}`.'.format(1, 0, type(3), type('10')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = self.tagger.check_y(source_labels, lengths_of_texts, 'yyy')

    def test_check_y_10(self):
        source_labels = [
            (('ORG', 0, 4), ('PER', 10, '3')),
            (('LOC', 6, 10),),
            (('PER', 0, 9),)
        ]
        lengths_of_texts = [45, 78, 17]
        true_err_msg = re.escape('Description of tag {0} for sample {1} is wrong! Third element (entity length) must '
                                 'be `{2}`, but it is `{3}`.'.format(1, 0, type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = self.tagger.check_y(source_labels, lengths_of_texts, 'yyy')

    def test_check_y_11(self):
        source_labels = [
            (('ORG', 0, 4), ('PER', -1, 3)),
            (('LOC', 6, 10),),
            (('PER', 0, 9),)
        ]
        lengths_of_texts = [45, 78, 17]
        true_err_msg = re.escape('Description of tag {0} for sample {1} is wrong! ({2}, {3}) is inadmissible value of '
                                 'entity bounds.'.format(1, 0, -1, 3))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = self.tagger.check_y(source_labels, lengths_of_texts, 'yyy')

    def test_check_y_12(self):
        source_labels = [
            (('ORG', 0, 4), ('PER', 10, 3)),
            (('LOC', 6, 10),),
            (('PER', 0, 9),)
        ]
        lengths_of_texts = [45, 78, 5]
        true_err_msg = re.escape('Description of tag {0} for sample {1} is wrong! ({2}, {3}) is inadmissible value of '
                                 'entity bounds.'.format(0, 2, 0, 9))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = self.tagger.check_y(source_labels, lengths_of_texts, 'yyy')

    def test_check_params(self):
        params = self.tagger.get_params(deep=True)
        params['use_crf'] = False
        params['use_lstm'] = False
        true_err_msg = re.escape('`use_lstm` or `use_crf` must be True.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.tagger.check_params(**params)

    def load_dataset(self, file_name: str) -> Tuple[List[str], List[tuple]]:
        texts = []
        labels = []
        new_text = ''
        labels_of_new_text = []
        ne_start_idx = -1
        ne_type = ''
        token_idx = 0
        line_idx = 0
        with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
            cur_line = fp.readline()
            while len(cur_line) > 0:
                err_msg = 'File `{0}`: line {1} is wrong!'.format(file_name, line_idx)
                prep_line = cur_line.strip()
                if len(prep_line) > 0:
                    parts_of_line = prep_line.split()
                    if len(parts_of_line) != 2:
                        raise ValueError(err_msg)
                    new_token = parts_of_line[0].strip()
                    if len(new_token) == 0:
                        raise ValueError(err_msg)
                    if (parts_of_line[1] != 'O') and (not parts_of_line[1].startswith('B-'))  and \
                            (not parts_of_line[1].startswith('I-')):
                        raise ValueError(err_msg)
                    if parts_of_line[1] == 'O':
                        if ne_start_idx >= 0:
                            labels_of_new_text.append((ne_type, ne_start_idx, len(new_text.strip()) - ne_start_idx))
                            ne_start_idx = -1
                            ne_type = ''
                    elif parts_of_line[1].startswith('B-'):
                        if ne_start_idx >= 0:
                            labels_of_new_text.append((ne_type, ne_start_idx, len(new_text.strip()) - ne_start_idx))
                        ne_type = parts_of_line[1][2:].strip()
                        if (not ne_type.isupper()) or (len(ne_type) == 0):
                            raise ValueError(err_msg)
                        ne_start_idx = token_idx
                    else:
                        if ne_start_idx < 0:
                            raise ValueError(err_msg)
                        if parts_of_line[1][2:].strip() != ne_type:
                            raise ValueError(err_msg)
                    token_idx += 1
                    if new_token in {'.', ','}:
                        new_text = new_text.strip()
                    new_text += (new_token + ' ')
                else:
                    new_text = new_text.strip()
                    if len(new_text) == 0:
                        raise ValueError(err_msg)
                    if ne_start_idx >= 0:
                        labels_of_new_text.append((ne_type, ne_start_idx, len(new_text) - ne_start_idx))
                        ne_start_idx = -1
                        ne_type = ''
                    texts.append(new_text)
                    labels.append(tuple(labels_of_new_text))
                    token_idx = 0
                cur_line = fp.readline()
                line_idx += 1
        new_text = new_text.strip()
        if len(new_text) > 0:
            if ne_start_idx >= 0:
                labels_of_new_text.append((ne_type, ne_start_idx, len(new_text) - ne_start_idx))
            texts.append(new_text)
            labels.append(tuple(labels_of_new_text))
        return texts, labels

    @staticmethod
    def load_russian_elmo(url: str, target_dir: str) -> str:
        if not os.path.isdir(os.path.dirname(target_dir)):
            raise ValueError('The directory `{0}` does not exist!'.format(os.path.dirname(target_dir)))
        slash_pos = url.rfind('/')
        if slash_pos < 0:
            raise ValueError('{0} is wrong URL for the Russian ELMo!'.format(url))
        base_name = url[(slash_pos + 1):].strip()
        if len(base_name) == 0:
            raise ValueError('{0} is wrong URL for the Russian ELMo!'.format(url))
        if not base_name.lower().endswith('.tar.gz'):
            raise ValueError('{0} is wrong URL for the Russian ELMo!'.format(url))
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
            elmo_gzip_archive = os.path.join(os.path.dirname(target_dir), base_name)
            try:
                with open(elmo_gzip_archive, 'wb') as fp:
                    response = get(url)
                    fp.write(response.content)
                with tarfile.open(elmo_gzip_archive) as fp:
                    fp.extractall(path=target_dir)
            finally:
                os.remove(elmo_gzip_archive)
        return target_dir


if __name__ == '__main__':
    unittest.main(verbosity=2)
