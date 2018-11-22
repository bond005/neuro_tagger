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
    from neuro_tagger.tokenizer import DefaultTokenizer, BaseTokenizer
    from neuro_tagger.dataset_loading import load_dataset_from_brat, load_dataset_from_factrueval2016
    from neuro_tagger.dataset_loading import tokenize_all_by_sentences
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from neuro_tagger.neuro_tagger import NeuroTagger
    from neuro_tagger.tokenizer import DefaultTokenizer, BaseTokenizer
    from neuro_tagger.dataset_loading import load_dataset_from_brat, load_dataset_from_factrueval2016
    from neuro_tagger.dataset_loading import tokenize_all_by_sentences


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
            validation_part=(1.0 / 6.0), batch_size=2, verbose=True, use_crf=True, use_lstm=True,
            tokenizer=DefaultTokenizer()
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
        self.assertTrue(hasattr(self.tagger, 'tokenizer'))
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
        self.assertIsInstance(self.tagger.tokenizer, BaseTokenizer)
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
        self.assertTrue(hasattr(other_tagger, 'tokenizer'))
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
        self.assertIsInstance(other_tagger.tokenizer, BaseTokenizer)
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
        self.assertFalse(hasattr(other_tagger, 'elmo_'))
        self.assertFalse(hasattr(other_tagger, 'classifier_'))
        self.assertFalse(hasattr(other_tagger, 'named_entities_'))
        self.assertFalse(hasattr(other_tagger, 'max_text_len_'))
        self.assertFalse(hasattr(other_tagger, 'embedding_size_'))

    def test_tokenize_1(self):
        s = ['Мама мыла раму.', 'Папа мыл синхрофазотрон!!!']
        true_tokens = [
            ((0, 4), (5, 4), (10, 4), (14, 1)),
            ((0, 4), (5, 3), (9, 14), (23, 1), (24, 1), (25, 1))
        ]
        predicted_tokens = self.tagger.tokenize(s)
        self.assertIsInstance(predicted_tokens, list)
        self.assertEqual(true_tokens, predicted_tokens)

    def test_tokenize_2(self):
        s = ('Мама мыла раму.', 'Папа -- синхрофазотрон!')
        true_tokens = [
            ((0, 4), (5, 4), (10, 4), (14, 1)),
            ((0, 4), (5, 1), (6, 1), (8, 14), (22, 1))
        ]
        predicted_tokens = self.tagger.tokenize(s)
        self.assertIsInstance(predicted_tokens, list)
        self.assertEqual(true_tokens, predicted_tokens)

    def test_tokenize_3(self):
        s = np.array(['Мама мыла раму.', 'Папа мыл синхрофазотрон!'], dtype=object)
        true_tokens = [
            ((0, 4), (5, 4), (10, 4), (14, 1)),
            ((0, 4), (5, 3), (9, 14), (23, 1))
        ]
        predicted_tokens = self.tagger.tokenize(s)
        self.assertIsInstance(predicted_tokens, list)
        self.assertEqual(true_tokens, predicted_tokens)

    def test_tokenize_4(self):
        s = np.array(['Мама мыла раму _.', 'Папа мыл синхрофазотрон!'], dtype=object)
        true_tokens = [
            ((0, 4), (5, 4), (10, 4), (15, 1), (16, 1)),
            ((0, 4), (5, 3), (9, 14), (23, 1))
        ]
        predicted_tokens = self.tagger.tokenize(s)
        self.assertIsInstance(predicted_tokens, list)
        self.assertEqual(true_tokens, predicted_tokens)

    def test_tokenize_5(self):
        s = []
        true_tokens = []
        predicted_tokens = self.tagger.tokenize(s)
        self.assertIsInstance(predicted_tokens, list)
        self.assertEqual(true_tokens, predicted_tokens)

    def test_tokenize_6(self):
        s = [
            'Paper Title: ____________________________________________________________________________________________'
            '________________________ _______________________________________________________________________________'
            '______________________________________________ Author(s): _______________________________________________'
            '______________________________________________________________________ _________________________________'
            '____________________________________________________________________________________________ Speaker: '
        ]
        true_tokens = [
            ((0, 5), (6, 5), (11, 1), (13, 1), (14, 1), (15, 1), (256, 6), (262, 1), (263, 1), (264, 1), (265, 1),
             (267, 1), (268, 1), (269, 1), (511, 7), (518, 1))
        ]
        predicted_tokens = self.tagger.tokenize(s)
        self.assertIsInstance(predicted_tokens, list)
        self.assertEqual(true_tokens, predicted_tokens)

    def test_tokenize_7(self):
        s = ['мама мыла . . . . . . . . . . . . . раму']
        true_tokens = [
            ((0, 4), (5, 4), (10, 1), (12, 1), (14, 1), (36, 4))
        ]
        predicted_tokens = self.tagger.tokenize(s)
        self.assertIsInstance(predicted_tokens, list)
        self.assertEqual(true_tokens, predicted_tokens)

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
            (('org', 8, 15),),
            (('LOC', 29, 7),),
            (('PER', 0, 12), ('loc', 28, 12))
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

    def test_f1_macro_1(self):
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
        pred_labels = np.array(
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
        lengths_of_texts = [3, 3, 5, 5]
        self.assertAlmostEqual(1.0, NeuroTagger.f1_macro(true_labels, pred_labels, lengths_of_texts))

    def test_f1_macro_2(self):
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
        pred_labels = np.array(
            [
                [
                    [1, 0, 0, 0, 0, 0, 0],
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
        lengths_of_texts = [3, 3, 5, 5]
        true_f1 = (1.0 + 1.0 + 1.0 + 2.0 * 0.5 * 1.0 / (0.5 + 1.0)) / 4.0
        self.assertAlmostEqual(true_f1, NeuroTagger.f1_macro(true_labels, pred_labels, lengths_of_texts))

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

    def test_score(self):
        self.tagger.fit(self.texts, self.labels)
        new_texts = [
            'Мама мыла раму',
            'All bugs must be fixed!'
        ]
        new_labels = [
            (('OPERATION', 5, 4), ('EQUIPMENT', 10, 4)),
            (('OPERATION', 9, 13),)
        ]
        true_err_msg = re.escape('`y` is wrong! These entities are unknown: [{0}].'.format(
            ', '.join(['EQUIPMENT', 'OPERATION'])))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = self.tagger.score(new_texts, new_labels)

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
        self.assertTrue(hasattr(other_tagger, 'tokenizer'))
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
        self.assertIsInstance(other_tagger.tokenizer, BaseTokenizer)
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

    def test_load_dataset_from_brat(self):
        brat_data_path = os.path.join(os.path.dirname(__file__), 'testdata', 'brat')
        true_texts = [
            'FIG. 14-128 Superheated high-pressure hot-water requirements for 99 per-cent collection as a function of '
            'particle size in a two-phase eductor jet scrubber. To convert gallons per 1000 cubic feet to cubic meters '
            'per 1000 cubic meters, multiply by 0.134. [Gardenier, J. Air Pollut. Control Assoc., 24, 954 (1974).]',
            'To  a  good  approximation,  P / T  conditions  for  LW-H-V  of  the  pure  components  in  Table 11.7  '
            'lie  on  a  straight  line  between  Q1  and  Q2,  on  a  semilogarithmic  plot  (ln  p  vs.  1 / Ta).  '
            'As discussed  below  in  the  Hand  Calculations  of  Hydrate  Formation  Conditions  section,  there  '
            'is no simple way to expand the above pure lines into that for a mixture, though there are several ways '
            'to hand-calculate LW-H-V conditions (P / T) for mixed hydrocarbon hydrate formers.',
            '15.1.6 Industry  Standards.    As  mentioned,  despite  the  large  number  of  installations  '
            'world-wide,  PC  pumps  and  drive  systems  do  not,  in  general,  conform  to  any  industry  '
            'standards  or common specifications. As a result, there is significant variation in the products '
            'available from different vendors, which generally precludes interchangeability of equipment components. '
            'The nomenclature  (e.g.,  naming  conventions,  ratings)  used  in  conjunction  with  both  pumps  and '
            'drive units also varies considerably, which can make it difficult for users to easily compare and '
            'select  products  from  different  suppliers.  Nevertheless,  there  have  been  some  recent  efforts  '
            'to develop industry standards for PCP systems.',
            'Drums, hand-trucks, pallets, forklifts Transfer piping, hoses, pumps Transfer piping, hoses, pumps Heat '
            'sterilization prior to bagging; special heavy-duty bags with hazard warning printed on sides Fume '
            'ventilation, temperature control',
            'The full text of Oil & Gas Journal is available through OGJ Online, Oil & Gas Journal’s internet-based '
            'energy information service, at http://www.ogjonline.com. For information, send an e-mail message to '
            'webmaster@ogjonline.com.'
        ]
        true_labels = [
            (
                ('equipment', 12, 47), ('property_values', 65, 11), ('equipment', 77, 10), ('properties', 93, 8),
                ('equipment', 124, 30), ('property_values', 167, 65)
            ),
            (
                ('properties', 29, 17), ('properties', 53, 6), ('equipment', 76, 10), ('properties', 141, 2),
                ('properties', 150, 2), ('equipment', 162, 21), ('equipment', 233, 23), ('properties', 262, 30),
                ('equipment', 294, 7), ('equipment', 353, 5), ('equipment', 375, 7), ('properties', 432, 6),
                ('equipment', 439, 10), ('properties', 450, 45)
            ),
            (
                ('equipment', 80, 13), ('equipment', 108, 9), ('equipment', 124, 14), ('equipment', 182, 8),
                ('equipment', 314, 7), ('equipment', 371, 20), ('equipment', 427, 11), ('equipment', 630, 9),
                ('properties', 710, 8), ('equipment', 733, 11)
            ),
            (
                ('equipment', 0, 5), ('properties', 7, 11), ('equipment', 20, 7), ('equipment', 29, 25),
                ('equipment', 56, 5), ('equipment', 63, 21), ('equipment', 86, 5), ('equipment', 93, 5),
                ('operations', 99, 18), ('equipment', 144, 15), ('equipment', 215, 19)
            ),
            (
                ('equipment', 17, 9), ('equipment', 68, 9), ('equipment', 103, 6)
            )
        ]
        loaded_texts, loaded_labels = load_dataset_from_brat(brat_data_path)
        self.assertEqual(true_texts, loaded_texts)
        self.assertEqual(true_labels, loaded_labels)

    def test_load_dataset_from_factrueval2016(self):
        factrueval_data_path = os.path.join(os.path.dirname(__file__), 'testdata', 'factrueval')
        true_texts = [
            'Встреча с послом Италии в миде Грузии',
            '  По инициативе итальянской стороны чрезвычайный и полномочный посол Италии в Грузии Виторио Сандали '
            'встретился с заместителем министра иностранных дел Грузии Александром Налбандовым.',
            ' Предметом обсуждения стали вопросы сотрудничества в международных организациях.',
            'Совершенно новую технологию перекачки российской водки за рубеж начали использовать контрабандисты.',
            '  Произошло это осенью 2004 года, но только сейчас эстонские власти предъявили обвинения своим и '
            'российским гражданам, участвовавшим в этом процессе.',
            ' Таковых оказалось одиннадцать.',
            '  Пока эстонская прокуратура будет разбираться во всех тонкостях этого редкого "бизнеса", можно вспомнить '
            'кое-что из его небольшой истории.',
            ' В августе 2004 года "продуктопровод", проложенный нелегально через пограничную реку Нарву и имеющий '
            'общую протяжённость около двух километров, был испытан и начал функционировать.',
            ' Преступники пользовались им до ноября того же года и успели перекачать более шести тонн водки.',
            ' Но вот с её реализацией дело застопорилось, по крайней мере в Таллине, хотя в Тарту дело пошло.',
            ' В конце концов эстонская полиция наткнулась на грузовик с водкой и в ходе разбирательств вышла на '
            '"водкопровод".'
        ]
        true_bounds_of_tokens = [
            ((0, 7), (8, 9), (10, 16), (17, 23), (24, 25), (26, 30), (31, 37)),
            ((2, 4), (5, 15), (16, 27), (28, 35), (36, 48), (49, 50), (51, 62), (63, 68), (69, 75), (76, 77), (78, 84),
             (85, 92), (93, 100), (101, 111), (112, 113), (114, 126), (127, 135), (136, 147), (148, 151), (152, 158),
             (159, 170), (171, 182), (182, 183)),
            ((1, 10), (11, 21), (22, 27), (28, 35), (36, 50), (51, 52), (53, 66), (67, 79), (79, 80)),
            ((0, 10), (11, 16), (17, 27), (28, 37), (38, 48), (49, 54), (55, 57), (58, 63), (64, 70), (71, 83),
             (84, 98), (98, 99)),
            ((2, 11), (12, 15), (16, 22), (23, 27), (28, 32), (32, 33), (34, 36), (37, 43), (44, 50), (51, 60),
             (61, 67), (68, 78), (79, 88), (89, 94), (95, 96), (97, 107), (108, 117), (117, 118), (119, 132),
             (133, 134), (135, 139), (140, 148), (148, 149)),
            ((1, 8), (9, 18), (19, 30), (30, 31)),
            ((2, 6), (7, 16), (17, 28), (29, 34), (35, 46), (47, 49), (50, 54), (55, 64), (65, 70), (71, 78), (79, 80),
             (80, 87), (87, 88), (88, 89), (90, 95), (96, 105), (106, 113), (114, 116), (117, 120), (121, 130),
             (131, 138), (138, 139)),
            ((1, 2), (3, 10), (11, 15), (16, 20), (21, 22), (22, 36), (36, 37), (37, 38), (39, 50), (51, 61), (62, 67),
             (68, 79), (80, 84), (85, 90), (91, 92), (93, 100), (101, 106), (107, 120), (121, 126), (127, 131),
             (132, 142), (142, 143), (144, 147), (148, 155), (156, 157), (158, 163), (164, 179), (179, 180)),
            ((1, 12), (13, 25), (26, 28), (29, 31), (32, 38), (39, 43), (44, 46), (47, 51), (52, 53), (54, 60),
             (61, 71), (72, 77), (78, 83), (84, 88), (89, 94), (94, 95)),
            ((1, 3), (4, 7), (8, 9), (10, 12), (13, 24), (25, 29), (30, 43), (43, 44), (45, 47), (48, 55), (56, 60),
             (61, 62), (63, 70), (70, 71), (72, 76), (77, 78), (79, 84), (85, 89), (90, 95), (95, 96)),
            ((1, 2), (3, 8), (9, 15), (16, 25), (26, 33), (34, 44), (45, 47), (48, 56), (57, 58), (59, 65), (66, 67),
             (68, 69), (70, 74), (75, 89), (90, 95), (96, 98), (99, 100), (100, 111), (111, 112), (112, 113))
        ]
        true_labels = [
            (('LOC', 17, 6), ('ORG', 26, 11)),
            (('LOC', 69, 6), ('LOC', 78, 6), ('PER', 85, 15), ('LOC', 152, 6), ('PER', 159, 23)),
            tuple(),
            tuple(),
            tuple(),
            tuple(),
            tuple(),
            (('LOC', 80, 10),),
            tuple(),
            (('LOC', 63, 7), ('LOC', 79, 5)),
            tuple()
        ]
        true_names = ['book_58', 'book_58', 'book_58', 'book_146', 'book_146', 'book_146', 'book_146', 'book_146',
                      'book_146', 'book_146', 'book_146']
        loaded_texts, loaded_bounds_of_tokens, loaded_labels, loaded_names = load_dataset_from_factrueval2016(
            factrueval_data_path)
        self.assertEqual(true_texts, loaded_texts)
        self.assertEqual(true_bounds_of_tokens, loaded_bounds_of_tokens)
        self.assertEqual(true_labels, loaded_labels)
        self.assertEqual(true_names, loaded_names)

    def test_tokenize_all_by_sentences_1(self):
        source_texts = [
            'As  mentioned,  despite  the  large  number  of  installations  world-wide,  PC  pumps  and  drive  '
            'systems  do  not,  in  general,  conform  to  any  industry  standards  or common specifications. '
            'As a result, there is significant variation in the products available from different vendors, which '
            'generally precludes interchangeability of equipment components.',
            'The nomenclature  (e.g.,  naming  conventions,  ratings)  used  in  conjunction  with  both  pumps  and '
            'drive units also varies considerably, which can make it difficult for users to easily compare and select  '
            'products  from  different  suppliers.  Nevertheless,  there  have  been  some  recent  efforts  to '
            'develop industry standards for PCP systems.'
        ]
        source_labels = [
            (
                ('equipment', 49, 13), ('equipment', 77, 9), ('equipment', 93, 14), ('equipment', 151, 8),
                ('equipment', 283, 7), ('equipment', 340, 20)
            ),
            (
                ('equipment', 34, 11), ('equipment', 237, 9), ('properties', 317, 8), ('equipment', 340, 11)
            )
        ]
        true_texts = [
            'As  mentioned,  despite  the  large  number  of  installations  world-wide,  PC  pumps  and  drive  '
            'systems  do  not,  in  general,  conform  to  any  industry  standards  or common specifications.',
            'As a result, there is significant variation in the products available from different vendors, which '
            'generally precludes interchangeability of equipment components.',
            'The nomenclature  (e.g.,  naming  conventions,  ratings)  used  in  conjunction  with  both  pumps  and '
            'drive units also varies considerably, which can make it difficult for users to easily compare and select  '
            'products  from  different  suppliers.',
            'Nevertheless,  there  have  been  some  recent  efforts  to develop industry standards for PCP systems.'
        ]
        true_labels = [
            (
                ('equipment', 49, 13), ('equipment', 77, 9), ('equipment', 93, 14), ('equipment', 151, 8)
            ),
            (
                ('equipment', 85, 7), ('equipment', 142, 20)
            ),
            (
                ('equipment', 34, 11), ('equipment', 237, 9)
            ),
            (
                ('properties', 68, 8), ('equipment', 91, 11)
            )
        ]
        pred_texts, pred_labels = tokenize_all_by_sentences(source_texts, source_labels)
        self.assertEqual(true_texts, pred_texts)
        self.assertEqual(true_labels, pred_labels)

    def test_tokenize_all_by_sentences_2(self):
        source_texts = [
            'As  mentioned,  despite  the  large  number  of  installations  world-wide,  PC  pumps  and  drive  '
            'systems  do  not,  in  general,  conform  to  any  industry  standards  or common specifications. '
            'As a result, there is significant variation in the products available from different vendors, which '
            'generally precludes interchangeability of equipment components.',
            'The nomenclature  (e.g.,  naming  conventions,  ratings)  used  in  conjunction  with  both  pumps  and '
            'drive units also varies considerably, which can make it difficult for users to easily compare and select  '
            'products  from  different  suppliers.  Nevertheless,  there  have  been  some  recent  efforts  to '
            'develop industry standards for PCP systems.'
        ]
        source_labels = [
            (
                ('equipment', 49, 13), ('equipment', 77, 9), ('equipment', 93, 14), ('equipment', 151, 8),
                ('equipment', 283, 7), ('equipment', 340, 20)
            ),
            (
                ('equipment', 34, 11), ('equipment', 237, 24), ('properties', 317, 8), ('equipment', 340, 11)
            )
        ]
        true_texts = [
            'As  mentioned,  despite  the  large  number  of  installations  world-wide,  PC  pumps  and  drive  '
            'systems  do  not,  in  general,  conform  to  any  industry  standards  or common specifications.',
            'As a result, there is significant variation in the products available from different vendors, which '
            'generally precludes interchangeability of equipment components.',
            'The nomenclature  (e.g.,  naming  conventions,  ratings)  used  in  conjunction  with  both  pumps  and '
            'drive units also varies considerably, which can make it difficult for users to easily compare and select  '
            'products  from  different  suppliers.  Nevertheless,  there  have  been  some  recent  efforts  to '
            'develop industry standards for PCP systems.'
        ]
        true_labels = [
            (
                ('equipment', 49, 13), ('equipment', 77, 9), ('equipment', 93, 14), ('equipment', 151, 8)
            ),
            (
                ('equipment', 85, 7), ('equipment', 142, 20)
            ),
            (
                ('equipment', 34, 11), ('equipment', 237, 24), ('properties', 317, 8), ('equipment', 340, 11)
            )
        ]
        pred_texts, pred_labels = tokenize_all_by_sentences(source_texts, source_labels)
        self.assertEqual(true_texts, pred_texts)
        self.assertEqual(true_labels, pred_labels)

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
