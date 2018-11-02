from setuptools import setup, find_packages

import neuro_tagger

long_description = '''
neuro_tagger
============

Text tagger based on recurrent neural network. It can be used as NER,
dependency parser, morphoanalyzer etc.

The goal of this project is creation of a simple Python package with
the sklearn-like interface for solution of different tasks of text
tagging (named entity recognition, dependency parsing, etc) in case
number of labeled texts is very small (not greater than several
thousands). Special word embeddings named as `ELMo
<https://arxiv.org/pdf/1802.05365.pdf>` (**E**mbeddings from **L**anguage
**Mo**dels) ensure this possibility, because these embeddings are
contextual and they allow to design more simple and separable feature
space for words in texts.

ELMo embeddings are used as features of words in text, and different
variants of neural network architecture (BiLSTM, hybrid BiLSTM-CRF or
pure CRF) can be used as final classifier (tagger). I recommend to
use a special `TensorFlow Hub ELMo<https://tfhub.dev/google/elmo/2>`_
for English NLP tasks and a `DeepPavlov ELMo
<http://docs.deeppavlov.ai/en/master/apiref/models/embedders.html#deeppavlov.models.embedders.elmo_embedder.ELMoEmbedder>`_
(`http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz
<http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz>`_)
for for same tasks in Russian.

Getting Started
---------------

Installing
~~~~~~~~~~

To install this project on your local machine, you should run the
following commands in Terminal:

.. code::

    git clone https://github.com/bond005/neuro_tagger.git
    cd neuro_tagger
    sudo python setup.py

You can also run the tests:

.. code::

    python setup.py test

But I recommend you to use pip and install this package from PyPi:

.. code::

    pip install neuro_tagger

or (using ``sudo``):

.. code::

    sudo pip install neuro_tagger

Usage
~~~~~

There are two examples of the ``neuro_tagger`` usage in the ``demo``
subdirectory:

1. **demo_brat.py** - example of neuro-tagger creating and its
cross-validated estimating on the labeled text corpus in the ``brat``
format (`brat<http://brat.nlplab.org>`) is popular tool for manual
annotating of texts);

2. **demo_factrueval.py** - example of experiments on the FactRuEval-2016
text corpus, which is part of special competition devoted to named
entity recognition and fact extraction in Russian (it is described in the
paper `FactRuEval 2016: Evaluation of Named Entity Recognition and Fact
Extraction Systems for Russian<http://www.dialog-21.ru/media/3430/starostinaetal.pdf>`).
Use of `special ELMo<http://docs.deeppavlov.ai/en/master/apiref/models/embedders.html#deeppavlov.models.embedders.elmo_embedder.ELMoEmbedder>`
from the `iPavlov project<https://ipavlov.ai>` and CRF as final classifier
allows to reach best result for first track of this competition (F1-score
is greater than 0,89, and it is state-fo-the-art solution for named entity
recognition in Russian).

Note
~~~~~

You have to use short texts such as sentences or small paragraphs,
because long texts will be processed worse. If you train ``neuro_tagger``
on corpus of long texts, then the training can be converged slowly. If
you use the ``neuro_tagger``, trained on short texts, for recognizing of
long text, then only some initial words of this text can be tagged, and
remaining words at the end of text will not be considered by algorithm.
Besides, you need to use a very large volume of RAM for processing of
long texts.

For solving of above-mentioned problem you can split long texts by
shorter sentences using well-known NLP libraries such as `NLTK
<http://www.nltk.org/api/nltk.tokenize.html?highlight=sent_tokenize#nltk.tokenize.sent_tokenize>`
or `SpaCy`<https://spacy.io/api/token#is_sent_start>`. Also, if you
want to correctly split long text with its tag labels, then you can
use the special function ``tokenize_all_by_sentences`` from the module
``neuro_tagger.dataset_loading``.

Acknowledgment
---------------

The work was supported by National Technology Initiative and PAO Sberbank
project ID 0000000007417F630002.
'''

setup(
    name='neuro_tagger',
    version=neuro_tagger.__version__,
    packages=find_packages(exclude=['tests', 'demo']),
    include_package_data=True,
    description='Text tagger, based on the ELMo embeddings and recurrent neural network, with the simple sklearn-like '
                'interface',
    long_description=long_description,
    url='https://github.com/bond005/neuro_tagger',
    author='Ivan Bondarenko',
    author_email='bond005@yandex.ru',
    license='Apache License Version 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords=['elmo', 'ner', 'lstm', 'crf', 'nlp', 'keras', 'tensorflow', 'scikit-learn'],
    install_requires=['h5py', 'keras', 'numpy', 'scikit-learn'],
    test_suite='tests'
)
