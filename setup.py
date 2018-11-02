from setuptools import setup, find_packages

import neuro_tagger

long_description = '''
NeuroTagger
============

NeuroTagger is a text tagger based on recurrent neural network. It can
be used as NER, dependency parser, morphoanalyzer etc.

The goal of this project is creation of a simple Python package with
the sklearn-like interface for solution of different tasks of text
tagging (named entity recognition, dependency parsing, etc) in case
number of labeled texts is very small (not greater than several
thousands). Special word embeddings named as `ELMo<https://arxiv.org/abs/1802.05365>`_
(Embeddings from Language Models) ensure this possibility, because these
embeddings are contextual and they allow to design more simple and
separable feature space for words in texts.

ELMo embeddings are used as features of words in text, and different
variants of neural network architecture (BiLSTM, hybrid BiLSTM-CRF or
pure CRF) can be used as final classifier (tagger). I recommend to
use a special `TensorFlow Hub ELMo<>https://tfhub.dev/google/elmo/2`_
for English NLP tasks and a `DeepPavlov ELMo
<http://docs.deeppavlov.ai/en/master/apiref/models/embedders.html#deeppavlov.models.embedders.elmo_embedder.ELMoEmbedder>`_
for for same tasks in Russian.

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
