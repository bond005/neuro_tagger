from setuptools import setup, find_packages

import neuro_tagger


description = 'Text tagger, based on the ELMo embeddings and recurrent neural network, with the simple sklearn-like ' \
              'interface'


setup(
    name='neuro_tagger',
    version=neuro_tagger.__version__,
    packages=find_packages(exclude=['tests', 'demo']),
    include_package_data=True,
    description=description,
    long_description=description,
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
