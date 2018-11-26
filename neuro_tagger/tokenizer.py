import copy
from typing import List, Tuple

from nltk.tokenize.nist import NISTTokenizer


class BaseTokenizer:
    def tokenize(self, source_text: str) -> List[Tuple[int, int]]:
        raise NotImplemented()


class DefaultTokenizer(BaseTokenizer):
    def __init__(self, max_num_of_nonalnum_tokens: int=3):
        self.tokenizer = NISTTokenizer()
        self.max_num_of_nonalnum_tokens = max_num_of_nonalnum_tokens

    def tokenize(self, source_text: str) -> List[Tuple[int, int]]:
        if self.max_num_of_nonalnum_tokens < 1:
            raise ValueError('`max_num_of_nonalnum_tokens` must be a positive integer number, but {0} is not '
                             'positive!'.format(self.max_num_of_nonalnum_tokens))
        token_bounds_in_text = []
        start_pos = 0
        nonalnum_tokens_counter = 0
        prev_token = ''
        for cur_token in self.tokenizer.international_tokenize(source_text):
            if self.is_alnum(cur_token):
                nonalnum_tokens_counter = 0
            else:
                if cur_token == prev_token:
                    nonalnum_tokens_counter += 1
                else:
                    nonalnum_tokens_counter = 1
            found_pos = source_text.find(cur_token, start_pos)
            if found_pos < 0:
                raise ValueError('Text `{0}` cannot be tokenized! Token `{1}` is not found.'.format(
                    source_text, cur_token))
            if nonalnum_tokens_counter <= self.max_num_of_nonalnum_tokens:
                token_bounds_in_text.append((found_pos, len(cur_token)))
            start_pos = found_pos + len(cur_token)
            prev_token = cur_token
        return token_bounds_in_text

    @staticmethod
    def is_alnum(source_token: str) -> bool:
        if not source_token.isalnum():
            return False
        return (set(source_token) != {'_'})

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.tokenizer = NISTTokenizer()
        result.max_num_of_nonalnum_tokens = self.max_num_of_nonalnum_tokens
        return result

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.tokenizer = NISTTokenizer()
        result.max_num_of_nonalnum_tokens = self.max_num_of_nonalnum_tokens
        return result

    def __getstate__(self):
        return {'max_num_of_nonalnum_tokens': self.max_num_of_nonalnum_tokens}

    def __setstate__(self, state):
        self.tokenizer = NISTTokenizer()
        self.max_num_of_nonalnum_tokens = state['max_num_of_nonalnum_tokens']


class FactRuEvalTokenizer(BaseTokenizer):
    def __init__(self, texts: List[str], bounds_of_tokens_in_texts: List[tuple]):
        n = len(texts)
        if n != len(bounds_of_tokens_in_texts):
            raise ValueError('Number of texts does not correspond to number of token bounds list! {0} != {1}.'.format(
                n, len(bounds_of_tokens_in_texts)))
        self.dictionary_of_texts = dict()
        for idx in range(n):
            self.dictionary_of_texts[texts[idx]] = [(cur[0], cur[1] - cur[0]) for cur in bounds_of_tokens_in_texts[idx]]

    def tokenize(self, source_text: str) -> List[Tuple[int, int]]:
        return self.dictionary_of_texts[source_text]

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.dictionary_of_texts = copy.copy(self.dictionary_of_texts)
        return result

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.dictionary_of_texts = copy.deepcopy(self.dictionary_of_texts)
        return result

    def __getstate__(self):
        return {'dictionary_of_texts': self.dictionary_of_texts}

    def __setstate__(self, state):
        self.dictionary_of_texts = state['dictionary_of_texts']
