import codecs
import copy
import os
import re
from typing import List, Tuple, Union
import warnings

from nltk import sent_tokenize


def tokenize_text(src: str) -> Tuple[str, dict]:
    re_for_tokenization = [re.compile(r'\w:\d', re.U), re.compile(r'\d%\w', re.U), re.compile(r'\w[\\/]\w', re.U),
                           re.compile(r'.\w[\\/]', re.U), re.compile(r'\w\+\w', re.U), re.compile(r'.\w\+\S', re.U)]
    tokenized = src
    indices_of_characters = list(range(len(src)))
    for cur_re in re_for_tokenization:
        search_res = cur_re.search(tokenized)
        while search_res is not None:
            if (search_res.start() < 0) or (search_res.end() < 0):
                search_res = None
            else:
                tokenized = tokenized[:(search_res.start() + 2)] + ' ' + tokenized[(search_res.start() + 2):]
                for char_idx in range(len(indices_of_characters)):
                    if indices_of_characters[char_idx] >= (search_res.start() + 2):
                        indices_of_characters[char_idx] += 1
                search_res = cur_re.search(tokenized, pos=search_res.end() + 1)
    indices_of_characters = dict(
        [(char_idx, indices_of_characters[char_idx]) for char_idx in range(len(indices_of_characters))]
    )
    return tokenized, indices_of_characters


def load_annotation_from_brat(file_name: str, source_text: str) -> List[Tuple[str, int, int]]:
    res = []
    line_idx = 1
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = 'File "{0}": line {1} is wrong!'.format(file_name, line_idx)
                line_parts = prep_line.split('\t')
                if len(line_parts) < 2:
                    raise ValueError(err_msg)
                line_parts = list(filter(lambda it1: len(it1) > 0, map(lambda it2: it2.strip(), line_parts)))
                if len(line_parts) < 2:
                    raise ValueError(err_msg)
                if len(line_parts) >= 3:
                    ne_info = list(
                        filter(lambda it1: len(it1) > 0, map(lambda it2: it2.strip(), line_parts[1].split())))
                    if len(ne_info) < 3:
                        raise ValueError(err_msg + ' "{0}" is wrong description of named entity.'.format(ne_info))
                    ne_type = ne_info[0].strip()
                    if ne_type == 'O':
                        raise ValueError(err_msg + ' "{0}" is inadmissible type of named entity.'.format(ne_info[0]))
                    if len(ne_type) == 0:
                        raise ValueError(err_msg + ' "There is empty type of named entity.')
                    min_ne_start = None
                    max_ne_end = None
                    for bounds_description in filter(lambda it2: len(it2) > 0,
                                                     map(lambda it1: it1.strip(), ' '.join(ne_info[1:]).split(';'))):
                        bounds_parts = bounds_description.split()
                        if len(bounds_parts) != 2:
                            raise ValueError(err_msg + ' "{0}" is wrong description of named entity.'.format(ne_info))
                        try:
                            ne_start = int(bounds_parts[0])
                            ne_end = int(bounds_parts[1])
                            if (ne_start < 0) or (ne_end <= ne_start):
                                ne_start = None
                                ne_end = None
                        except:
                            ne_start = None
                            ne_end = None
                        if (ne_start is None) or (ne_end is None):
                            raise ValueError(err_msg + ' "{0}" is wrong description of named entity.'.format(ne_info))
                        if min_ne_start is None:
                            min_ne_start = ne_start
                        elif ne_start < min_ne_start:
                            min_ne_start = ne_start
                        if max_ne_end is None:
                            max_ne_end = ne_end
                        elif ne_end > max_ne_end:
                            max_ne_end = ne_end
                    if (min_ne_start is None) or (max_ne_end is None):
                        raise ValueError(err_msg + ' "{0}" is wrong description of named entity.'.format(ne_info))
                    ne_text = line_parts[2].strip()
                    if len(ne_text) == 0:
                        warnings.warn(err_msg)
                    else:
                        if max_ne_end > len(source_text):
                            raise ValueError(err_msg + ' Annotation does not correspond to text!')
                        while min_ne_start < len(source_text):
                            if not source_text[min_ne_start].isspace():
                                break
                            min_ne_start += 1
                        if min_ne_start > len(source_text):
                            raise ValueError(err_msg + ' Annotation does not correspond to text!')
                        max_ne_end -= 1
                        while max_ne_end > ne_start:
                            if not source_text[max_ne_end].isspace():
                                break
                            max_ne_end -= 1
                        max_ne_end += 1
                        new_idx = 0
                        while new_idx < len(res):
                            if max_ne_end <= res[new_idx][1]:
                                break
                            new_idx += 1
                        ok = True
                        if (len(res) > 0) and (new_idx > 0):
                            if min_ne_start < res[new_idx - 1][2]:
                                warnings.warn(err_msg)
                                ok = False
                        if ok:
                            res.insert(new_idx, (ne_type, min_ne_start, max_ne_end))
            cur_line = fp.readline()
            line_idx += 1
    if len(res) > 0:
        idx = 1
        while idx < len(res):
            if (res[idx - 1][0] == res[idx][0]) and (source_text[res[idx - 1][2]:res[idx][1]].strip() == ''):
                res[idx - 1] = (res[idx - 1][0], res[idx - 1][1], res[idx][2])
                del res[idx]
            else:
                idx += 1
    return res


def tokenize_all_by_sentences(texts: List[str], labels: List[tuple],
                              language: str='english') -> Tuple[List[str], List[tuple]]:

    def select_labels_in_interval(labels_for_text: tuple, interval_start_idx: int,
                                  interval_length: int) -> Union[tuple, None]:
        ok = True
        label_start_idx = -1
        for label_idx in range(len(labels_for_text)):
            if labels_for_text[label_idx][1] >= interval_start_idx:
                label_start_idx = label_idx
                break
        if label_start_idx < 0:
            return tuple()
        if labels_for_text[label_start_idx][1] >= (interval_start_idx + interval_length):
            return tuple()
        if (labels_for_text[label_start_idx][1] + labels_for_text[label_start_idx][2]) > \
                (interval_start_idx + interval_length):
            return None
        label_end_idx = len(labels_for_text) - 1
        for label_idx in range(label_start_idx + 1, len(labels_for_text)):
            if (labels_for_text[label_idx][1] + labels_for_text[label_idx][2]) >= \
                    (interval_start_idx + interval_length):
                label_end_idx = label_idx - 1
                break
        if label_start_idx > 0:
            if (labels_for_text[label_start_idx - 1][1] + labels_for_text[label_start_idx - 1][2]) > interval_start_idx:
                ok = False
        if not ok:
            if label_end_idx < (len(labels_for_text) - 1):
                if labels_for_text[label_end_idx + 1][1] < (interval_start_idx + interval_length):
                    ok = False
        if not ok:
            return None
        return tuple(map(lambda cur_label: (cur_label[0], cur_label[1] - interval_start_idx, cur_label[2]),
                         labels_for_text[label_start_idx:(label_end_idx + 1)]))

    n = len(texts)
    if n != len(labels):
        raise ValueError('Number of labels does not correspond to number of texts! {0} != {1}'.format(len(labels), n))
    new_texts = []
    new_labels = []
    for idx in range(n):
        sentences = list(filter(
            lambda it2: len(it2) > 0,
            map(lambda it1: it1.strip(), sent_tokenize(texts[idx], language))
        ))
        if len(sentences) > 0:
            start_pos = 0
            sentence_start_pos = -1
            new_texts_ = []
            new_labels_ = []
            for cur_sent in sentences:
                start_pos = texts[idx].find(cur_sent, start_pos)
                if start_pos < 0:
                    raise ValueError('Text {0} cannot be tokenized!'.format(idx))
                if sentence_start_pos < 0:
                    labels_for_sentence = select_labels_in_interval(labels[idx], start_pos, len(cur_sent))
                    if labels_for_sentence is None:
                        sentence_start_pos = start_pos
                    else:
                        new_texts_.append(cur_sent)
                        new_labels_.append(labels_for_sentence)
                else:
                    labels_for_sentence = select_labels_in_interval(labels[idx], sentence_start_pos,
                                                                    start_pos + len(cur_sent) - sentence_start_pos)
                    if labels_for_sentence is not None:
                        new_texts_.append(texts[idx][sentence_start_pos:(start_pos + len(cur_sent))])
                        new_labels_.append(labels_for_sentence)
                        sentence_start_pos = -1
                start_pos += len(cur_sent)
            if sentence_start_pos >= 0:
                new_texts.append(texts[idx])
                new_labels.append(labels[idx])
            else:
                new_texts += new_texts_
                new_labels += new_labels_
        else:
            new_texts.append(texts[idx])
            new_labels.append(labels[idx])
    return new_texts, new_labels


def load_dataset_from_brat(dir_name: str) -> Tuple[List[str], List[tuple]]:
    if not os.path.isdir(os.path.normpath(dir_name)):
        raise ValueError('Directory "{0}" does not exist!'.format(dir_name))
    annotation_files = sorted(list(filter(lambda it: it.lower().endswith('.ann'),
                                          os.listdir(os.path.normpath(dir_name)))))
    text_files = sorted(list(filter(lambda it: it.lower().endswith('.txt'),
                                    os.listdir(os.path.normpath(dir_name)))))
    if len(annotation_files) != len(text_files):
        raise ValueError('Number of annotation files does not equal to number of text files! {0} != {1}.'.format(
            len(annotation_files), len(text_files)))
    pairs_of_files = list()
    for idx in range(len(annotation_files)):
        if annotation_files[idx][:-4].lower() != text_files[idx][:-4].lower():
            raise ValueError('The annotation file "{0}" does not correspond to the text file "{1}"!'.format(
                annotation_files[idx], text_files[idx]))
        pairs_of_files.append((text_files[idx], annotation_files[idx]))
    list_of_texts = list()
    list_of_annotations = list()
    re_for_unicode = re.compile(r'&#\d+;*', re.U)
    for cur_pair in pairs_of_files:
        text_file_name = os.path.join(dir_name, cur_pair[0])
        annotation_file_name = os.path.join(dir_name, cur_pair[1])
        with codecs.open(text_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
            text = ' '.join(filter(lambda it1: len(it1) > 0, map(lambda it2: it2.strip(), fp.readlines())))
        if len(text) == 0:
            raise ValueError('The text file "{0}" is empty!'.format(text_file_name))
        annotation = load_annotation_from_brat(annotation_file_name, text)
        search_res = re_for_unicode.search(text)
        while search_res is not None:
            if (search_res.start() < 0) or (search_res.end() < 0):
                search_res = None
            else:
                start_pos = search_res.start() + 2
                end_pos = search_res.end()
                n_old = search_res.end() - search_res.start()
                while end_pos > (start_pos + 1):
                    if text[end_pos - 1] != ';':
                        break
                    end_pos -= 1
                new_char = chr(int(text[start_pos:end_pos]))
                text = text[:search_res.start()] + new_char + text[search_res.end():]
                for ne_idx in range(len(annotation)):
                    if annotation[ne_idx][1] >= search_res.end():
                        annotation[ne_idx] = (
                            annotation[ne_idx][0],
                            annotation[ne_idx][1] - (n_old - 1),
                            annotation[ne_idx][2]
                        )
                    elif annotation[ne_idx][1] >= search_res.start():
                        annotation[ne_idx] = (annotation[ne_idx][0], search_res.start(), annotation[ne_idx][2])
                    if annotation[ne_idx][2] >= search_res.end():
                        annotation[ne_idx] = (
                            annotation[ne_idx][0],
                            annotation[ne_idx][1],
                            annotation[ne_idx][2] - (n_old - 1)
                        )
                    elif annotation[ne_idx][2] > search_res.start():
                        annotation[ne_idx] = (annotation[ne_idx][0], annotation[ne_idx][1], search_res.start() + 1)
                search_res = re_for_unicode.search(text, pos=search_res.start() + 1)
        prepared_text, indices_of_characters = tokenize_text(text)
        for ne_idx in range(len(annotation)):
            annotation[ne_idx] = (
                annotation[ne_idx][0],
                indices_of_characters[annotation[ne_idx][1]],
                indices_of_characters[annotation[ne_idx][2] - 1] + 1
            )
        list_of_texts.append(prepared_text)
        list_of_annotations.append(tuple(map(lambda it: (it[0], it[1], it[2] - it[1]), annotation)))
    subdirectories = sorted(list(filter(
        lambda it1: os.path.isdir(it1),
        map(
            lambda it2: os.path.join(dir_name, it2),
            filter(lambda it3: it3 not in {'.', '..'}, os.listdir(dir_name))
        )
    )))
    if len(subdirectories) > 0:
        for cur_subdir in subdirectories:
            subdir_list_of_texts, subdir_list_of_annotations = load_dataset_from_brat(cur_subdir)
            list_of_texts += subdir_list_of_texts
            list_of_annotations += subdir_list_of_annotations
    return list_of_texts, list_of_annotations


def load_document_from_factrueval2016(tokens_file_name: str, spans_file_name: str,
                                      objects_file_name: str) -> Tuple[List[str], List[tuple]]:
    texts = []
    new_text = []
    tokens_dict = dict()
    total_text_len = 0
    new_text_len = 0
    with codecs.open(tokens_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        line_idx = 1
        cur_line = fp.readline()
        while len(cur_line) > 0:
            err_msg = 'File `{0}`: line {1} is wrong!'.format(tokens_file_name, line_idx)
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                token_description = prep_line.split()
                if len(token_description) != 4:
                    raise ValueError(err_msg)
                if (not token_description[0].isdigit()) or (not token_description[1].isdigit()) or \
                        (not token_description[2].isdigit()):
                    raise ValueError(err_msg)
                token_id = int(token_description[0])
                if token_id in tokens_dict:
                    raise ValueError(err_msg)
                new_text.append(token_id)
                token_start = int(token_description[1])
                token_length = int(token_description[2])
                tokens_dict[token_id] = (token_description[-1], 'O', token_start - total_text_len, token_length,
                                         (len(texts), len(new_text) - 1))
                new_text_len = tokens_dict[token_id][2] + tokens_dict[token_id][3]
            else:
                if len(new_text) == 0:
                    raise ValueError(err_msg)
                texts.append(copy.copy(new_text))
                new_text.clear()
                total_text_len += new_text_len
                new_text_len = 0
            cur_line = fp.readline()
            line_idx += 1
    if len(new_text) > 0:
        texts.append(copy.copy(new_text))
        new_text.clear()
    spans_dict = dict()
    with codecs.open(spans_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        line_idx = 1
        cur_line = fp.readline()
        while len(cur_line) > 0:
            err_msg = 'File `{0}`: line {1} is wrong!'.format(spans_file_name, line_idx)
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                comment_pos = prep_line.find('#')
                if comment_pos < 0:
                    raise ValueError(err_msg)
                prep_line = prep_line[:comment_pos].strip()
                if len(prep_line) == 0:
                    raise ValueError(err_msg)
                span_description = prep_line.split()
                if len(span_description) != 6:
                    raise ValueError(err_msg)
                if (not span_description[0].isdigit()) or (not span_description[-1].isdigit()) or \
                        (not span_description[-2].isdigit()):
                    raise ValueError(err_msg)
                span_id = int(span_description[0])
                token_IDs = list()
                start_token_id = int(span_description[-2])
                n_tokens = int(span_description[-1])
                if (n_tokens <= 0) or (start_token_id not in tokens_dict):
                    raise ValueError(err_msg)
                text_idx = tokens_dict[start_token_id][4][0]
                token_pos_in_text = tokens_dict[start_token_id][4][1]
                for idx in range(n_tokens):
                    token_id = texts[text_idx][token_pos_in_text + idx]
                    if token_id not in tokens_dict:
                        raise ValueError(err_msg)
                    token_IDs.append(token_id)
                if span_id not in spans_dict:
                    spans_dict[span_id] = tuple(token_IDs)
            cur_line = fp.readline()
            line_idx += 1
    with codecs.open(objects_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        line_idx = 1
        cur_line = fp.readline()
        while len(cur_line) > 0:
            err_msg = 'File `{0}`: line {1} is wrong!'.format(objects_file_name, line_idx)
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                comment_pos = prep_line.find('#')
                if comment_pos < 0:
                    raise ValueError(err_msg)
                prep_line = prep_line[:comment_pos].strip()
                if len(prep_line) == 0:
                    raise ValueError(err_msg)
                object_description = prep_line.split()
                if len(object_description) < 3:
                    raise ValueError(err_msg)
                if object_description[1] not in {'LocOrg', 'Org', 'Person', 'Location'}:
                    warnings.warn(err_msg + ' The entity `{0}` is unknown.'.format(object_description[1]))
                else:
                    span_IDs = []
                    for idx in range(2, len(object_description)):
                        if not object_description[idx].isdigit():
                            raise ValueError(err_msg)
                        span_id = int(object_description[idx])
                        if span_id not in spans_dict:
                            raise ValueError(err_msg)
                        span_IDs.append(span_id)
                    span_IDs.sort(key=lambda span_id: tokens_dict[spans_dict[span_id][0]][2])
                    token_IDs = []
                    for span_id in span_IDs:
                        start_token_id = spans_dict[span_id][0]
                        end_token_id = spans_dict[span_id][-1]
                        text_idx = tokens_dict[start_token_id][4][0]
                        token_pos_in_text = tokens_dict[start_token_id][4][1]
                        while token_pos_in_text < len(texts[text_idx]):
                            token_id = texts[text_idx][token_pos_in_text]
                            token_IDs.append(token_id)
                            if token_id == end_token_id:
                                break
                            token_pos_in_text += 1
                        if token_pos_in_text >= len(texts[text_idx]):
                            raise ValueError(err_msg)
                    if object_description[1] in {'LocOrg', 'Location'}:
                        class_label = 'LOC'
                    elif object_description[1] == 'Person':
                        class_label = 'PER'
                    else:
                        class_label = 'ORG'
                    tokens_are_used = False
                    if tokens_dict[token_IDs[0]][1] != 'O':
                        tokens_are_used = True
                    else:
                        for token_id in token_IDs[1:]:
                            if tokens_dict[token_id][1] != 'O':
                                tokens_are_used = True
                                break
                    if not tokens_are_used:
                        tokens_dict[token_IDs[0]] = (
                            tokens_dict[token_IDs[0]][0], 'B-' + class_label,
                            tokens_dict[token_IDs[0]][2], tokens_dict[token_IDs[0]][3],
                            tokens_dict[token_IDs[0]][4]
                        )
                        for token_id in token_IDs[1:]:
                            tokens_dict[token_id] = (
                                tokens_dict[token_id][0], 'I-' + class_label,
                                tokens_dict[token_id][2], tokens_dict[token_id][3],
                                tokens_dict[token_id][4]
                            )
            cur_line = fp.readline()
            line_idx += 1
    list_of_texts = []
    list_of_labels = []
    for tokens_sequence in texts:
        new_text = ''
        new_labels_sequence = []
        start_idx = -1
        end_idx = -1
        ne_type = ''
        for token_id in tokens_sequence:
            while len(new_text) < tokens_dict[token_id][2]:
                new_text += ' '
            new_text += tokens_dict[token_id][0]
            if tokens_dict[token_id][1] == 'O':
                if ne_type != '':
                    new_labels_sequence.append((ne_type, start_idx, end_idx - start_idx))
                    start_idx = -1
                    end_idx = -1
                    ne_type = ''
            else:
                if tokens_dict[token_id][1].startswith('B-'):
                    if ne_type != '':
                        new_labels_sequence.append((ne_type, start_idx, end_idx - start_idx))
                    start_idx = tokens_dict[token_id][2]
                    end_idx = start_idx + tokens_dict[token_id][3]
                    ne_type = tokens_dict[token_id][1][2:]
                else:
                    end_idx = tokens_dict[token_id][2] + tokens_dict[token_id][3]
        if ne_type != '':
            new_labels_sequence.append((ne_type, start_idx, end_idx - start_idx))
        list_of_labels.append(tuple(new_labels_sequence))
        list_of_texts.append(new_text)
    return list_of_texts, list_of_labels


def load_dataset_from_factrueval2016(dir_name: str) -> Tuple[List[str], List[tuple], List[str]]:

    def get_file_ID(src_name: str) -> int:
        point_pos_ = src_name.find('.')
        if point_pos_ <= 5:
            raise ValueError('The file name `{0}` is wrong!'.format(src_name))
        return int(src_name[5:point_pos_])

    names_of_files = sorted(
        list(filter(lambda it: it.startswith('book_'), os.listdir(dir_name))),
        key=lambda it: (get_file_ID(it), it)
    )
    if len(names_of_files) == 0:
        raise ValueError('The directory `{0}` is empty!'.format(dir_name))
    if (len(names_of_files) % 6) != 0:
        raise ValueError('The directory `{0}` contains wrong data!'.format(dir_name))
    list_of_all_texts = []
    list_of_all_labels = []
    list_of_all_names = []
    for idx in range(len(names_of_files) // 6):
        base_name = names_of_files[idx * 6]
        point_pos = base_name.rfind('.')
        if point_pos <= 0:
            raise ValueError('The file `{0}` has incorrect name.'.format(base_name))
        prepared_base_name = base_name[:point_pos].strip()
        if len(prepared_base_name) == 0:
            raise ValueError('The file `{0}` has incorrect name.'.format(base_name))
        tokens_file_name = os.path.join(dir_name, prepared_base_name + '.tokens')
        if not os.path.isfile(tokens_file_name):
            raise ValueError('The file `{0}` does not exist!'.format(tokens_file_name))
        spans_file_name = os.path.join(dir_name, prepared_base_name + '.spans')
        if not os.path.isfile(spans_file_name):
            raise ValueError('The file `{0}` does not exist!'.format(spans_file_name))
        objects_file_name = os.path.join(dir_name, prepared_base_name + '.objects')
        if not os.path.isfile(objects_file_name):
            raise ValueError('The file `{0}` does not exist!'.format(objects_file_name))
        list_of_texts, list_of_labels = load_document_from_factrueval2016(tokens_file_name, spans_file_name,
                                                                          objects_file_name)
        list_of_all_texts += list_of_texts
        list_of_all_labels += list_of_labels
        list_of_all_names += [prepared_base_name] * len(list_of_texts)
    return list_of_all_texts, list_of_all_labels, list_of_all_names
