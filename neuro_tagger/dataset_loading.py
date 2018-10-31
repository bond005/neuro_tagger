import codecs
import os
import re
from typing import List, Tuple


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


def load_annotation(file_name: str, source_text: str) -> List[Tuple[str, int, int]]:
    res = []
    line_idx = 1
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = 'File "{0}": line {1} is wrong!'.format(file_name, line_idx)
                line_parts = prep_line.split('\t')
                if len(line_parts) != 3:
                    raise ValueError(err_msg)
                line_parts = list(filter(lambda it1: len(it1) > 0, map(lambda it2: it2.strip(), line_parts)))
                if len(line_parts) != 3:
                    raise ValueError(err_msg)
                ne_info = list(filter(lambda it1: len(it1) > 0, map(lambda it2: it2.strip(), line_parts[1].split())))
                if len(ne_info) != 3:
                    raise ValueError(err_msg + ' "{0}" is wrong description of named entity.'.format(ne_info))
                ne_type = ne_info[0].strip()
                if ne_type == 'O':
                    raise ValueError(err_msg + ' "{0}" is inadmissible type of named entity.'.format(ne_info[0]))
                if len(ne_type) == 0:
                    raise ValueError(err_msg + ' "There is empty type of named entity.')
                try:
                    ne_start = int(ne_info[1])
                    ne_end = int(ne_info[2])
                    if (ne_start < 0) or (ne_end <= ne_start):
                        ne_start = None
                        ne_end = None
                except:
                    ne_start = None
                    ne_end = None
                if (ne_start is None) or (ne_end is None):
                    raise ValueError(err_msg)
                ne_text = line_parts[2].strip()
                if len(ne_text) == 0:
                    raise ValueError(err_msg)
                if ne_end > len(source_text):
                    raise ValueError(err_msg + ' Annotation does not correspond to text!')
                while ne_start < len(source_text):
                    if not source_text[ne_start].isspace():
                        break
                    ne_start += 1
                if ne_start > len(source_text):
                    raise ValueError(err_msg + ' Annotation does not correspond to text!')
                ne_end -= 1
                while ne_end > ne_start:
                    if not source_text[ne_end].isspace():
                        break
                    ne_end -= 1
                ne_end += 1
                if source_text[ne_start:ne_end] != ne_text:
                    raise ValueError(err_msg + ' Annotation does not correspond to text!')
                new_idx = 0
                while new_idx < len(res):
                    if ne_end <= res[new_idx][1]:
                        break
                    new_idx += 1
                if (len(res) > 0) and (new_idx > 0):
                    if ne_start < res[new_idx - 1][2]:
                        raise ValueError(err_msg)
                res.insert(new_idx, (ne_type, ne_start, ne_end))
            cur_line = fp.readline()
            line_idx += 1
    return res


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
        annotation = load_annotation(annotation_file_name, text)
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
