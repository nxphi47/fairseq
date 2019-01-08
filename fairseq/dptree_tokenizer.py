# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import os, re

import torch
from multiprocessing import Pool
from nltk import Tree

from fairseq.dptree import tree_process

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class DPTreeTokenizer(object):

    @staticmethod
    def line2example(s, consumer, tokenize=tokenize_line,
                     append_eos=False, reverse_order=False,
                     offset=0, end=-1):
        tree_string = s
        tree_line = Tree.fromstring(tree_string)
        tree_process.padding_leaves(tree_line)
        tree_line = tree_process.clean_node(tree_line)
        line_leaves, line_matrix, line_node_label, line_leaf_label = tree_process.tree2matrix(tree_line)
        line_node, line_label, line_index = tree_process.generate_data(tree_line)

        node_indices = DPTreeTokenizer.tokenize(
            words=line_node,
            dict=dict,
            tokenize=tokenize,
            add_if_not_exist=False,
            consumer=consumer,
            append_eos=append_eos,
            reverse_order=reverse_order,
        )

        labels_indices = DPTreeTokenizer.tokenize(
            words=line_label,
            dict=dict,
            tokenize=tokenize,
            add_if_not_exist=False,
            consumer=consumer,
            append_eos=append_eos,
            reverse_order=reverse_order,
        )

        # tree_length = len(line_node)
        line_length = len(line_leaves)

        # line_node += [0]
        # line_label += [0]
        # line_index += [[line_length, line_length]]

        # TODO: add pads
        # FIXME: MUST CHECK pad_index = 1 systematically!
        pad_index = 1
        node_indices = torch.cat([node_indices, torch.tensor([pad_index]).int()], 0)
        labels_indices = torch.cat([labels_indices, torch.tensor([pad_index]).int()], 0)
        line_index += [[line_length, line_length]]

        line_indices = torch.tensor(line_index).int()
        line_len = torch.tensor([line_indices]).int()

        example = {
            "nodes": node_indices,
            "labels": labels_indices,
            "indices": line_indices,
            "length": line_len
        }
        return example

    @staticmethod
    def add_file_to_dictionary_single_worker(filename, tokenize, eos_word, worker_id=0, num_workers=1):
        counter = Counter()
        with open(filename, 'r') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)  # drop first incomplete line
            line = f.readline()
            while line:
                for word in tokenize(line):
                    counter.update([word])
                counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):
        def merge_result(counter):
            for w, c in counter.items():
                dict.add_symbol(w, c)

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(pool.apply_async(
                    DPTreeTokenizer.add_file_to_dictionary_single_worker,
                    (filename, tokenize, dict.eos_word, worker_id, num_workers)
                ))
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(DPTreeTokenizer.add_file_to_dictionary_single_worker(filename, tokenize, dict.eos_word))

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line,
                 append_eos=False, reverse_order=False,
                 offset=0, end=-1):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, 'r') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                # asdasd
                # ids = DPTreeTokenizer.tokenize(
                #     line=line,
                #     dict=dict,
                #     tokenize=tokenize,
                #     add_if_not_exist=False,
                #     consumer=replaced_consumer,
                #     append_eos=append_eos,
                #     reverse_order=reverse_order,
                # )
                example = DPTreeTokenizer.line2example(
                    s=line,
                    consumer=replaced_consumer,
                    tokenize=tokenize,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                    offset=offset,
                    end=end
                )
                nseq += 1
                ntok += len(example['nodes'])
                consumer(example)
                line = f.readline()
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(filename, 'r') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets

    @staticmethod
    def tokenize(words, dict, tokenize=tokenize_line, add_if_not_exist=True,
                 consumer=None, append_eos=True, reverse_order=False):
        # words = tokenize(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = dict.add_symbol(word)
            else:
                idx = dict.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = dict.eos_index
        return ids
