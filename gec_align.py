# encoding: utf-8
"""
@author: Yuxian Meng 
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: align
@time: 2019/5/23 16:28

    独立生成gec align文件，并且支持多进程
"""
import argparse
import os
from multiprocessing import Pool

import torch

from fairseq.binarizer import Binarizer
from fairseq.binarizer import safe_readline
from fairseq.data import indexed_dataset
from fairseq.tokenizer import tokenize_line


def make_binary_dataset(input_file, output_prefix, num_workers, append_eos=True):
    offsets = Binarizer.find_offsets(input_file, num_workers)
    pool = None
    if num_workers > 1:
        pool = Pool(processes=num_workers - 1)
        for worker_id in range(1, num_workers):
            prefix = "{}{}".format(output_prefix, worker_id)
            pool.apply_async(
                binarize,
                (
                    input_file,
                    prefix,
                    offsets[worker_id],
                    offsets[worker_id + 1],
                    append_eos
                ),
            )
        pool.close()

    # worker 0
    ds = indexed_dataset.IndexedDatasetBuilder(output_prefix + ".label.bin")
    first_end = offsets[1]
    with open(input_file, 'r', encoding='utf-8') as f:
        f.seek(0)
        # next(f) breaks f.tell(), hence readline() must be used
        line = safe_readline(f)
        while line:
            if f.tell() > first_end > 0:
                break
            words = [int(x) for x in tokenize_line(line)]
            if append_eos:
                words.append(0)
            ids = torch.IntTensor(words)
            ds.add_item(ids)
            line = f.readline()

    if num_workers > 1:
        pool.join()
        for worker_id in range(1, num_workers):
            prefix = "{}{}".format(output_prefix, worker_id)
            temp_file_path = prefix + ".label"
            ds.merge_file_(temp_file_path)
            os.remove(indexed_dataset.data_file_path(temp_file_path))
            os.remove(indexed_dataset.index_file_path(temp_file_path))

    ds.finalize(output_prefix + ".label.idx")


def binarize(filename, output_prefix, offset, end, append_eos=True,
             line_tokenizer=tokenize_line):
    ds = indexed_dataset.IndexedDatasetBuilder(output_prefix + ".label.bin")

    def consumer(tensor):
        ds.add_item(tensor)

    with open(filename, 'r', encoding='utf-8') as f:
        f.seek(offset)
        # next(f) breaks f.tell(), hence readline() must be used
        line = safe_readline(f)

        while line:
            if f.tell() > end > 0:
                break
            words = [int(x) for x in line_tokenizer(line)]
            if append_eos:
                words.append(0)
            ids = torch.IntTensor(words)
            consumer(ids)
            line = f.readline()

    ds.finalize(output_prefix + ".label.idx")
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-file', type=str,
                        default="/data/nfsdata/data/yuxian/datasets/gec/wiki/zh_giga_test_s.gec_src")
    parser.add_argument('--output-prefix', type=str,
                        default="/data/nfsdata/data/yuxian/datasets/gec/wiki/zh_giga_test_s/data_bin/train.src-tgt.gec_src")
    parser.add_argument('--nworkers', type=int, default=1)
    parser.add_argument('--append_eos', action="store_true")

    args = parser.parse_args()

    make_binary_dataset(args.input_file, args.output_prefix, args.nworkers, append_eos=True)
