# encoding: utf-8
"""
@author: Yuxian Meng 
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: extract_gigaword
@time: 2019/5/17 20:51

    这一行开始写关于本文件的说明与解释
"""

import os
from typing import List
from data_utils.utils import sep_text
from lxml import etree
from hanziconv import HanziConv


def valid(s: str):
    if s[-1] in set("。？！”…") and len(s) > 8:
        return True
    return False


def parse(x: str) -> List[str]:
    data = etree.HTML(x)
    paras = data.xpath('//doc[@type="story"]/text/p')
    sentences: List[str] = ["".join(s) for p in paras for s in sep_text(p.text.strip(), ignore_space=True)]
    return [HanziConv.toSimplified(s) for s in sentences if valid(s)]


if __name__ == '__main__':
    from tqdm import tqdm
    data_dir = "/data/nfsdata/nlp/datasets/language_modeling/chinese_gigaword/all"
    files = os.listdir(data_dir)
    ofile = "/data/nfsdata/data/yuxian/datasets/gec/zh_giga_tgt.txt"
    with open(ofile, "w") as fout:
        for file in tqdm(files[:]):
            file_path = os.path.join(data_dir, file)
            with open(file_path, "r") as fin:
                try:
                    sentences = parse(fin.read())
                    fout.writelines("\n".join(sentences))
                except Exception as e:
                    print(file_path, e)
                # todo: 解决个别coding的问题



