# encoding: utf-8
"""
@author: Yuxian Meng 
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: extract_wiki_data
@time: 2019/5/22 21:01

    这一行开始写关于本文件的说明与解释
"""

#  读取wiki数据
import os
import json
from data_utils.utils import sep_text


minimun_length = 10
eos = set("。？！”")
wiki_dir = "/data/nfsdata/data/yuxian/datasets/nlp_corpus/wiki_zh"
save_path = "/data/nfsdata/data/yuxian/datasets/gec/zh_wiki_test.txt"
n_sents = 0
with open(save_path, "w") as fout:
    sub_dirs = os.listdir(wiki_dir)
    for sub_dir in sub_dirs[:1]:
        data_dir = os.path.join(wiki_dir, sub_dir)
        wiki_files = os.listdir(data_dir)
        for wiki_file in wiki_files:
            wiki_file = os.path.join(data_dir, wiki_file)
            with open(wiki_file) as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    text = data["text"]
                    texts = text.split("\n\n")
                    for para in texts:
                        sents = sep_text(para)
                        for sent in sents:
                            sent = " ".join(sent)
                            if len(sent) > minimun_length and sent[-1] in eos:
                                fout.write(sent+"\n")
                                n_sents += 1
        print(sub_dir, n_sents)
