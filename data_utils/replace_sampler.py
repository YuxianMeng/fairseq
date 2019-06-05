# encoding: utf-8
"""
@author: Yuxian Meng 
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: replace_sampler
@time: 2019/5/28 20:16

    这一行开始写关于本文件的说明与解释
"""

import json
import pickle
import re

import numpy as np


class ReplaceSampler:
    def __init__(self, freq_file: str = "/data/nfsdata/data/yuxian/datasets/gec/freq.json",
                 pinyin_file: str = "/data/nfsdata/data/yuxian/datasets/gec/wiki_dict.src.txt.pinyin",
                 uniform_ratio: float = 0.2, pinyin_ratio: float = 0.4, freq_ratio: float = 0.4, prefetch=True):
        assert uniform_ratio + pinyin_ratio + freq_ratio == 1.0, "三种分布的和应该为1"
        self.str2freq = json.load(open(freq_file))
        self.prefetch = prefetch

        pinyin_stats = pickle.load(open(pinyin_file, "rb"))
        self.char2pinyin, self.pinyin2char = pinyin_stats["char2pinyin"], pinyin_stats["pinyin2char"]

        # 用freq file初始化替换词表
        self.chars = sorted(list(set(self.str2freq.keys())))  # & set(self.char2pinyin.keys())))
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.num_chars = num_chars = len(self.chars)

        # 纯随机的采样分布
        self.uniform_ratio = uniform_ratio
        self.uniform_distribution = np.ones([num_chars])/num_chars

        #  基于频率的采样分布
        self.freq_ratio = freq_ratio
        self.freq_distribution = np.array([self.str2freq[char] for char in self.chars])
        # np.repeat(np.array([self.str2freq[char] for char in self.chars]).reshape(1, -1),
        #                                    repeats=num_chars, axis=0)

        #  基于拼音的采样分布
        self.pinyin_ratio = pinyin_ratio
        self.pinyin_distribution = np.ones([num_chars, num_chars]) / num_chars
        for idx, char in enumerate(self.chars):
            char_pinyin = self.char2pinyin.get(char, None)
            if char_pinyin is not None:
                char_neighbors = []
                replace_distribution = np.zeros([num_chars])
                for i in range(1, 6):
                    char_pinyin = (char_pinyin[0], str(i))
                    pinyin_neighbors = self.pinyin2char.get(char_pinyin, [])
                    char_neighbors += [n for n in pinyin_neighbors if n in self.char2idx]
                neighbor_num = len(char_neighbors)
                char_neighbors = sorted(char_neighbors, key=lambda x: self.freq_distribution[self.char2idx[x]],
                                        reverse=True)[: min(max(10, neighbor_num), 25)]
                # print(char_neighbors)
                for neighbor in char_neighbors:
                    replace_distribution[self.char2idx[neighbor]] = 1
                    replace_distribution /= replace_distribution.sum()
                    self.pinyin_distribution[idx] = replace_distribution

        self.distribution = (
                self.uniform_ratio * self.uniform_distribution +
                self.pinyin_ratio * self.pinyin_distribution +
                self.freq_ratio * self.freq_distribution
        )

        # 基于规则的采样分布

        # 的地得
        dedide = self.char2idx["的"], self.char2idx["地"], self.char2idx["得"]
        replace_distribution = np.zeros([num_chars])
        for char_idx in dedide:
            replace_distribution[char_idx] = 1/3

        for idx, char_idx in enumerate(dedide):
            self.distribution[char_idx] = replace_distribution

        ones = np.ones([num_chars])
        assert np.allclose(np.sum(self.distribution, axis=1), ones), "final distribution should have sum 1 along axis 1"

        # 并行random，不然太慢了
        self.cache_choice = dict()

        self.protected = set("一二三四五六七八九十百千万两") | \
                         set(char for char in self.chars if re.search("[^\u4E00-\u9FA5]", char))

    def sample(self, char: str) -> str or None:
        """随机sample一个char用于替换输入的char"""
        if char in self.protected:  # 不替换数字或其他特殊符号
            return
        char_idx = self.char2idx.get(char, None)
        if char_idx is not None:
            if self.prefetch:  # 去cache取
                if char_idx not in self.cache_choice or self.cache_choice[char_idx] == []:
                    self.cache_choice[char_idx] = list(np.random.choice(self.num_chars, 100, p=self.distribution[char_idx]))
                idx = self.cache_choice[char_idx].pop()
            else:
                idx = np.random.choice(self.num_chars, 1, p=self.distribution[char_idx])[0]
            replace = self.chars[idx]
            if replace in self.protected:  # 不替换为数字或其它特殊符号
                return
            return self.chars[idx]
        return


if __name__ == '__main__':
    s = ReplaceSampler(prefetch=False)
    chars = "我随便说一句话请帮我替换一下好不好呀呀呀呀呀得得得得得的的的的地地"
    replace = "".join(str(s.sample(char)) for char in chars)
    print(chars)
    print(replace)


