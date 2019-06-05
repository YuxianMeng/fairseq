# encoding: utf-8
"""
@author: Yuxian Meng 
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: utils
@time: 2019/5/17 20:55

    这一行开始写关于本文件的说明与解释
"""


def sep_text(txt, return_index=False, ignore_space=False):
    """
    将一段中文txt切分为句子
    :param txt: str或list of str或list of tuple
    :return: list of str
    """
    sentences = []
    stack = []
    after2before = {'”': '“', '）': '（', }
    end_signals = ['。', '！', '？']
    sentence = []
    l = len(txt)
    for i in range(l):
        ch = txt[i]
        if ignore_space and ch in ['\n', '\t', '\r', '\f', '\v', ' ']:
            continue
        if return_index:
            sentence.append(i)
        else:
            sentence.append(ch)
        if ch in after2before.values():
            stack.append(ch)
        if ch in after2before:
            pre = after2before[ch]
            if pre in stack:  # 因为有的文章有符号错误
                stack.remove(pre)
        if ((ch in end_signals and i < l - 1 and txt[i + 1] not in after2before.keys()) or
            (ch in after2before.keys() and txt[i - 1] in end_signals)) and not stack:
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append(sentence)
    return sentences


if __name__ == '__main__':
    x = "简单得一句话。“带引号得第二句话。”他说：“这要是对了就成功了！”"
    y = sep_text(x)
    y = ["".join(s) for s in y]
    assert y == ["简单得一句话。",
                 "“带引号得第二句话。”",
                 "他说：“这要是对了就成功了！”"]
