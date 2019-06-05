import numpy as np

from tqdm import tqdm
import pickle
from data_utils.replace_sampler import ReplaceSampler


class NoiseInjector(object):

    def __init__(self, shuffle_sigma=0.5,
                 replace_mean=0.1, replace_std=0.03,
                 delete_mean=0.1, delete_std=0.03,
                 add_mean=0.1, add_std=0.03):
        # READ-ONLY, do not modify
        self.shuffle_sigma = shuffle_sigma
        self.replace_a, self.replace_b = self._solve_ab_given_mean_var(replace_mean, replace_std ** 2)
        self.delete_a, self.delete_b = self._solve_ab_given_mean_var(delete_mean, delete_std ** 2)
        self.add_a, self.add_b = self._solve_ab_given_mean_var(add_mean, add_std ** 2)
        # self.protected = set(r"1234567890一二三四五六七八九十百千万两\n 。.，,！!？?“”'：:（）()「」、%$#@~&*-+")
        self.replace_sampler = ReplaceSampler()

    @staticmethod
    def _solve_ab_given_mean_var(mean, var):
        a = mean * mean * (1. - mean) / var - mean
        b = (1. - mean) * (mean * (1. - mean) / var - 1.)
        return a, b

    def _shuffle_func(self, tgt):
        if self.shuffle_sigma < 1e-6:
            return tgt

        shuffle_key = [i + np.random.normal(loc=0, scale=self.shuffle_sigma) for i in range(len(tgt))]
        new_idx = np.argsort(shuffle_key)
        res = [tgt[i] for i in new_idx]

        return res

    def _replace_func(self, tgt):
        replace_ratio = np.random.beta(self.replace_a, self.replace_b)
        ret = []
        rnd = np.random.random(len(tgt))
        ratio_decay = 1.0  # 用于控制不连续修改,若修改则/2,否则*2
        for i, p in enumerate(tgt):
            rnd_word = None
            if (
                    (p[-1] == "O" or p[-1] is None)  # 不修改NER
                    and rnd[i] < replace_ratio * ratio_decay
            ):
                rnd_word = self.replace_sampler.sample(p[1])
            if rnd_word is not None:
                ret.append((-1, rnd_word))
                ratio_decay /= 2
            else:
                ratio_decay = min(ratio_decay*2, 1)
                ret.append((p[0], p[1]))
        return ret

    def _delete_func(self, tgt):
        delete_ratio = np.random.beta(self.delete_a, self.delete_b)
        ret = []
        rnd = np.random.random(len(tgt))
        for i, p in enumerate(tgt):
            if rnd[i] < delete_ratio:
                continue
            ret.append(p)
        return ret

    def _add_func(self, tgt):
        add_ratio = np.random.beta(self.add_a, self.add_b)
        ret = []
        rnd = np.random.random(len(tgt))
        for i, p in enumerate(tgt):
            if rnd[i] < add_ratio:
                rnd_ex = self.corpus[np.random.randint(len(self.corpus))]
                rnd_word = rnd_ex[np.random.randint(len(rnd_ex))]
                ret.append((-1, rnd_word))
            ret.append(p)

        return ret

    def _parse(self, pairs):
        align = []
        art = []
        for si in range(len(pairs)):
            ti = pairs[si][0]
            w = pairs[si][1]
            art.append(w)
            if ti >= 0:
                align.append('{}-{}'.format(si, ti))
        return art, align

    def inject_noise(self, tokens, ner=None):

        # funcs = [self._add_func, self._shuffle_func, self._replace_func, self._delete_func]
        funcs = [self._replace_func]  # 只替换
        np.random.shuffle(funcs)
        if ner is None:
            ner = [None] * len(tokens)
        pairs = [(i, w, ner[i]) for (i, w) in enumerate(tokens)]
        for f in funcs:
            pairs = f(pairs)

        return self._parse(pairs)


def save_file(filename, contents):
    with open(filename, 'w') as ofile:
        for content in contents:
            ofile.write(' '.join(content) + '\n')


def noise(filename, ofile_prefix, noise, limit=None):
    print("loading data")

    # tgts = [tokenize_line(line.strip()) for line in lines]
    # tgts = [line.strip().split() for line in lines[:limit]]
    lines = open(filename).readlines()
    limit = limit or len(lines)
    tgts = [list(line.strip()) for line in lines[:limit]]  # todo:note wiki没有空格

    # 过滤空句
    tgts = [tgt for tgt in tgts if tgt]

    print("data loaded.")
    noise_injector = NoiseInjector(tgts, replace_mean=noise, delete_mean=noise, add_mean=noise, shuffle_sigma=0.1)

    with open('{}.src'.format(ofile_prefix), "w") as src_file:
        with open('{}.tgt'.format(ofile_prefix), "w") as tgt_file:
            with open('{}.forward'.format(ofile_prefix), "w") as align_file:
                with open('{}.gec_src'.format(ofile_prefix), "w") as gec_src_file:
                    with open('{}.gec_tgt'.format(ofile_prefix), "w") as gec_tgt_file:
                        for idx, tgt in tqdm(enumerate(tgts), total=len(tgts)):
                            src, align = noise_injector.inject_noise(tgt)

                            # build gec labels
                            ai = list(map(lambda x: tuple(x.split("-")), align))
                            src_labels = [1] * len(src)
                            tgt_labels = [1] * len(tgt)
                            for sai, tai in ai:
                                if int(tai) >= len(tgt) or int(sai) >= len(src):
                                    continue
                                src_labels[int(sai)] = 0
                                tgt_labels[int(tai)] = 0

                            src_file.write(' '.join(src) + '\n')
                            tgt_file.write(' '.join(tgt) + '\n')
                            align_file.write(' '.join(align) + '\n')
                            gec_src_file.write(' '.join(gec_src_file) + '\n')
                            gec_tgt_file.write(' '.join(gec_tgt_file) + '\n')
                            if idx % 10000 == 0:
                                print(''.join(src))
                                print(''.join(tgt))
                                print("".join([str(x) for x in src_labels]))


def noise_with_ner(filename, ofile_prefix, noise, limit=None, nerfilename=None):
    """
    对提供的filename添加noise，并存储为preprosess.py的输入
    Args:
        filename: text文件，每行一句话，每个字用空格隔开
        ofile_prefix: 输出文件的前缀
        noise: 添加的noise比例
        limit: 只取file中的前多少行
        nerfilename: ner文件，每行一句话，每个tag用空格隔开
        pinyinfile: json文件，由data_utils/stat_pinyin.py生成

    Returns:

    """

    noise_injector = NoiseInjector(replace_mean=noise, delete_mean=noise, add_mean=noise, shuffle_sigma=0.1)
    textfin = open(filename)
    try:
        nerfin = open(nerfilename)
    except FileNotFoundError:
        def fake_nerfin():
            while True:
                yield " ".join(["O"]*1000)
        nerfin = fake_nerfin()
    with open(f'{ofile_prefix}.src', "w") as src_file:
        with open(f'{ofile_prefix}.tgt', "w") as tgt_file:
            with open(f'{ofile_prefix}.forward', "w") as align_file:
                with open(f'{ofile_prefix}.gec_src', "w") as gec_src_file:
                    with open(f'{ofile_prefix}.gec_tgt', "w") as gec_tgt_file:
                        idx = 0
                        for tgt, ner in tqdm(zip(textfin, nerfin)):
                            tgt, ner = tgt.strip(), ner.strip()
                            idx += 1
                            if limit is not None and idx > limit:
                                break
                            if not tgt:
                                continue
                            tgt, ner = tgt.split(), ner.split()
                            src, align = noise_injector.inject_noise(tgt, ner)

                            # build gec labels
                            ai = list(map(lambda x: tuple(x.split("-")), align))
                            src_labels = [1] * len(src)
                            tgt_labels = [1] * len(tgt)
                            for sai, tai in ai:
                                if int(tai) >= len(tgt) or int(sai) >= len(src):
                                    continue
                                src_labels[int(sai)] = 0
                                tgt_labels[int(tai)] = 0
                            src_labels = [str(x) for x in src_labels]
                            tgt_labels = [str(x) for x in tgt_labels]

                            src_file.write(' '.join(src) + '\n')
                            tgt_file.write(' '.join(tgt) + '\n')
                            align_file.write(' '.join(align) + '\n')
                            gec_src_file.write(' '.join(src_labels) + '\n')
                            gec_tgt_file.write(' '.join(tgt_labels) + '\n')
                            if idx % 10000 == 0:
                                print(''.join(src))
                                print(''.join(tgt))
                                print("".join(src_labels))

    textfin.close()
    nerfin.close()
    print(f"data saved to {ofile_prefix}.src/tgt/forward")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=0)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-l', '--limit', type=int, default=1e9)
    parser.add_argument('-n', '--noise', type=float, default=0.15)
    parser.add_argument('--ofile-prefix', type=str, default=f'/data/nfsdata2/yuxian/datasets/gec/zh_giga_20190525')
    parser.add_argument('--txtfile', type=str, default=f'/data/nfsdata2/yuxian/datasets/nlp_corpus/zh_giga_tgt.txt.text')
    parser.add_argument('--nerfile', type=str, default=f'/data/nfsdata2/yuxian/datasets/nlp_corpus/zh_giga_tgt.txt.ner')

    args = parser.parse_args()
    np.random.seed(args.seed)
    print(args)

    noise_with_ner(args.txtfile, f"{args.ofile_prefix}{args.epoch}", noise=args.noise,
                   limit=int(args.limit), nerfilename=args.nerfile)

