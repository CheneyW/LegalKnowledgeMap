#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/9/2
"""
import re
import json
import os

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
LAW_WORD_PATH = os.path.join(DIR_PATH, os.pardir, 'data', 'law_word', 'law_word.txt')

porn_pattern = r'\s*(英\s*[/\[].+?[/\]])\s*(美\s*[/\[].+?[/\]])'


class LawWord(object):

    def __init__(self, lines):
        self.name = lines[0]
        self.pronunciation = list(re.match(porn_pattern, lines[1]).groups())  # 发音
        self.pos = []  # 词性

        attr_rows, attrs = [], []
        for i, s in enumerate(lines):
            m = re.match(r'【(.+)】', s)
            if m:
                attr_rows.append(i)
                attrs.append(m.group(1))
        attr_range = list(zip(attr_rows, attr_rows[1:] + [len(lines)]))

        # 解析英文释义
        try:
            begin, end = attr_range[attrs.index('英文释义')]
            self.en_interpretation = lines[begin + 1:end]
        except ValueError:
            self.en_interpretation = []

        # 解析中文释义、词性
        try:
            begin, end = attr_range[attrs.index('中文释义')]
            self.cn_interpretation = lines[begin + 1:end]
            for s in self.cn_interpretation:
                self.pos += re.findall(r'[a-z]+\.', s)
        except ValueError:
            self.cn_interpretation = []

        # 解析词组
        try:
            begin, end = attr_range[attrs.index('词组')]
            self.phrase = []
            for s in lines[begin + 1:end]:
                cn_index = 0
                for i, ch in enumerate(s):
                    if not is_letter(ch):
                        cn_index = i
                        break
                self.phrase.append([s[:cn_index], s[cn_index:]])
        except ValueError:
            self.phrase = []

        # 解析例句
        try:
            begin, end = attr_range[attrs.index('例句')]
            example_rows = [begin + 1 + i for i, s in enumerate(lines[begin + 1:end]) if s.startswith('例')]
            example_range = list(zip(example_rows, example_rows[1:] + [end]))
            self.example = [lines[a + 1:b] for a, b in example_range]
            for i, e in enumerate(self.example):
                if len(e) % 2 != 0:
                    print(e)
                if len(e) != 2:
                    t = int(len(e) / 2)
                    self.example[i] = ['\n'.join(e[:t]), '\n'.join(e[t:])]
        except ValueError:
            self.example = []

        self.cn_translation = []  # 中文译文

    def dump(self):
        return {'name': self.name,
                'pronunciation': self.pronunciation,
                'pos': self.pos,
                'en_interpretation': self.en_interpretation,
                'cn_interpretation': self.cn_interpretation,
                'cn_translation': self.cn_translation,
                'phrase': self.phrase,
                'example': self.example,
                }


def parse():
    with open(LAW_WORD_PATH, 'r', encoding='utf-8')as f:
        lines = [s.strip() for s in f.readlines()]
    lines = [s for s in lines if s]

    # 每个词的起始行数
    word_rows = [i - 1 for i, s in enumerate(lines) if re.match(porn_pattern, s)]
    # 每个词的行数范围
    word_range = list(zip(word_rows, word_rows[1:] + [len(lines)]))

    words = [LawWord(lines[begin:end]) for begin, end in word_range]
    print('find %d words' % len(words))

    with open('data.json', 'w', encoding='utf-8')as f:
        json.dump([w.dump() for w in words], f, indent=4, ensure_ascii=False)


def is_letter(ch):
    return 'a' <= ch <= 'z' or 'A' <= ch <= 'Z' or ch == ' '


if __name__ == '__main__':
    parse()
