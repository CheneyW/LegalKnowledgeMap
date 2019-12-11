#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/8/22
"""

import os
import json
import random
import torch

import jieba

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(DIR_PATH, os.pardir, os.pardir, 'corpus_labeling', 'corpus2.json')

CORPUS_NUM = 500

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SOS_token = 0
EOS_token = 1
SPACE_token = 2


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: " "}
        self.n_words = 3  # Count SOS and EOS

    def add_sent(self, sentence):
        for word in jieba.lcut(sentence):
            self.add_word(word)

    def add_words(self, words):
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def sentence2tensor(self, sentence):
        tensor = [self.word2index[word] for word in sentence]
        tensor.append(EOS_token)
        tensor = torch.tensor(tensor, dtype=torch.long, device=device).view(-1, 1)
        return tensor

    def pair2tensors(self, pair):
        input_tensor = self.sentence2tensor(pair['content'])

        output_tensor = []
        for kw in pair['keywords']:
            output_tensor += [self.word2index[w] for w in kw if self.word2index[w] not in output_tensor]

        output_tensor.append(EOS_token)
        output_tensor = torch.tensor(output_tensor, dtype=torch.long, device=device).view(-1, 1)

        return input_tensor, output_tensor


# 读取训练数据文件
def prepare_data():
    print("Reading train file...")
    with open(CORPUS_PATH, 'r', encoding='utf-8')as f:
        corpus = json.load(f)[:CORPUS_NUM]

    lang = Lang()
    pairs = []
    max_length = 0
    for sample in corpus:
        content = jieba.lcut(sample['content'])
        max_length = max(max_length, len(content))
        keywords = [jieba.lcut(kw) for kw in sample['keywords'].split()]
        lang.add_words(content)
        for kw in keywords:
            lang.add_words(kw)
        pairs.append({'content': content, 'keywords': keywords})

    print("Read %d samples" % len(pairs))
    print("Counted words: %d" % lang.n_words)
    print("Max length: %d" % max_length)
    return lang, pairs, max_length + 1


if __name__ == '__main__':
    lang, pairs, _ = prepare_data()
    print(random.choice(pairs))
