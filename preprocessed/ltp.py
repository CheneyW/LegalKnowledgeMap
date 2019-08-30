#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/7/8
"""
import os
from pyltp import Segmentor, Postagger, NamedEntityRecognizer, Parser

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data', 'ltp_model')


class LTP(object):
    def __init__(self, seg=True, pos=False, ner=False, parser=False):
        # 停用词
        with open(os.path.join(MODEL_PATH, 'stopwords.txt'), 'r', encoding='utf-8') as f:
            self.stopwords = set(x.strip() for x in f.readlines())
        # 分词
        self.segmentor = Segmentor()  # 初始化实例
        # 词性标注
        self.postagger = Postagger()  # 初始化实例
        # 命名实体识别
        self.recognizer = NamedEntityRecognizer()
        # 依存句法分析
        self.parser = Parser()

        # 加载模型
        if seg:
            self.segmentor.load(os.path.join(MODEL_PATH, 'cws.model'))
        if pos:
            self.postagger.load(os.path.join(MODEL_PATH, 'pos.model'))
        if ner:
            self.recognizer.load(os.path.join(MODEL_PATH, 'ner.model'))
        if parser:
            self.parser.load(os.path.join(MODEL_PATH, 'parser.model'))

    def seg(self, sent):
        return list(self.segmentor.segment(sent))

    def pos(self, words):
        return list(self.postagger.postag(words))

    def recognize(self, words, postags):
        return list(self.recognizer.recognize(words, postags))

    def parse(self, words, postags):
        return list(self.parser.parse(words, postags))


def test(seg):
    pos = ltp.pos(seg)
    arcs = ltp.parse(seg, pos)

    inf = ["%-5s\t%s\t%d:%s" % (a, b, arc.head, arc.relation) for a, b, arc in
           zip(seg, pos, arcs)]
    print('\n'.join(["%d\t%s" % (i + 1, x) for i, x in enumerate(inf)]))


if __name__ == '__main__':
    ltp = LTP()
    print(ltp.seg('附 则'))
    test(['负有', '安全', '生产', '监督', '管理', '职责', '的', '部门', '对', '涉及', '安全', '生产', '的', '事项', '进行', '审查', '、', '验收'])
    print()
    test(['负有安全生产监督管理职责的部门', '对', '涉及', '安全', '生产', '的', '事项', '进行', '审查', '、', '验收'])

    # while True:
    #     s = input()
    #     seg = ltp.seg(s)
    #     pos = ltp.pos(seg)
    #     arcs = ltp.parse(seg, pos)
    #
    #     inf = ["%-5s\t%s\t%d:%s" % (a, b, arc.head, arc.relation) for a, b, arc in
    #            zip(seg, pos, arcs)]
    #     print('\n'.join(["%d\t%s" % (i + 1, x) for i, x in enumerate(inf)]))
