#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/6/8
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessed.ltp import LTP
from preprocessed.preprocess import load_data, save_data
from extract_keyword.baseline.textrank import textrank_keyword

count = 0


class Extractor(object):
    def __init__(self):
        self.law_lst = load_data()
        self.filter_rate = []
        self.ltp = LTP(seg=True, pos=True, ner=True, parser=True)

        for i, law in enumerate(self.law_lst):
            print('(%d/%d) %s' % (i + 1, len(self.law_lst), law.name))
            self.extract_law(law)

        self.filter_rate = 1 - sum(self.filter_rate) / len(self.filter_rate)
        print('filter_rate :', self.filter_rate)
        save_data(self.law_lst)

    # 提取
    def extract_law(self, law, textrank=True):
        general_term = set()
        m = re.search('中华人民共和国(.+)法', law.name)
        if m is not None:
            general_term.add(m.group(1))
        corpus = []
        for cpt in law.chapters:  # 章
            for sc in cpt.sections:  # 节
                for ent in sc.entrys:  # 条
                    # 收集统称词
                    m = re.search(r'统称(.*?)(，|。|（|）|；|……|！|？)', ent.content)
                    if m is not None and len(self.ltp.seg(m.group(1))) > 1:
                        general_term.add(m.group(1))
                    # 分句
                    sents = [x for x in re.split('，|。|：|；|……|！|？', ent.content) if x]
                    # 分词
                    for s in sents:
                        seg = self._segment(s, general_term)
                        pos = self.ltp.pos(seg)
                        seg = [w for i, w in enumerate(seg) if
                               pos[i][0] in ['n', 'v', 'a'] and w not in self.ltp.stopwords]
                        ent.segment += seg

                    corpus.append(' '.join(ent.segment))

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vec = tfidf_vectorizer.fit_transform(corpus).toarray()
        vocab = {v: k for k, v in tfidf_vectorizer.vocabulary_.items()}  # index -> word
        entry_idx = 0
        for cpt in law.chapters:  # 章
            for sc in cpt.sections:  # 节
                for ent in sc.entrys:  # 条
                    # 提取候选词
                    candidate = set()
                    words = set()
                    for sent in [x for x in re.split('，|。|：|；|……|！|？', ent.content) if x]:
                        seg = self._segment(sent, general_term)
                        words.update(seg)
                        candidate.update(self._extract_candidate_words(seg))
                    self.filter_rate.append(len(candidate) / len(words))

                    # 候选词排序排序
                    #   tf-idf 方法
                    tfidf_weight = {vocab[i]: w for i, w in enumerate(tfidf_vec[entry_idx]) if w != 0}
                    tfidf_weight_sum = sum(tfidf_weight.values())  # 归一化
                    tfidf_weight = {word: weight / tfidf_weight_sum for word, weight in tfidf_weight.items()}
                    sorted_word = [i[0] for i in sorted(tfidf_weight.items(), key=lambda d: d[1], reverse=True)]
                    ent.tfidf = sorted_word[:4]
                    #   textrank 方法
                    textrank_weight = textrank_keyword([w for w in ent.segment if w in tfidf_weight.keys()])
                    ent.textrank = [i[0] for i in sorted(textrank_weight.items(), key=lambda d: d[1], reverse=True)][:4]
                    #   综合方法
                    if textrank:
                        weight = {w: 0.7 * tfidf_weight[w] + 0.3 * textrank_weight[w] for w in textrank_weight.keys()}
                        sorted_word = [i[0] for i in sorted(weight.items(), key=lambda d: d[1], reverse=True)]

                    ent.candidate = [word for word in sorted_word if word in candidate]

                    # 将候选词的前几个合并为语块
                    ent.keywords = self.comb_keyword(ent.content, ent.candidate, 4)

                    # 提取分点条目的关键词
                    if len(ent.lst) != 0:
                        # 以‘的’为结尾的并列短句
                        m = [re.match('([^，。；！？]+的)[，；。]', s) for s in ent.lst]
                        if None in m:
                            ent.keywords.append([])
                            seg_lst = [self._segment(point, general_term) for point in ent.lst]
                            lst_corpus = [' '.join(x) for x in seg_lst]
                            lst_tfidf_vectorizer = TfidfVectorizer()
                            lst_tfidf_vec = lst_tfidf_vectorizer.fit_transform(lst_corpus).toarray()
                            lst_vocab = {v: k for k, v in lst_tfidf_vectorizer.vocabulary_.items()}  # index -> word
                            for i, point in enumerate(ent.lst):
                                seg = seg_lst[i]
                                pos = self.ltp.pos(seg)  # 词性过滤
                                candidate = set(w for idx, w in enumerate(seg) if pos[idx][0] in ['n', 'v', 'a'])
                                vec = -np.array(lst_tfidf_vec[i])
                                sorted_word = [lst_vocab[i] for i in vec.argsort()]

                                candidate = [w for w in sorted_word if w in candidate and w not in self.ltp.stopwords]
                                ent.keywords[-1].append(' '.join(self.comb_keyword(point, candidate, 3)))
                        else:
                            ent.keywords.append([x.group(1) for x in m])

                    entry_idx += 1

    def comb_keyword(self, content, candidate, threshold):
        """
        合并候选词
        :param content: 原文
        :param candidate: 候选词
        :param threshold: 词数阈值
        :return: 数量小于阈值的关键词集合
        """
        threshold = min(len(candidate), threshold)
        keywords = candidate[:threshold]
        while True:
            if not self._comb(content, keywords, candidate, threshold):
                break
            else:
                threshold += 1

        # 将关键词按照文中出现顺序排序
        # keywords.sort(key=lambda w: content.index(w))
        try:
            keywords.sort(key=lambda w: content.index(w))
        except:
            print(keywords, content)
        return keywords

    def _comb(self, content, keywords, candidate, threshold):
        def get_seg_idx(p, segment):
            idx = 0
            while p > 0:
                p -= len(segment[idx])
                idx += 1
            if p < 0:
                idx -= 1
            return idx

        for i in range(len(keywords)):
            for j in range(len(keywords)):
                if i == j:
                    continue
                w1, w2 = keywords[i], keywords[j]
                for sent in [x for x in re.split('，|。|：|；|……|！|？', content) if x]:
                    if w1 + w2 not in sent:
                        continue
                    seg = self.ltp.seg(sent)
                    pos = self.ltp.pos(seg)
                    arcs = self.ltp.parse(seg, pos)

                    p = sent.index(w1 + w2)
                    begin1, end1 = p, p + len(w1) - 1
                    begin2, end2 = p + len(w1), p + len(w1) + len(w2) - 1
                    w1_idx = set(range(get_seg_idx(begin1, seg), get_seg_idx(end1, seg) + 1))
                    w2_idx = set(range(get_seg_idx(begin2, seg), get_seg_idx(end2, seg) + 1))
                    w1_head, w2_head = set(arcs[i].head - 1 for i in w1_idx), set(arcs[i].head - 1 for i in w2_idx)

                    if len(w1_head & w2_idx) != 0 or len(w2_head & w1_idx) != 0:
                        keywords.remove(w1)
                        keywords.remove(w2)
                        keywords.append(w1 + w2)
                        if threshold < len(candidate):
                            keywords.append(candidate[threshold])
                        return True
        return False

    def _extract_candidate_words(self, seg):
        """
        使用词性和依存句法分析提取出候选词
        过滤比例0.535
        """
        # 词性标注
        pos = self.ltp.pos(seg)
        # 依存句法分析
        arcs = self.ltp.parse(seg, pos)

        # 核心谓语
        HEDs = set(idx for idx, arc in enumerate(arcs) if arc.relation == 'HED')
        HEDs.update(idx for idx, arc in enumerate(arcs) if arc.relation == 'COO' and arc.head - 1 in HEDs)

        # 主语
        SBJs = set(idx for idx, arc in enumerate(arcs) if arc.relation == 'SBV' and arc.head - 1 in HEDs)
        SBJs.update(idx for idx, arc in enumerate(arcs) if arc.relation == 'COO' and arc.head - 1 in SBJs)

        # 宾语
        OBJs = set(idx for idx, arc in enumerate(arcs) if arc.relation == 'VOB' and arc.head - 1 in HEDs)
        OBJs.update(idx for idx, arc in enumerate(arcs) if arc.relation == 'COO' and arc.head - 1 in OBJs)
        old_len = 0
        while old_len != len(OBJs):
            old_len = len(OBJs)
            OBJs.update(idx for idx, arc in enumerate(arcs) if arc.relation == 'VOB' and arc.head - 1 in OBJs)
            OBJs.update(idx for idx, arc in enumerate(arcs) if arc.relation == 'COO' and arc.head - 1 in OBJs)
        obj_tag = ['IOB', 'FOB', 'POB']
        OBJs.update(idx for idx, arc in enumerate(arcs) if arc.relation in obj_tag and arc.head - 1 in HEDs)
        OBJs.update(idx for idx, arc in enumerate(arcs) if arc.relation == 'COO' and arc.head - 1 in OBJs)

        # 定语
        candidate = HEDs | SBJs | OBJs
        ATTs = set(idx for idx, arc in enumerate(arcs) if arc.relation == 'ATT' and arc.head - 1 in candidate)

        # 指向特定词的介宾关键词
        kw_idx = [seg.index(kw) for kw in ['对', '由', '为', '按照'] if kw in seg]
        POBs = set(idx for idx, arc in enumerate(arcs) if arc.relation == 'POB' and arc.head - 1 in kw_idx)
        POBs.update(idx for idx, arc in enumerate(arcs) if arc.relation == 'COO' and arc.head - 1 in POBs)

        return set(seg[i] for i in candidate | ATTs | POBs if pos[i][0] in ['n', 'v', 'a'])  # 词性过滤

    def _segment(self, sent, general_term):
        """
        考虑统称词的分词分词
        :param sent: 原句子
        :param general_term: 统称词集合
        :return: 将统称词分为一个词的分词结果
        """
        seg = self.ltp.seg(sent)
        for term in general_term:  # 统称词分为一个词
            if term not in sent:
                continue
            term_sent_idx, begin = sent.index(term), 0
            while term_sent_idx > 0:
                term_sent_idx -= len(seg[begin])
                begin += 1
            if term_sent_idx < 0:  # 分词错误
                bad_seg = seg[begin - 1]
                seg = seg[:begin - 1] + [bad_seg[:term_sent_idx], bad_seg[term_sent_idx:]] + seg[begin:]
            end, tmp = begin, term
            while tmp:
                tmp = tmp[len(seg[end]):]
                end += 1
            seg = seg[:begin] + [''.join(seg[begin:end])] + seg[end:]  # 重组
            # if seg[begin] != term and begin + 1 != end:
            #     print(term, begin, end)
            #     print(self.ltp.seg(sent))
            #     print(seg)
        return seg


if __name__ == '__main__':
    Extractor()
