#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/8/30

评价 使用 Micro F1
"""
import json

import config
from preprocessed.parse_law_file import load_data

corpus = {}
with open(config.corpus_path, 'r', encoding='utf-8')as f:
    for sample in json.load(f)[:config.corpus_num]:
        corpus[sample['content']] = sample['keywords'].split()

law_lst = load_data()
F = [[], [], []]
for law in law_lst:
    for cpt in law.chapters:
        for sc in cpt.sections:
            for ent in sc.entrys:
                if ent.content in corpus and ent.keywords and ent.tfidf and ent.textrank:
                    kw = corpus[ent.content]
                    n1 = sum([1 for w in ent.keywords if w in kw])
                    n2 = sum([1 for w in ent.tfidf if w in kw])
                    n3 = sum([1 for w in ent.textrank if w in kw])
                    p1, p2, p3 = n1 / len(ent.keywords), n2 / len(ent.tfidf), n3 / len(ent.textrank)
                    r1, r2, r3 = n1 / len(kw), n2 / len(kw), n3 / len(kw)
                    f1 = 2 * p1 * r1 / (p1 + r1) if p1 + r1 != 0 else 0
                    f2 = 2 * p2 * r2 / (p2 + r2) if p2 + r2 != 0 else 0
                    f3 = 2 * p3 * r3 / (p3 + r3) if p3 + r3 != 0 else 0
                    F[0].append(f1)
                    F[1].append(f2)
                    F[2].append(f3)

print('Micro F1 of keywords: %f' % (sum(F[0]) / len(F[0])))
print('Micro F1 of tfidf: %f' % (sum(F[1]) / len(F[1])))
print('Micro F1 of textrank: %f' % (sum(F[2]) / len(F[2])))
