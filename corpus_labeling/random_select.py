#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/8/24

随机抽取1000条法律条例
"""
import json
import random
from preprocessed.parse_law_file import load_data

corpus = []

law_lst = load_data()

for law in law_lst:
    for cpt in law.chapters:
        for sc in cpt.sections:
            for ent in sc.entrys:
                path = ' '.join(x for x in [law.name, cpt.title, sc.title, ent.title] if x)
                sample = {'path': path, 'content': ent.content, 'keywords': ''}
                corpus.append(sample)

corpus = random.sample(corpus, 1000)
for i, sample in enumerate(corpus):
    sample['number'] = i + 1

with open('corpus.json', 'w', encoding='utf-8') as f:
    json.dump(corpus, f, indent=4, ensure_ascii=False)
