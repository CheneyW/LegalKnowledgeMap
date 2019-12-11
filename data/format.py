#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/6/8
"""
import os
import json

with open('data.json', 'r', encoding='utf-8')as f:
    laws = [json.loads(line.strip()) for line in f.readlines()]

data = {'name': '法律集合', 'children': []}
for l in laws:
    if not l['name'].startswith('中华人民共和国'):
        print(l['name'])
    law = {'name': l['name'], 'children': []}
    data['children'].append(law)

    for c in l['chapters']:
        chapter = {'name': c['title'], 'children': []}
        law['children'].append(chapter)

        for s in c['sections']:
            section = {'name': s['title'], 'children': []}
            chapter['children'].append(section)

            for e in s['entrys']:
                if len(e['keywords']) == 0:
                    kw = ''
                elif isinstance(e['keywords'][-1], list):
                    kw = ' '.join(e['keywords'][:-1])
                else:
                    kw = ' '.join(e['keywords'])
                # content = {'name': e['content']}
                entry = {'name': e['title'], 'children': [{'name': kw, 'lines': e['lines']}]}
                section['children'].append(entry)

        if len(chapter['children']) == 1:
            chapter['children'] = chapter['children'][0]['children']

    if len(law['children']) == 1 and law['children'][0]['name'] == '':
        law['children'] = law['children'][0]['children']

with open(os.path.join(os.pardir, 'data.json'), 'w', encoding='utf-8')as f:
    json.dump(data, f, ensure_ascii=False)
