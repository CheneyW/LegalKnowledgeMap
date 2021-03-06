#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/6/7

法律文本预处理

法律的结构：Law -> Chapter -> Section -> Entry
个别法律没有Chapter和Section
"""
import os
import re
import json

import config

FLAG_Chapter = 1
FLAG_Section = 2
FLAG_Entry = 3


def index_all(lst, val):
    return [i for i, v in enumerate(lst) if v == val]


class Law(object):

    def __init__(self, name, lines, mark, load=None):
        if load is not None:
            """json -> class"""
            self.name = load['name']
            self.has_chapter = load['has_chapter']
            self.chapters = [Chapter(None, None, load=x) for x in load['chapters']]
            return

        # 解析文本
        self.name = name
        self.chapters = []

        first_entry = mark.index(FLAG_Entry)
        first_chapter = first_entry - 1
        while first_chapter >= 0:
            if mark[first_chapter] == FLAG_Chapter:
                break
            first_chapter -= 1
        if first_chapter < 0:
            for i in range(first_entry, len(lines)):
                if mark[i] == FLAG_Chapter:
                    first_chapter = i
                    break
        if first_chapter < 0:
            self.has_chapter = False
            lines, mark = lines[first_entry:], mark[first_entry:]
            self.chapters = [Chapter(lines, mark)]
        else:
            self.has_chapter = True
            lines, mark = lines[first_chapter:], mark[first_chapter:]
            chapter_idx = index_all(mark, FLAG_Chapter)
            self.chapters = [Chapter(lines[a:b], mark[a:b]) for a, b in
                             zip(chapter_idx, (chapter_idx + [len(lines)])[1:])]

    def dumps(self):
        """class -> json"""
        this = dict()
        this['name'] = self.name
        this['has_chapter'] = self.has_chapter
        this['chapters'] = [cpt.dumps() for cpt in self.chapters]
        return this


class Chapter(object):
    def __init__(self, lines, mark, load=None):
        if load is not None:
            """json -> class"""
            self.title = load['title']
            self.name = load['name']
            self.has_section = load['has_section']
            self.sections = [Section(None, None, load=x) for x in load['sections']]
            return

        # 解析文本
        self.title = lines[0] if mark[0] == FLAG_Chapter else ''
        self.name = ''
        if mark[0] == FLAG_Chapter:
            self.name = ''.join(re.split('\s+', re.search('第[\u4E00-\u9FA5]+?章(.+)', lines[0]).group(1)))
            # self.name = list(segmentor.segment(self.name))
            lines, mark = lines[1:], mark[1:]

        section_idx = index_all(mark, FLAG_Section)
        self.has_section = False if len(section_idx) == 0 else True

        self.sections = [Section(lines[a:b], mark[a:b]) for a, b in
                         zip(section_idx, (section_idx + [len(lines)])[1:])] if self.has_section else [
            Section(lines, mark)]

    def dumps(self):
        """class -> json"""
        this = dict()
        this['title'] = self.title
        this['name'] = self.name
        this['has_section'] = self.has_section
        this['sections'] = [sc.dumps() for sc in self.sections]
        return this


class Section(object):
    def __init__(self, lines, mark, load=None):
        if load is not None:
            """json -> class"""
            self.title = load['title']
            self.name = load['name']
            self.entrys = [Entry(None, None, load=x) for x in load['entrys']]
            return

        # 解析文本
        self.title = lines[0] if mark[0] == FLAG_Section else ''
        self.name = ''
        if mark[0] == FLAG_Section:
            self.name = ''.join(re.split('\s+', re.search('第[\u4E00-\u9FA5]+?节(.+)', lines[0]).group(1)))
            # self.name = list(segmentor.segment(self.name))
        entry_idx = index_all(mark, FLAG_Entry)
        self.entrys = [Entry(lines[a:b], mark[a:b]) for a, b in zip(entry_idx, (entry_idx + [len(lines)])[1:])]

    def dumps(self):
        """class -> json"""
        this = dict()
        this['title'] = self.title
        this['name'] = self.name
        this['entrys'] = [e.dumps() for e in self.entrys]
        return this


class Entry(object):
    def __init__(self, lines, mark, load=None):
        if load is not None:
            """json -> class"""
            self.title = load['title']
            self.content = load['content']
            self.lines = load['lines']
            self.lst = load['lst']
            self.keywords = load['keywords']
            self.segment = []
            self.candidate = []
            self.tfidf, self.textrank = load['tfidf'], load['textrank']
            return

        ## 解析文本
        # 标题
        self.title = re.search('第[\u4E00-\u9FA5]+?条', lines[0]).group()
        # 文本
        self.content = re.search('第[\u4E00-\u9FA5]+?条(.+)', lines[0]).group(1).strip().replace(' ', '')
        self.lines = '\n'.join(lines)
        # 分点的内容
        self.lst = []
        # 分词结果
        self.segment = []
        # 候选词
        self.candidate = []
        # tf-idf方法 textrank方法 提取的概括词
        self.tfidf, self.textrank = [], []
        for line in lines[1:]:
            m = re.match(r'（\w+）(.+)', line)
            if m is None:
                self.content += line.replace(' ', '')
            else:
                self.lst.append(m.group(1).strip().replace(' ', ''))

        self.keywords = []

    def dumps(self):
        """class -> json"""
        this = dict()
        this['title'] = self.title
        this['content'] = self.content
        this['lines'] = self.lines
        this['lst'] = self.lst
        this['keywords'] = self.keywords
        this['tfidf'], this['textrank'] = self.tfidf, self.textrank
        return this


def preprocessed():
    filepath_lst = os.listdir(config.text_path)
    law_lst = []
    for file in filepath_lst:
        with open(os.path.join(config.text_path, file), 'r', encoding='gbk', errors="ignore")as f:
            lines = [line.strip().replace('\u3000', ' ') for line in f.readlines()]
            lines = [line for line in lines if len(lines) > 0 and not line.startswith('北大法宝')]

            mark = [0] * len(lines)
            for i, line in enumerate(lines):
                if re.match('第[\u4E00-\u9FA5]+?章', line) is not None:
                    mark[i] = FLAG_Chapter
                elif re.match('第[\u4E00-\u9FA5]+?节', line) is not None:
                    mark[i] = FLAG_Section
                elif re.match('第[\u4E00-\u9FA5]+?条', line) is not None:
                    mark[i] = FLAG_Entry

            name = os.path.splitext(file)[0]
            law = Law(name, lines, mark)
            law_lst.append(law)
    print(len(law_lst), 'files')
    return law_lst


def save_data(law_lst, path=config.save_path):
    for law in law_lst:
        with open(os.path.join(config.data_path, 'law_preprocessed', law.name + '.json'), 'w',
                  encoding='utf-8')as json_f:
            json.dump(law.dumps(), json_f, indent=4, ensure_ascii=False)

    with open(path, 'w', encoding='utf-8')as f:
        for law in law_lst:
            json.dump(law.dumps(), f, ensure_ascii=False)
            f.write('\n')


def load_data(path=config.save_path):
    with open(path, 'r', encoding='utf-8')as f:
        lines = f.readlines()
    trees = [Law(None, None, None, load=json.loads(x)) for x in lines]
    return trees


if __name__ == '__main__':
    laws = preprocessed()
    save_data(laws)
