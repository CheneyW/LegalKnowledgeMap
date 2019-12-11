#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/7/14
"""

PARAM_d = 0.85


def textrank_keyword(seg, sort=False):
    # 建立图
    nodes = set(seg)
    edges = dict(zip(nodes, [set() for _ in range(len(nodes))]))  # word -> words
    windows = zip(seg[0:-4], seg[1:-3], seg[2:-2], seg[3:-1], seg[4:])
    for win in windows:
        for x in win:
            for y in win:
                if x != y:
                    edges[x].add(y)
    # 计算权重
    weight = dict(zip(nodes, [1 for _ in range(len(nodes))]))  # word -> weight
    while True:
        new_weight = dict()
        for word in nodes:
            new_weight[word] = 1 - PARAM_d + PARAM_d * sum(weight[w_in] / len(edges[w_in]) for w_in in edges[word])
        diff = sum(abs(new_weight[word] - weight[word]) for word in nodes)
        if diff < 0.1:
            break
        weight = new_weight
    # 归一化
    weight_sum = sum(weight.values())
    weight = {word: w / weight_sum for word, w in weight.items()}

    if sort:
        return [item[0] for item in sorted(weight.items(), key=lambda d: d[1], reverse=True)]
    else:
        return weight


if __name__ == '__main__':
    seg = ['程序员', '英文', '程序', '开发', '维护', '专业', '人员', '程序员', '分为', '程序', '设计', '人员', '程序', '编码', '人员', '界限', '特别', '中国',
           '软件', '人员', '分为', '程序员', '高级', '程序员', '系统', '分析员', '项目', '经理']
    print(textrank_keyword(seg))
