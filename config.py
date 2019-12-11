#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/9/2
"""

import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

data_path = os.path.join(pwd_path, 'data')

ltp_model_path = os.path.join(data_path, 'ltp_model')

text_path = os.path.join(data_path, 'law_text')  # 初始法律文本
preprocessed_path = os.path.join(data_path, 'law_preprocessed')  # 预处理的法律文本

save_path = os.path.join(data_path, 'data.json')  # 数据保存的路径

corpus_path = os.path.join(pwd_path, 'corpus_labeling', 'corpus_.json')  # 人工标注文本
corpus_num = 250  # 已标注数量
