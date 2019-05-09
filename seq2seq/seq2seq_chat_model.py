#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 pearxiang@outlook.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: seq2seq_chat_model.py
Author: pearxiang(pearxiang@outlook.com)
Date: 2019/05/09 22:18:31
"""

import pickle

class SimpleSeq2SeqChatModel(object):
    """
    一个基于LSTM-ENCODER/DECODER的简单seq2seq聊天对话模型
    """
    def __init__(self, 
            vocabulary_file = "./model/vocabulary", 
            weights_file = "./model/my_model_weights20.f5"):
        self.vocabulary = pickle.load(open(vocabulary_file, "rb"))
        self.index_to_word = []
        self.word_to_index = {}
        self.bos = -1
        self.eos = -1
        index = 0
        for item in self.vocabulary:
            self.index_to_word.append(item[0])
            self.word_to_index[item[0]] = item[1]
            if item[0] == "BOS": #开始符号单词
                self.bos = index
            elif item[0] == "EOS": #结束符号单词
                self.eos = index
            index += 1
        assert (self.bos >= 0 and self.eos >= 0)

    def train(self):
        pass

    def predict(self, input):
        pass
    


