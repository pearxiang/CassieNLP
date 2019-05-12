#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 pearxiang@outlook.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: data_process.py
Author: pearxiang(pearxiang@outlook.com)
Date: 2019/05/10
一些数据处理函数
"""

import jieba

def tokenizer(text):
	"""
	返回切词列表
	"""
	result = jieba.lcut(text)
	words = []
	for word in result:
		if word != " ":
			words.append(word)
	return words

def from_to_text_process(from_text, to_text):

	from_text_words = [tokenizer(sentence) for sentence in from_text]
	to_text_words = [tokenizer(sentence) for sentence in to_text]
	all_words = []
	for words in from_text_words:
		all_words.extend(words)
	for words in to_text_words:
		all_words.extend(words)
	return from_text_words, to_text_words, all_words


