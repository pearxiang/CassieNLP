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
from keras.layers import Embedding, Input, LSTM, Dense, concatenate
from keras.models import Model
from keras.preprocessing import sequence
from keras.optimizers import Adam
import pickle
import os
import nltk
import numpy as np
import itertools
import sys

from data_process import from_to_text_process
from data_process import tokenizer

class SimpleSeq2SeqChatModel(object):
    """
    一个基于LSTM-ENCODER/DECODER的简单seq2seq聊天对话模型
    """
    def __init__(self,
            maxlen_input = 50,
            vocabulary_file = "./model/vocabulary",
            dictionary_size = 17000,
            weights_file = "./model/my_model_weights20.f5"):
        self.dictionary_size = dictionary_size
        self.maxlen_input = maxlen_input
        self.vocabulary_file = vocabulary_file
        self.index_to_word = []
        self.word_to_index = {}
        self.bos = -1
        self.eos = -1
        if os.path.isfile(vocabulary_file):
            vocabulary = pickle.load(open(vocabulary_file, "rb"))
            self.build_word_index(vocabulary)
        self.weights_file = weights_file

    def build_word_index(self, vocabulary):
        """
        根据词汇列表建立word-2-id, id-to-word词典
        """
        index = 0
        for item in vocabulary:
            self.index_to_word.append(item[0])
            self.word_to_index[item[0]] = index
            if item[0] == "BOS": #开始符号单词
                self.bos = index
            elif item[0] == "EOS": #结束符号单词
                self.eos = index
            index += 1
        assert (self.bos >= 0 and self.eos >= 0)

    def build_model(self):
        """
        模型层构建
        """
        sentence_embedding_size = 300 #lstm后句子的维度
        word_embedding_size = 100 #词向量维度

        #对话中的from作为输入, 最多含有maxlen_input个单词的序列, 每个整数是其在dictionary_size中的序列号
        input_from = Input(shape=(self.maxlen_input,), dtype='int32', name='input_from')

        #对话中的to作为输入
        input_to = Input(shape=(self.maxlen_input,), dtype='int32', name='input_to')

        """
        input_from/to 共享一层word_embedding
        input_dim: Size of the vocabulary, i.e. maximum integer index + 1.
        output_dim: Dimension of the dense embedding.
        input_length: Length of input sequences
        这里embedding可以用pre-train的词向量,防止过拟合
        腾讯中文开源词向量: https://ai.tencent.com/ailab/nlp/embedding.html
        英文: https://nlp.stanford.edu/data/glove.6B.zip
        """
        Shared_Embedding = Embedding(input_dim=self.dictionary_size, 
            output_dim=word_embedding_size, input_length=self.maxlen_input)

        word_embedding_from = Shared_Embedding(input_from)
        word_embedding_to = Shared_Embedding(input_to)

        lstm_from_encoder = LSTM(sentence_embedding_size, activation='relu', dropout=0.2)(word_embedding_from)
        lstm_to_decoder = LSTM(sentence_embedding_size, activation='relu', dropout=0.2)(word_embedding_to)

        #连接
        merge_from_to = concatenate([lstm_from_encoder, lstm_to_decoder], axis=1)
        out = Dense(int(self.dictionary_size/2), activation="relu")(merge_from_to)
        #out = Dense(dictionary_size, activation="softmax")(out)
        out = Dense(self.dictionary_size, activation='softmax')(out)
        self.model = Model(input=[input_from, input_to], output=[out])
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005))
        self.model.summary()
        if os.path.isfile(self.weights_file): #如果有已经构建好的模型参数就加载
            model.load_weights(self.weights_file)

    def greedy_decoder(self, input):
        """
        贪心解码,每次取最大概率的单词，容易导致答案雷同
        """
        prob = 1
        ans_partial = np.zeros((1, self.maxlen_input))
        ans_partial[0: -1] = self.bos #末尾强制设置为BOS开始, 开始循环预测下一个单词
        for k in range(self.maxlen_input-1):
            ye = self.model.predict([input, ans_partial])
            yel = ye[0,:]
            p = np.max(yel) #选中单词的概率
            mp = np.argmax(ye) #选中的单词的
            ans_partial[0, 0:-1] = ans_partial[0, 1:] #预测的word全部向前挪动一步，末尾新预测的单词
            ans_partial[0, -1] = mp
            prob = prob + p #连续相乘概率
            #if mp == self.eos: #结束符号停止
            #    break
        text = ''
        for k in ans_partial[0]:
            k = k.astype(int)
            if k < self.dictionary_size:
                w = self.vocabulary[k]
                text += w[0] + ' '
        return (text, prob)

    def train(self, from_text, to_text):
        """
        根据聊天回复文本来训练样本, 这里用Teacher Forcing来构建训练样本训练

        """
        assert len(from_text) == len(to_text)
        print(len(from_text))
        print (len(to_text))
        from_text_words, to_text_words, all_words = from_to_text_process(from_text, to_text)
        for index in range(len(from_text_words)):
            from_text_words[index].insert(0, "BOS")
            from_text_words[index].append("EOS")
            to_text_words[index].insert(0, "BOS")
            to_text_words[index].append("EOS")

        word_freq = nltk.FreqDist(itertools.chain(all_words))
        self.vocabulary = word_freq.most_common(self.dictionary_size-3)
        self.vocabulary.append(("BOS", 1))
        self.vocabulary.append(("EOS", 1))
        unk = "UNK"
        self.vocabulary.append(("UNK", 1))
        with open(self.vocabulary_file, "wb") as f:
            pickle.dump(self.vocabulary, f)
        self.build_word_index(self.vocabulary)

        for i, sent in enumerate(from_text_words):
            from_text_words[i] = [w if w in self.word_to_index else unk for w in sent]
        for i, sent in enumerate(to_text_words):
            to_text_words[i] = [w if w in self.word_to_index else unk for w in sent]

        X = np.asarray([[self.word_to_index[w] for w in sent] for sent in from_text_words])
        Y = np.asarray([[self.word_to_index[w] for w in sent] for sent in to_text_words])

        X = sequence.pad_sequences(X, maxlen=self.maxlen_input)
        Y = sequence.pad_sequences(Y, maxlen=self.maxlen_input, padding='post')

        ###构建好FROM/TO文本的one-hot数据了###

        epochs = 5 #训练轮数
        n_exem, n_words = X.shape
        n_test = 800
        num_subsets = 10
        step = int(np.around((n_exem - n_test) / num_subsets)) #步长
        round_exem = int(step * num_subsets) #每一轮训练的总数

        train_x = X[n_test+1:]
        train_y = Y[n_test+1:]

        for poch in range(epochs):
            for n in range(0, round_exem, step):
                batch_x = train_x[n:n+step]
                batch_y = train_y[n:n+step]
                count = 0
                for i, sent in enumerate(batch_y):
                    l = np.where(sent == self.eos)
                    index = l[0][0] #找到训练集EOS的位置
                    count += index+1  #回答句子的单词长度为index+1, count统计本轮所有的长度和

                Q = np.zeros((count, self.maxlen_input))
                A = np.zeros((count, self.maxlen_input))
                Y = np.zeros((count, self.dictionary_size))
                print (count)

                #将A拆成一个单词一个单词的增长
                count = 0
                for i,sent in enumerate(batch_y):
                    ans_partial = np.zeros((1, self.maxlen_input))
                    l = np.where(sent == self.eos) #找到ans的结束符
                    index = l[0][0]
                    for k in range(1, index+1): #index+1是要把EOS也训练进来，需要训练到出现eos为止，代表生成结束
                        y = np.zeros((1, self.dictionary_size))
                        y[0, sent[k]] = 1 #这次训练结果的结果选中单词sent[k]
                        ans_partial[0, -k:] = sent[0:k] #把到k为止的出现的答案单词放到ans的末尾
                        Q[count, :] = batch_x[i:i+1] #from文本是完整的一句话单词内容
                        A[count, :] = ans_partial #to文本是到k位置出现前的所有单词
                        Y[count, :] = y #Y是表示第k个单词出现sent[k]都得概率最大
                        count += 1
                        print (batch_x[i:i+1])
                        print (ans_partial)
                        print (y)
                        print (self.eos, self.bos)
                print('Training epoch: %d, training examples: %d - %d' % (poch, n, n + step))
                self.model.fit([Q,A], Y, batch_size=128, epochs=1)
                test_input = train_x[41:42]
                print(self.greedy_decoder(test_input))
            model.save_weights(self.weights_file + str(poch), overwrite=True)

    def predict(self, input):
        """
        TODO
        """
        words = tokenizer(input)
        words.insert(0, "BOS")
        words.append(0, "EOS")
        unk = "UNK"
        words = [word if word in self.word_to_index else unk for word in words]
        X = np.asarray([self.word_to_index[w] for w in text])
        length = X.shape
        Q = np.zeros((1, self.maxlen_input))
        if length < (self.maxlen_input):
            Q[0, -length:] = X
        else:
            Q[0, :] = X[-self.maxlen_input] #超长了截断
        return self.greedy_decoder(Q[0:1])
    
def main():
    """
    测试执行下
    """
    simple_model = SimpleSeq2SeqChatModel()
    simple_model.build_model()

    from_text = []
    to_text = []
    for line in open(sys.argv[1]):
        attr = line.strip().split("\t")
        if len(attr) != 2:
            continue
        from_text.append(attr[0])
        to_text.append(attr[1])
    simple_model.train(from_text, to_text)

if __name__ == '__main__':
    main()

