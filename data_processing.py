# -*- coding: utf-8  -*-
# @Author  : Duguce 
# @Email   : zhgyqc@163.com
# @Time    : 2023/5/19 19:00
# @File    : data_processing.py
# @Software: PyCharm
import re
import os
import pickle
import jieba
import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_data(file_path, num_samples=None):
    """加载数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            if num_samples is None:
                samples = file.readlines()
            else:
                samples = file.readlines()[:num_samples]

            word_pairs = [[clean_text(text) for text in line.split("\t")] for line in samples]

            return zip(*word_pairs)
    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
        return []


def clean_text(text):
    """清洗数据"""
    text = text.strip()  # 去除头尾空格
    text = " ".join(text)  # 分词
    text = re.sub(r'\s+', ' ', text)  # 去除多余空格和换行符
    symbols = {
        r'…{1,100}': '…',
        r'\.{3,100}': '…',
        r'···{2,100}': '…',
        r',{1,100}': '，',
        r'\.{1,100}': '。',
        r'。{1,100}': '。',
        r'\?{1,100}': '？',
        r'？{1,100}': '？',
        r'!{1,100}': '！',
        r'！{1,100}': '！',
        r'~{1,100}': '～',
        r'～{1,100}': '～',
        r'[“”]{1,100}': '"',
        r'[^\s\w\u4e00-\u9fff"。，？！～·]+': '',
        r'[ˇˊˋˍεπのゞェーω]': ''
    }  # 去除其他无用符号
    for pattern, repl in symbols.items():
        text = re.sub(pattern, repl, text)

    return text


def build_vocab(text_lst, vocab_file="./data/vocab.pkl"):
    """构建词典"""
    if os.path.exists(vocab_file):
        with open(vocab_file, 'rb') as file:
            vocab_data = pickle.load(file)
            vocab_size = vocab_data['vocab_size']
            word2idx = vocab_data['word2idx']
            idx2word = vocab_data['idx2word']
    else:
        vocab = set()
        for seq in text_lst:
            vocab.update(seq.split())
        vocab = sorted(vocab)
        # 添加特殊标记
        vocab.insert(0, '<pad>')
        vocab.insert(1, '<unk>')
        vocab.insert(2, '<start>')
        vocab.insert(3, '<end>')
        # 词典大小
        vocab_size = len(vocab) + 1
        # 词典索引
        idx2word = dict(enumerate(vocab))
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        # 保存词典
        vocab_data = {
            'vocab_size': vocab_size,
            'word2idx': word2idx,
            'idx2word': idx2word
        }
        with open(vocab_file, 'wb') as file:
            pickle.dump(vocab_data, file)

    return vocab_size, word2idx, idx2word


def tokenize(inp_text, tar_text):
    """序列化生成词典"""
    inp_res = []
    tar_res = []
    for inp, tar in zip(inp_text, tar_text):
        inp_res.append(" ".join(jieba.cut(inp)))
        tar_res.append(" ".join(jieba.cut(tar)))

    vocab_size, word2idx, _ = build_vocab(inp_res + tar_res)
    # 序列化
    inp_seq = [[word2idx.get(word, word2idx['<unk>']) for word in seq.split()] for seq in inp_res]
    tar_seq = [[word2idx.get(word, word2idx['<unk>']) for word in seq.split()] for seq in tar_res]
    # 最大长度
    max_inp_len = max(len(seq) for seq in inp_seq)
    max_tar_len = max(len(seq) for seq in tar_seq)
    # 填充
    inp_seq = keras.preprocessing.sequence.pad_sequences(inp_seq, maxlen=max_inp_len, padding='post')
    tar_seq = keras.preprocessing.sequence.pad_sequences(tar_seq, maxlen=max_tar_len, padding='post')
    # 添加开头和结尾
    inp_seq = np.hstack((np.ones((inp_seq.shape[0], 1)) * word2idx['<start>'], inp_seq))
    tar_seq = np.hstack((np.ones((tar_seq.shape[0], 1)) * word2idx['<start>'], tar_seq))
    inp_seq = np.hstack((inp_seq, np.ones((inp_seq.shape[0], 1)) * word2idx['<end>']))
    tar_seq = np.hstack((tar_seq, np.ones((tar_seq.shape[0], 1)) * word2idx['<end>']))
    # 最大长度加2
    max_inp_len += 2
    max_tar_len += 2

    return inp_seq, tar_seq, vocab_size, max_inp_len, max_tar_len


class DataGenerator(keras.utils.Sequence):
    """
    数据生成器
    """

    def __init__(self, tokenizer_data, batch_size, shuffle=True):
        self.indices = None
        self.inp_seq, self.tar_seq = tokenizer_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.inp_seq) // self.batch_size  # 每个epoch的迭代次数

    def __getitem__(self, index):
        # 每个batch的索引
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_inp_seq = [self.inp_seq[i] for i in batch_indices]
        batch_tar_seq = [self.tar_seq[i] for i in batch_indices]
        # # 每个batch中的最大长度
        # max_inp_len = max(len(seq) for seq in batch_inp_seq)
        # max_tar_len = max(len(seq) for seq in batch_tar_seq)
        # # 填充
        # batch_inp_seq = keras.preprocessing.sequence.pad_sequences(batch_inp_seq, maxlen=max_inp_len, padding='post')
        # batch_tar_seq = keras.preprocessing.sequence.pad_sequences(batch_tar_seq, maxlen=max_tar_len, padding='post')
        # 添加开头和结尾标记
        # batch_inp_seq = np.insert(batch_inp_seq, 0, self.tokenizer.word_index['<start>'], axis=1)
        # batch_inp_seq = np.insert(batch_inp_seq, batch_inp_seq.shape[1], self.tokenizer.word_index['<end>'], axis=1)
        # batch_tar_seq = np.insert(batch_tar_seq, 0, self.tokenizer.word_index['<start>'], axis=1)
        # batch_tar_seq = np.insert(batch_tar_seq, batch_tar_seq.shape[1], self.tokenizer.word_index['<end>'], axis=1)

        # 将输入序列和目标序列转为张量
        batch_inp_seq = tf.convert_to_tensor(batch_inp_seq)
        batch_tar_seq = tf.convert_to_tensor(batch_tar_seq)

        return batch_inp_seq, batch_tar_seq

    def on_epoch_end(self):
        self.indices = np.arange(len(self.inp_seq))
        if self.shuffle:
            np.random.shuffle(self.indices)


if __name__ == '__main__':
    inp_text, tar_text = load_data(file_path="./data/xhj_data.tsv", num_samples=1000)
    inp_seq, tar_seq, vocab_size, _, _ = tokenize(inp_text, tar_text)
    print(f"test the clean_text func: {clean_text('怎么证明	你问～我爱～你有～多深～我爱～你有～几～分～～～')}")
    # print("-" * 100)
    # print(inp_text, tar_text)
    # print("-" * 100)
    # print(inp_seq, tar_seq)
    print("-" * 100)
    dataset = DataGenerator(tokenizer_data=(inp_seq, tar_seq), batch_size=64)

    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch, (inp_seq, tar_seq) in enumerate(dataset):
            print(f"Batch {batch + 1}/{len(dataset)}")
            print(inp_seq.shape, tar_seq.shape)
            print(inp_seq, tar_seq)
            print("-" * 100)

    print(f"vocab_size: {vocab_size}")
