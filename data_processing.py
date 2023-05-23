# -*- coding: utf-8  -*-
# @Author  : Duguce 
# @Email   : zhgyqc@163.com
# @Time    : 2023/5/19 19:00
# @File    : data_processing.py
# @Software: PyCharm
import re
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


def tokenize(inp_text, tar_text):
    """序列号生成词典"""
    inp_res = []
    tar_res = []
    for inp, tar in zip(inp_text, tar_text):
        inp_res.append(" ".join(jieba.cut(inp)))
        tar_res.append(" ".join(jieba.cut(tar)))

    tokenizer = keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(inp_res + tar_res)

    inp_seq = tokenizer.texts_to_sequences(inp_res)
    tar_seq = tokenizer.texts_to_sequences(tar_res)

    max_inp_len = max(len(seq) for seq in inp_seq)
    max_tar_len = max(len(seq) for seq in tar_seq)

    inp_seq = keras.preprocessing.sequence.pad_sequences(inp_seq, maxlen=max_inp_len, padding='post')
    tar_seq = keras.preprocessing.sequence.pad_sequences(tar_seq, maxlen=max_tar_len, padding='post')

    # 添加特殊标记
    tokenizer.word_index['<start>'] = len(tokenizer.word_index) + 1
    tokenizer.index_word[len(tokenizer.word_index)] = '<start>'
    tokenizer.word_index['<end>'] = len(tokenizer.word_index) + 1
    tokenizer.index_word[len(tokenizer.word_index)] = '<end>'
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    tokenizer.word_index['<unk>'] = len(tokenizer.word_index) + 1
    tokenizer.index_word[len(tokenizer.word_index)] = '<unk>'

    # 添加开头和结尾标记
    inp_seq = np.insert(inp_seq, 0, tokenizer.word_index['<start>'], axis=1)
    inp_seq = np.insert(inp_seq, max_inp_len + 1, tokenizer.word_index['<end>'], axis=1)
    tar_seq = np.insert(tar_seq, 0, tokenizer.word_index['<start>'], axis=1)
    tar_seq = np.insert(tar_seq, max_tar_len + 1, tokenizer.word_index['<end>'], axis=1)

    max_inp_len += 2
    max_tar_len += 2

    return inp_seq, tar_seq, tokenizer, max_inp_len, max_tar_len


class DataGenerator(keras.utils.Sequence):
    """
    数据生成器
    """

    def __init__(self, tokenizer_data, batch_size, shuffle=True):
        self.indices = None
        self.inp_seq, self.tar_seq, self.tokenizer = tokenizer_data
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
    inp_text, tar_text = load_data(file_path="./data/xhj_data.tsv", num_samples=None)
    inp_seq, tar_seq, tokenizer, _, _ = tokenize(inp_text, tar_text)
    print(f"test the clean_text func: {clean_text('怎么证明	你问～我爱～你有～多深～我爱～你有～几～分～～～')}")
    # print("-" * 100)
    # print(inp_text, tar_text)
    # print("-" * 100)
    # print(inp_seq, tar_seq)
    print("-" * 100)
    dataset = DataGenerator(tokenizer_data=(inp_seq, tar_seq, tokenizer), batch_size=64)

    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     print(f"Epoch {epoch + 1}/{num_epochs}")
    #     for batch, (inp_seq, tar_seq) in enumerate(dataset):
    #         print(f"Batch {batch + 1}/{len(dataset)}")
    #         print(inp_seq.shape, tar_seq.shape)
    #         print(inp_seq, tar_seq)
    #         print("-" * 100)
    vocab_size = len(tokenizer.word_index) + 1  # 词汇表大小
    print(f"vocab_size: {vocab_size}")