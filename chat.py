# -*- coding: utf-8  -*-
# @Author  : Duguce 
# @Email   : zhgyqc@163.com
# @Time    : 2023/5/24 17:53
# @File    : chat.py
# @Software: PyCharm
import os
import jieba
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from transformer import Transformer
from data_processing import clean_text
import config

vocab_data = pickle.load(open(config.vocab_path, 'rb'))

# 词典大小
word2idx = vocab_data["word2idx"]
idx2word = vocab_data["idx2word"]

# 创建模型
transformer = Transformer()
optimizer = tf.keras.optimizers.Adam()

checkpoint_path = os.path.join(config.BASE_MODEL_DIR, "ckpt")
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt.restore(checkpoint_path)

