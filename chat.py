# -*- coding: utf-8  -*-
# @Author  : Duguce 
# @Email   : zhgyqc@163.com
# @Time    : 2023/5/24 17:53
# @File    : chat.py
# @Software: PyCharm
import os
import jieba
import pickle
import tensorflow as tf
from transformer import Transformer
from data_processing import clean_text
from utils import create_masks
import config

vocab_data = pickle.load(open(config.vocab_path, 'rb'))

# 词典大小
word2idx = vocab_data["word2idx"]
idx2word = vocab_data["idx2word"]

# 创建模型
transformer = Transformer(**config.model_params)
optimizer = tf.keras.optimizers.Adam(**config.optimizer_params)

# 模型参数恢复
checkpoint = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

checkpoint.restore(tf.train.latest_checkpoint(config.BASE_MODEL_DIR))  # 恢复最新的检查点


def preprocess_input(user_input, word2idx):
    """
    预处理输入
    """
    proprecessed_input = clean_text(user_input)
    input_vector = [word2idx.get(word, word2idx["<unk>"]) for word in proprecessed_input.split()]
    input_vector = tf.keras.preprocessing.sequence.pad_sequences([input_vector],
                                                                 maxlen=config.model_params["pe_input"] - 2,
                                                                 padding='post')
    input_vector = tf.concat([[word2idx["<start>"]], input_vector[0]], axis=0)
    input_vector = tf.concat([input_vector, [word2idx["<end>"]]], axis=0)
    input_vector = tf.expand_dims(input_vector, axis=0)
    return input_vector


def generate_response(input_vector, transformer, word2idx, idx2word):
    """
    生成回复
    """
    output = tf.expand_dims([word2idx["<start>"]], axis=0)
    for _ in range(config.model_params["pe_target"]):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input_vector, output)
        predictions, _ = transformer(input_vector, output, False,
                                     enc_padding_mask, combined_mask, dec_padding_mask)
        predictions = predictions[:, -1, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if tf.equal(predicted_id, word2idx["<end>"]):
            break
        output = tf.concat([output, [predicted_id]], axis=-1)
    output = tf.squeeze(output, axis=0)
    output = output.numpy()
    # 将数字转换为文字，并去掉标记符号
    output = [idx2word.get(idx, "<unk>") for idx in output if idx > 3]
    return "".join(output)


user_input = input("User: ")
input_vector = preprocess_input(user_input, word2idx)
response = generate_response(input_vector, transformer, word2idx, idx2word)
print("Bot: ", response)
