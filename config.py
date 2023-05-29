# -*- coding: utf-8  -*-
# @Author  : Duguce 
# @Email   : zhgyqc@163.com
# @Time    : 2023/5/23 20:39
# @File    : config.py
# @Software: PyCharm
from utils import CustomizedSchedule

# 模型保存目录
BASE_MODEL_DIR = "./saved_models"

# 数据集参数
num_samples = None  # 读取的样本数，None表示全部读取
vocab_path = "./data/vocab.pkl"  # 词汇表路径
vocab_size = 6346  # 词汇表大小

n_epoch = 5  # 训练轮数
batch_size = 128  # batch样本数
d_model = 128  # 词嵌入的维度

# 模型参数
model_params = {
    "num_layers": 4,  # 编码器和解码器的层数
    "d_model": d_model,  # 词嵌入的维度
    "num_heads": 8,  # 多头注意力的头数
    "dff": 512,  # 前馈神经网络的中间层维度
    "input_vocab_size": vocab_size,  # 输入词汇表大小
    "target_vocab_size": vocab_size,  # 输出词汇表大小
    "pe_input": 241,  # 输入序列的最大长度
    "pe_target": 250,  # 输出序列的最大长度
    "rate": 0.1  # dropout的保留比例
}

# 优化器参数
optimizer_params = {
    "learning_rate": CustomizedSchedule(d_model),
    "beta_1": 0.9,
    "beta_2": 0.98,
    "epsilon": 1e-9
}
