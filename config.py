# -*- coding: utf-8  -*-
# @Author  : Duguce 
# @Email   : zhgyqc@163.com
# @Time    : 2023/5/23 20:39
# @File    : config.py
# @Software: PyCharm
from utils import CustomizedSchedule

# 模型保存目录
BASE_MODEL_DIR = "./saved_models"

n_epoch = 20  # 训练轮数
batch_size = 64  # batch样本数
dropout_rate = 0.1  # 训练时dropout的保留比例

num_layers = 4  # 编码器和解码器的层数
d_model = 256  # 词嵌入的维度
dff = 1024  # 前馈神经网络的中间层维度
num_heads = 8  # 多头注意力的头数

# 优化器参数
learning_rate = CustomizedSchedule(d_model)
beta_1 = 0.9
beta_2 = 0.98

# 数据集参数
num_samples = None  # 读取的样本数，None表示全部读取
vocab_size = 6306 # 词汇表大小
