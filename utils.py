# -*- coding: utf-8  -*-
# @Author  : Duguce 
# @Email   : zhgyqc@163.com
# @Time    : 2023/5/23 17:03
# @File    : utils.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow import keras
from transformer import create_padding_mask, create_look_ahead_mask
import matplotlib.pyplot as plt


class CustomizedSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    自定义学习率
    """

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomizedSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        根据当前训练步数和模型维度计算学习率
        """
        step = tf.cast(step, tf.float32)  # 将步数转换为浮点数

        arg1 = tf.math.rsqrt(step)  # 计算 rsqrt(step)
        arg2 = step * (self.warmup_steps ** (-1.5))  # 计算 step * (warmup_steps ^ (-1.5))
        arg3 = tf.math.rsqrt(self.d_model)  # 计算 rsqrt(d_model)

        return arg3 * tf.math.minimum(arg1, arg2)  # 返回最终的学习率


def create_masks(inputs, target):
    """
    创建填充遮蔽和解码器遮蔽
    """
    enc_padding_mask = create_padding_mask(inputs)  # 创建编码器填充遮蔽
    dec_padding_mask = create_padding_mask(inputs)  # 创建解码器填充遮蔽
    look_ahead_mask = create_look_ahead_mask(target)  # 创建解码器前瞻遮蔽，用于遮蔽未生成的部分
    dec_target_padding_mask = create_padding_mask(target)  # 创建解码器目标填充遮蔽
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)  # 将解码器填充遮蔽和解码器前瞻遮蔽合并

    return enc_padding_mask, combined_mask, dec_padding_mask


if __name__ == '__main__':
    print("----------------------test-----------")
    learning_rate = CustomizedSchedule(128)
    plt.plot(learning_rate(tf.range(20000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()
