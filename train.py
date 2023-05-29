# -*- coding: utf-8  -*-
# @Author  : Duguce 
# @Email   : zhgyqc@163.com
# @Time    : 2023/5/23 16:56
# @File    : train.py
# @Software: PyCharm
import time, os
import pickle
import tensorflow as tf
from transformer import Transformer
from data_processing import load_data, DataGenerator, tokenize
from utils import create_masks
from tqdm import tqdm
import config

# 加载数据
inp_text, tar_text = load_data(file_path="./data/xhj_data.tsv", num_samples=config.num_samples)
inp_seq, tar_seq, _, _, _ = tokenize(inp_text, tar_text)

train_dataset = DataGenerator(tokenizer_data=(inp_seq, tar_seq), batch_size=config.batch_size)

# 定义模型
transformer = Transformer(**config.model_params)

# 定义优化器和损失函数
optimizer = tf.keras.optimizers.Adam(**config.optimizer_params)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 定义评价指标
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# 模型保存
checkpoint_path = config.BASE_MODEL_DIR
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")
checkpoint = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))  # 恢复最新的检查点


def loss_function(real, pred):
    """
    损失函数
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# 定义训练步骤
@tf.function
def train_step(inp, tar):
    """
    训练步骤
    """
    tar_inp = tar[:, :-1]  # decoder输入
    tar_real = tar[:, 1:]  # decoder输出

    # 创建mask
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    # 计算梯度
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True,
                                     enc_padding_mask,
                                     combined_mask, dec_padding_mask)
        loss = loss_function(tar_real, predictions)  # 计算损失

    gradients = tape.gradient(loss, transformer.trainable_variables)  # 计算梯度
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))  # 更新参数
    train_loss(loss)  # 计算平均损失
    train_accuracy(tar_real, predictions)  # 计算平均准确率


EPOCHS = config.n_epoch

for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()

    progress = tqdm(
        train_dataset,
        total=len(train_dataset),
        desc=f'Epoch {epoch + 1}/{EPOCHS}',
        unit_scale=True
    )

    for (batch, (inp, tar)) in enumerate(progress):
        train_step(inp, tar)

        if batch % 100 == 0:
            progress.set_postfix({'Loss': train_loss.result().numpy(),
                                  'Accuracy': train_accuracy.result().numpy()})

    progress.close()

    progress.write('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch + 1, train_loss.result(), train_accuracy.result()))
    progress.write(f'Time taken for 1 epoch: {time.time() - start} secs\n')
    checkpoint.save(file_prefix=checkpoint_prefix)
