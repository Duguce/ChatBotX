# -*- coding: utf-8  -*-
# @Author  : Duguce 
# @Email   : zhgyqc@163.com
# @Time    : 2023/5/19 19:01
# @File    : seq2seq_model.py.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
from tensorflow import keras


class PositionalEncoding(keras.layers.Layer):
    """
    位置编码
    """

    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def build(self, input_shape):
        seq_len = input_shape[1]  # 获取序列长度
        d_model = input_shape[2]  # 获取词向量维度
        self.positional_encoding = self.calculate_positional_encoding(seq_len, d_model)

    def calculate_positional_encoding(self, seq_len, d_model):
        positional_encoding = np.zeros((seq_len, d_model))
        angles = np.arange(seq_len)[:, np.newaxis] / np.power(10000, np.arange(d_model)[np.newaxis, :] / d_model)
        positional_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        positional_encoding[:, 1::2] = np.cos(angles[:, 1::2])

        return tf.cast(positional_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]


def create_padding_mask(seq):
    """
    创建padding mask
    :param seq: 输入序列
    :return: mask
    """
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(seq):
    """
    创建look ahead mask
    :param seq: 输入序列
    :return: mask
    """
    seq_len = tf.shape(seq)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    return look_ahead_mask


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    计算attention权重
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # 计算q和k的点积
    dk = tf.cast(tf.shape(k)[-1], tf.float32)  # 获取k的维度
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # 缩放点积
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # 将mask加到缩放的点积上
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1  # 计算attention权重
    )
    output = tf.matmul(attention_weights, v)  # 计算输出

    return output, attention_weights


class MultiHeadAttention(keras.layers.Layer):
    """
    多头注意力机制
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads  # 头数
        self.d_model = d_model  # 词向量维度
        assert d_model % self.num_heads == 0  # 确保可以整除
        self.depth = d_model // self.num_heads  # 每个头的维度
        self.wq = keras.layers.Dense(d_model)  # q的全连接层
        self.wk = keras.layers.Dense(d_model)  # k的全连接层
        self.wv = keras.layers.Dense(d_model)  # v的全连接层
        self.dense = keras.layers.Dense(d_model)  # 输出的全连接层

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  # 将x分割成多个头
        return tf.transpose(x, perm=[0, 2, 1, 3])  # 转置

    def call(self, k, v, q, mask=None):
        batch_size = tf.shape(q)[0]  # 获取batch_size
        q = self.wq(q)  # q的全连接
        k = self.wk(k)  # k的全连接
        v = self.wv(v)  # v的全连接
        q = self.split_heads(q, batch_size)  # q分割成多个头
        k = self.split_heads(k, batch_size)  # k分割成多个头
        v = self.split_heads(v, batch_size)  # v分割成多个头
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)  # 计算attention权重
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # 转置
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # 合并多个头
        output = self.dense(concat_attention)  # 全连接

        return output, attention_weights


class PointwiseFeedForward(keras.layers.Layer):
    """
    前馈网络
    """

    def __init__(self, d_model, dff):
        super(PointwiseFeedForward, self).__init__()
        self.dense1 = keras.layers.Dense(dff, activation="relu")  # 第一个全连接层
        self.dense2 = keras.layers.Dense(d_model)  # 第二个全连接层

    def call(self, inputs):
        x = self.dense1(inputs)  # 第一个全连接层
        x = self.dense2(x)  # 第二个全连接层

        return x


class EncoderLayer(keras.layers.Layer):
    """
    编码器层
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)  # 多头注意力机制
        self.ffn = PointwiseFeedForward(d_model, dff)  # 前馈网络
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)  # 第一个归一化层
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)  # 第二个归一化层
        self.dropout1 = keras.layers.Dropout(rate)  # 第一个dropout层
        self.dropout2 = keras.layers.Dropout(rate)  # 第二个dropout层

    def call(self, inputs, training, mask=None):
        attn_output, _ = self.multihead_attention(inputs, inputs, inputs, mask)  # 多头注意力机制
        attn_output = self.dropout1(attn_output, training=training)  # dropout
        out1 = self.layernorm1(inputs + attn_output)  # 第一个归一化层
        ffn_output = self.ffn(out1)  # 前馈网络
        ffn_output = self.dropout2(ffn_output, training=training)  # dropout
        out2 = self.layernorm2(out1 + ffn_output)  # 第二个归一化层

        return out2


class Encoder(keras.layers.Layer):
    """
    编码器
    """

    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model  # 词向量维度
        self.num_layers = num_layers  # 编码器层数
        self.maximum_position_encoding = maximum_position_encoding  # 最大位置编码
        self.embedding = keras.layers.Embedding(input_vocab_size, d_model)  # 词嵌入层
        self.pos_encoding = PositionalEncoding()  # 位置编码层
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]  # 编码器层
        self.dropout = keras.layers.Dropout(rate)  # dropout层

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]  # 获取序列长度
        tf.debugging.assert_less_equal(
            seq_len, self.maximum_position_encoding,
            "seq_len should be less than or equal to self.maximum_position_encoding"
        )  # 断言序列长度小于等于最大位置编码
        x = self.embedding(x)  # 词嵌入
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # 乘以根号d_model
        x = self.pos_encoding(x)  # 位置编码
        x = self.dropout(x, training=training)  # dropout
        for i in range(self.num_layers):  # 编码器层
            x = self.enc_layers[i](x, training, mask)

        return x


class DecoderLayer(keras.layers.Layer):
    """
    解码器层
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.multihead_attention1 = MultiHeadAttention(d_model, num_heads)  # 多头注意力机制
        self.multihead_attention2 = MultiHeadAttention(d_model, num_heads)  # 多头注意力机制
        self.ffn = PointwiseFeedForward(d_model, dff)  # 前馈网络
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)  # 第一个归一化层
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)  # 第二个归一化层
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)  # 第三个归一化层
        self.dropout1 = keras.layers.Dropout(rate)  # 第一个dropout层
        self.dropout2 = keras.layers.Dropout(rate)  # 第二个dropout层
        self.dropout3 = keras.layers.Dropout(rate)  # 第三个dropout层

    def call(self, inputs, enc_output, training,
             look_ahead_mask=None, padding_mask=None):
        # 解码器自注意力层
        attn1, attn_weights_block1 = self.multihead_attention1(inputs, inputs, inputs, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)  # dropout
        out1 = self.layernorm1(attn1 + inputs)  # 第一个归一化层
        # 编码器-解码器注意力层
        attn2, attn_weights_block2 = self.multihead_attention2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)  # dropout
        out2 = self.layernorm2(attn2 + out1)  # 第二个归一化层
        # 前馈网络层
        ffn_output = self.ffn(out2)  # 前馈网络
        ffn_output = self.dropout3(ffn_output, training=training)  # dropout
        out3 = self.layernorm3(ffn_output + out2)  # 第三个归一化层

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(keras.layers.Layer):
    """
    解码器
    """

    def __init__(self, num_layers, d_model, num_heads, dff,
                 target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model  # 词向量维度
        self.num_layers = num_layers  # 解码器层数
        self.maximum_position_encoding = maximum_position_encoding  # 最大位置编码
        self.embedding = keras.layers.Embedding(target_vocab_size, d_model)  # 词嵌入层
        self.pos_encoding = PositionalEncoding()  # 位置编码层
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]  # 解码器层
        self.dropout = keras.layers.Dropout(rate)  # dropout层

    def call(self, x, enc_output, training,
             look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]  # 获取序列长度
        tf.debugging.assert_less_equal(
            seq_len, self.maximum_position_encoding,
            "seq_len should be less than or equal to self.maximum_position_encoding"
        )
        attention_weights = {}  # 注意力权重 # 注意力权重
        x = self.embedding(x)  # 词嵌入
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # 乘以根号d_model
        x = self.pos_encoding(x)  # 位置编码
        x = self.dropout(x, training=training)  # dropout
        for i in range(self.num_layers):  # 解码器层
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        return x, attention_weights


class Transformer(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
                 pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads,
                               dff, input_vocab_size, pe_input, rate)  # 编码器
        self.decoder = Decoder(num_layers, d_model, num_heads,
                               dff, target_vocab_size, pe_target, rate)  # 解码器
        self.final_layer = keras.layers.Dense(target_vocab_size)  # 最后一层

    def call(self, en_inputs, de_inputs, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(en_inputs, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(
            de_inputs, enc_output, training, look_ahead_mask, dec_padding_mask
        )
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights


if __name__ == '__main__':
    print("------------------测试 PositionalEncoding()-------------------")
    pe = PositionalEncoding()
    inputs = tf.random.uniform((1, 50, 512))
    outputs = pe(inputs)
    print(outputs.shape)
    print("------------------测试 create_padding_mask()-------------------")
    seq = tf.constant([[1, 2, 0, 0, 3], [0, 4, 5, 0, 0]])
    mask = create_padding_mask(seq)
    print(mask)
    print("------------------测试 create_look_ahead_mask()-------------------")
    seq = tf.constant([[1, 2, 3, 4, 5], [0, 4, 5, 0, 0],
                       [0, 0, 0, 0, 0], [0, 4, 5, 0, 0],
                       [0, 4, 5, 0, 0], [0, 4, 5, 0, 0]])
    mask = create_look_ahead_mask(seq)
    print(mask)
    print("------------------测试 scaled_dot_product_attention()-------------------")
    temp_k = tf.constant([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10],
                          [0, 0, 10]], dtype=tf.float32)  # (4, 3)
    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 6]], dtype=tf.float32)  # (4, 2)
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v)
    print(temp_out)
    print("------------------测试 MultiHeadAttention()-------------------")
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, y, y, mask=None)
    print(out.shape)
    print(attn.shape)
    print("------------------测试 PointwiseFeedForward()-------------------")
    sample_ffn = PointwiseFeedForward(512, 2048)
    print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)
    print("------------------测试 EncoderLayer()-------------------")
    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 40, 512)), False, None)
    print(sample_encoder_layer_output.shape)
    print("------------------测试 Encoder()-------------------")
    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, input_vocab_size=8500,
                             maximum_position_encoding=10000)
    sample_encoder_output = sample_encoder(tf.random.uniform((64, 62)), False, None)
    print(sample_encoder_output.shape)
    print("------------------测试 DecoderLayer()-------------------")
    sample_decoder_layer = DecoderLayer(512, 8, 2048)
    sample_decoder_input = tf.random.uniform((64, 37, 512))
    sample_decoder_layer_output, sample_decoder_attention_weights1, sample_decoder_attention_weights2 = sample_decoder_layer(
        sample_decoder_input, sample_encoder_layer_output, False, None, None)
    print(sample_decoder_layer_output.shape)
    print(sample_decoder_attention_weights1.shape)
    print(sample_decoder_attention_weights2.shape)
    print("------------------测试 Decoder()-------------------")
    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, target_vocab_size=8000,
                             maximum_position_encoding=5000)
    output, attn = sample_decoder(tf.random.uniform((64, 26)),
                                  enc_output=sample_encoder_output,
                                  training=False,
                                  look_ahead_mask=None,
                                  padding_mask=None)
    print(output.shape)
    print(attn['decoder_layer2_block2'].shape)

    print("------------------测试 Transformer()-------------------")
    sample_transformer = Transformer(
        num_layers=4, d_model=128, num_heads=8, dff=512,
        input_vocab_size=6304, target_vocab_size=6304,
        pe_input=241, pe_target=250)
    temp_input = tf.random.uniform((64, 26))
    temp_target = tf.random.uniform((64, 31))
    fn_out, attention_weights = sample_transformer(temp_input, temp_target, training=False,
                                                   enc_padding_mask=None,
                                                   look_ahead_mask=None,
                                                   dec_padding_mask=None)

    print(fn_out.shape)
    for key in attention_weights:
        print(key, attention_weights[key].shape)
    print(sample_transformer.summary())
