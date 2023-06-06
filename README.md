# ChatBotX

本项目是基于50w小黄鸡对话语料构建的Transformer生成式单轮对话模型。本项目受启发于另外一个使用seq2seq模型构建的单轮对话模型。（项目地址：https://github.com/Schellings/Seq2SeqModel）

目前模型的效果相对一般（具体效果如下图所示），待进一步完善...

<img src="https://zhgyqc.oss-cn-hangzhou.aliyuncs.com/image-20230606190909976.png" alt="模型效果展示" style="zoom:67%;" />

## 项目结构

```shell
│  .gitignore
│  chat.py
│  config.py
│  data_processing.py
│  LICENSE
│  README.md
│  requirements.txt
│  train.py
│  train_helper.ipynb
│  transformer.py
│  utils.py
│
├─data
│      vocab.pkl
│      xhj_data.tsv
│
├─saved_models
```

## 环境配置

环境配置只需要两个步骤：

- 下载项目文件
- 安装相关的第三方库

具体命令如下：

```
git@github.com:Duguce/ChatBotX.git && cd ChatBotX
pip install requirements.txt
```

## 模型使用

- 生成词表

```
python data_processing.py
```

- 模型训练

```
python train.py
```

在进行模型训练之前，需要调整一下`config.py`中的参数设置。

- 模型预测

训练好的模型会保存在`saved_models`目录下。

- 体验交互式聊天

```
python chat.py
```

## 开源协议

本仓库下的作品若无特殊说明均采用 [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) 开源协议进行许可
