# CCKS 2021答非所问 白[MASK] 队 训练复现代码



#### 此文件夹为训练文件夹，运行此文件夹中代码，可复现模型。

```
复现所需python库：
pytorch 
transformers
tqdm
sklearn
pandas
numpy
```

想要训练，可运行run.sh或按本readme具体代码文件一次运行。



### 额外开源数据

##### [Chinese-medical-dialogue-data-master](https://github.com/Toyhom/Chinese-medical-dialogue-data)

### 开源预训练模型

- [hfl/chinese-roberta-wwm-ext]([hfl/chinese-roberta-wwm-ext · Hugging Face](https://huggingface.co/hfl/chinese-roberta-wwm-ext))
- [hfl/chinese-macbert-large]([hfl/chinese-macbert-large · Hugging Face](https://huggingface.co/hfl/chinese-macbert-large))
- [luhua/chinese_pretrain_mrc_macbert_large]([luhua/chinese_pretrain_mrc_macbert_large · Hugging Face](https://huggingface.co/luhua/chinese_pretrain_mrc_macbert_large))

## 具体代码文件及介绍

### 数据处理

1. 读取训练集，并转换为csv文件。

2. 读取开源数据，并转换为csv文件

   `python read.py`

   `python read_other.py`

3. 从训练集标注为’回答正确‘的样本中，依据编辑距离，选取其余样本中最相近（但不完全相同）问句的答句作为负样本，实现数据增广，下称（编辑距离扩增样本）。

   `python edit_distance.py`

### 训练模型

#####                                                                                 训练参数

| 文本最大长度 | 学习率 | 梯度累积 | batchsize_size | 权重衰减 |
| ------------ | ------ | -------- | -------------- | -------- |
| 300          | 8e-6   | 3        | 10             | 2e-4     |



1. main1.py中使用mrcmac_large预训练模型，并使用fgm对抗学习，训练出一个模型，并在最后使用1e-6的学习率对划分的valid集一并训练。训练出的模型保存为1_mrcmac_fgm.pt
2. main2.py中使用roberta_large预训练模型，并使用fgm对抗学习，训练出一个模型，并在最后使用1e-6的学习率对划分的valid集一并训练。训练出的模型保存为2_ro_fgm.pt
3. main3.py中使用macbert_large预训练模型，并使用fgm对抗学习，训练出一个模型，并在最后使用1e-6的学习率对划分的valid集一并训练。训练出的模型保存为3_mac_fgm.pt
4. main4.py中使用mrcmac_large预训练模型，并使用fgm+rdrop，训练出一个模型，并在最后使用1e-6的学习率对划分的valid集一并训练。训练出的模型保存为4_mrc_fgm_rdrop.pt
5. main5.py中使用mrcmac_large预训练模型，并使用fgm，同时使用编辑距离扩增样本训练出一个模型，并在最后使用1e-6的学习率对划分的valid集一并训练。训练出的模型保存为5_mrcmac_fgm_kz.pt

```
python main1.py
python main2.py
python main3.py
python main4.py
python main5.py
```



### 使用开源数据辅助查找易错样本

我们对[Chinese-medical-dialogue-data-master](https://github.com/Toyhom/Chinese-medical-dialogue-data)数据进行伪标签处理，并仅使用此伪标签数据训练模型。

`python prediction.py` `python train_other.py`

之后使用此伪标签模型，对train集标注，并从中找到train集容易错的样本，作为error集。

`python prediction_train.py`



#### 使用易错样本辅助训练模型

将error集一并加入训练

```
python main6.py
python main7.py
python main8.py
```

1. main6.py中使用mrcmac_large预训练模型，并使用fgm对抗学习，训练出一个模型，在除第一轮外使用error集训练模型，并在最后使用1e-6的学习率对划分的valid集一并训练。训练出的模型保存为6_mrcmac_error_fgm.pt
2. main7.py中使用mrcmac_large预训练模型，并使用fgm对抗学习，训练出一个模型，在每一轮都使用error集，并在最后使用1e-6的学习率对划分的valid集一并训练。训练出的模型保存为7_mrcmac_error_fgm.pt
3. main8.py中使用mrcmac_large预训练模型，并使用fgm对抗学习，训练出一个模型，首轮使用编辑距离扩增样本，除此之外使用error集训练数据。并在最后使用1e-6的学习率对划分的valid集一并训练。训练出的模型保存为8_mrcmac_error_fgm.pt



### 在A榜数据上微调

#####                                                                                 训练参数

| 文本最大长度 | 学习率 | 梯度累积 | batchsize_size | 权重衰减 |
| ------------ | ------ | -------- | -------------- | -------- |
| 300          | 2e-6   | 3        | 10             | 2e-4     |

此处为防止较大学习率造成模型遗忘，因此使用较小学习率微调模型

对A榜进行预测，并将最高分数据给原模型训练。

```
python testA.py
python train_testA.py
```



### 预测B榜

代码与预测代码相同