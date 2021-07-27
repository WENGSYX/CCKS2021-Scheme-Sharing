## CCKS 2021答非所问 白[MASK] 队 解决方案

### CCKS 2021：面向中文医疗科普知识的内容理解（二）医疗科普知识答非所问识别 冠军方案

##### 大致方案，使用基于词向量替换相似词和通过编辑距离查找相似语句对数据进行扩增；通过使用fgm+rdrop增加模型鲁棒性；仅使用伪标签数据训练模型，之后使用此模型反向标注train集，找到train集中容易出错的题，使用此错题多次训练；使用官方测试集的伪标签数据进行训练；使用[mrc_macbert预训练模型](https://github.com/basketballandlearn/MRC_Competition_Dureader)。

配置 ：

```
系统：ubuntu20.04
cpu：10900X
gpu：4 X 3090
内存：104G
```

数据：

```
开源无标注语料：
公开链接：https://github.com/Toyhom/Chinese-medical-dialogue-data
已下载，并存放至 '训练/Chinese-medical-dialogue-data-master'

官方数据集存放位置：
训练/wa.train.fixtab.valid.tsv
训练/wa.test.phase1.fixtab.valid.tsv
训练/test.phase2.10k.tsv.docid.send.tsv
预测/test.phase2.10k.tsv.docid.send.tsv
```



具体内容：

```
预测相关可查看 '预测/Readme.md'
训练相关可查看 '训练/Readme.md'
```

