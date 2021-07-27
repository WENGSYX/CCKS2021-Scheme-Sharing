# CCKS 2021答非所问 白[MASK] 队 预测复现代码



#### 此文件夹为预测文件夹，运行此文件夹中代码，可复现最高分成绩。

```
复现所需python库：
pytorch 
transformers
tqdm
sklearn
pandas
numpy
```

请提前下载训练模型，并放置于此文件夹中。

链接：https://pan.baidu.com/s/1IClzpFgCHDKUe-Ah8Wd3Gg 
提取码：8888

共有以下五个模型

```
3_error_mac_csj3.pt
1_mrc_mac_csj3.pt
2_error_mac_csj3.pt
error_mac_csj3.pt
ro1_csj3.pt
```

另外‘test.phase2.10k.tsv.docid.send.tsv’和原最高分文件‘0.706975白[MASK]_test_result.txt’均已放置于此文件夹中。



通过运行test文件，将会读取‘‘test.phase2.10k.tsv.docid.send.tsv’‘并生成test2.csv文件，之后会依次通过模型进行预测，并保存相关预测分值。最后加权获得最优成绩。

`python test.py `



运行完成，将在此文件夹中生成’'白[MASK]_test_result.txt'‘文件，此文件分数将与最高分结果符合。
