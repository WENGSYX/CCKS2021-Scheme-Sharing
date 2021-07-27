import pandas as pd
import ast
from tqdm import trange
import random
train = []
train_data = []
def ap(c,train_data):
    for i in range(len(c)):
        item = c.iloc[i]
        if item['ask'] == '无':
            continue
        if len(item['title']) < 10 or len(item['ask']) < 10:
            continue
        if len(item['title']) + len(item['ask']) + len(item['answer']) > 300 :
            continue
        train_data.append({'label':1,'docid':0,'question':item['title'],'description':item['ask'],'answer':item['answer']})
    return train_data
c1 = pd.read_csv('Chinese-medical-dialogue-data-master/Data_数据/Andriatria_男科/男科5-13000.csv',encoding='ANSI')
c2 = pd.read_csv('Chinese-medical-dialogue-data-master/Data_数据/IM_内科/内科5000-33000.csv',encoding='ANSI')
c3 = pd.read_csv('Chinese-medical-dialogue-data-master/Data_数据/OAGD_妇产科/妇产科6-28000.csv',encoding='ANSI')
c4 = pd.read_csv('Chinese-medical-dialogue-data-master/Data_数据/Oncology_肿瘤科/肿瘤科5-10000.csv',encoding='ANSI')
c5 = pd.read_csv('Chinese-medical-dialogue-data-master/Data_数据/Pediatric_儿科/儿科5-14000.csv',encoding='ANSI')
c6 = pd.read_csv('Chinese-medical-dialogue-data-master/Data_数据/Surgical_外科/外科5-14000.csv',encoding='ANSI')
train_data = ap(c1,train_data)
train_data = ap(c2,train_data)
train_data = ap(c3,train_data)
train_data = ap(c4,train_data)
train_data = ap(c5,train_data)
train_data = ap(c6,train_data)

train_data = pd.DataFrame(train_data)
train_data.to_csv('other.csv')
