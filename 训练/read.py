import pandas as pd
import ast
from tqdm import trange
import random
train = []
train_data = []

with open("wa.train.fixtab.valid.tsv", "r",encoding='utf-8') as f:
    for line in f.readlines()[1:]:
        line = line.strip('\n')
        line = line.split('\t')
        if len(line[2]+line[3]+line[4]) > 300:
            if 300-len(line[2])-len(line[3]) < 100:
                answer = line[4][:100]
                description = line[3][:200-len(line[2])]
            else:
                answer = line[4][:300-len(line[2])-len(line[3])]
                description = line[3]
        else:
            answer = line[4]
            description = line[3]
        train.append({'label':line[0],'docid':line[1],'question':line[2],'description':description,'answer':answer})

dev = train[:3000]
train = train[3000:]
train = pd.DataFrame(train)
dev = pd.DataFrame(dev)
train.to_csv('train.csv')
dev.to_csv('valid.csv')
