import random

import numpy as np
import pandas as pd
from tqdm import tqdm,trange
import Levenshtein
import os


pt = pd.read_csv('wa.train.fixtab.valid.tsv',sep='\t')

t = []
for p in trange(len(pt)):
    item = pt.iloc[p]
    t.append({'label': item['Label'], 'docid': item['Docid'], 'question': item['Question'],
              'description': item['Description'], 'answer': item['Answer']})
answers = []
o = []

for item in tqdm(t):
    question = item['question']
    len_num = len(question)
    f = {}
    best_distance = 1000
    if item['label'] == 0:
        for itemt1 in t:
            questiont1 = itemt1['question']
            len_num2= len(questiont1)
            distance = Levenshtein.distance(question,questiont1)
            if distance < best_distance and distance>1 and itemt1['answer'] not in answers and itemt1['label']==0:
                best_distance = distance
                f = {'label': 1, 'docid': item['docid'], 'question': item['question'],
                          'description': item['description'], 'answer': itemt1['answer']}
                if best_distance < 3:
                    break
        if f != {} and best_distance < 20:
            answers.append(f['answer'])
            o.append({'label': 0, 'docid': item['docid'], 'question': item['question'],
                      'description': item['description'], 'answer': item['answer']})
            o.append(f)

o = pd.DataFrame(o)
o.to_csv('edit_distance.csv')