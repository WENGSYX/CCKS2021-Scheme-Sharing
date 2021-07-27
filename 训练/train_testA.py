import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import os
import time
from sklearn.model_selection import *
from transformers import *
from torch.autograd import Variable

CFG = { #训练的参数配置
    'fold_num': 5, #五折交叉验证
    'seed': 2,
    'model': 'luhua/chinese_pretrain_mrc_macbert_large', #预训练模型
    'max_len': 300, #文本截断的最大长度
    'epochs': 8,
    'train_bs': 10, #batch_size，可根据自己的显存调整
    'valid_bs': 10,
    'lr': 8e-6, #学习率
    'num_workers': 0,
    'accum_iter': 2, #梯度累积，相当于将batch_size*2
    'weight_decay': 2e-4, #权重衰减，防止过拟合
    'device': 0,
    'batch_size_rdrop':2
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['seed']) #固定随机种子

torch.cuda.set_device(CFG['device'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_df =  pd.read_csv('白[MASK]_test_resultA.csv')
tokenizer = BertTokenizer.from_pretrained(CFG['model'])


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        label = self.df.label.values[idx]
        question = self.df.question.values[idx]
        description = self.df.description.values[idx]
        answer = self.df.answer.values[idx]

        question = question + '[SEP]' + description
        return question, answer, label

def collate_fn(data):
    input_ids, attention_mask, token_type_ids, label = [], [], [], []
    for x in data:
        text = tokenizer(x[0], text_pair=x[1], padding='max_length',
                         truncation=True, max_length=CFG['max_len'], return_tensors='pt')
        for i in range(CFG['batch_size_rdrop']):
            input_ids.append(text['input_ids'].squeeze().tolist())
            attention_mask.append(text['attention_mask'].squeeze().tolist())
            token_type_ids.append(text['token_type_ids'].squeeze().tolist())
            label.append(x[-1])
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    label = torch.tensor(label)
    return input_ids, attention_mask, token_type_ids, label

class CE_with_rdrop(nn.Module):
    def __init__(self, alpha=2.5, gamma=2, logits=True, reduce=True):
        super(CE_with_rdrop, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.CE = nn.CrossEntropyLoss()
        self.KLD = nn.KLDivLoss()

    def forward(self, inputs, targets):
        #targets = torch.eye(2)[targets.reshape(-1)].to(device)

        loss_ce = self.CE(inputs, targets)
        targets = torch.eye(2)[targets.reshape(-1)].to(device)

        loss_kl1 = self.KLD(F.log_softmax(
            inputs[::2]), F.softmax(targets[1::2]))
        loss_kl2 = self.KLD(F.log_softmax(
            inputs[1::2]), F.softmax(targets[::2]))

        return torch.mean(loss_ce) + torch.mean(loss_kl1+loss_kl2) / 4 * self.alpha
class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def macro_f1(pred,gold):
    intersection = 0
    GOLD = 0
    PRED = 0
    intersection1 = 0
    GOLD1 = 0
    PRED1 = 0
    for i in range(len(pred)):
        p = pred[i]
        g = gold[i]

        if p == 1:
            PRED1 += 1
        if g == 1:
            GOLD1 += 1
        if p == 1 and g == 1:
            intersection1 += 1
        if p == 0:
            PRED += 1
        if g == 0:
            GOLD += 1
        if p == 0 and g == 0:
            intersection += 1

    R = intersection / GOLD
    P = intersection / PRED
    F1 = 2 * P * R / (P + R)
    R1 = intersection1 / GOLD1
    P1 = intersection1 / PRED1
    F11 = 2 * P1 * R1 / (P1 + R1)

    return F1,GOLD,PRED,intersection,F11,GOLD1,PRED1,intersection1

def train_model(model, fgm,pgd,train_loader):  # 训练一个epoch
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()
    f1s  = AverageMeter()
    K=3
    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

    for step, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
        input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
            device), token_type_ids.to(device), y.to(device).long()

        with autocast():  # 使用半精度训练
            output = model(input_ids, attention_mask, token_type_ids)[0]
            loss = criterion(output, y) / CFG['accum_iter']
            scaler.scale(loss).backward()
            fgm.attack()  # 在embedding上添加对抗扰动
            output2 = model(input_ids, attention_mask, token_type_ids)[0]
            loss2 = criterion(output2, y) / CFG['accum_iter']
            scaler.scale(loss2).backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()
            if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()


        acc = (output.argmax(1) == y).sum().item() / y.size(0)

        losses.update(loss.item() * CFG['accum_iter'], y.size(0))
        accs.update(acc, y.size(0))

        tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg


def test_model(model, val_loader):  # 验证
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()
    y_truth, y_pred = [], []

    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
            input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device), y.to(device).long()

            output = model(input_ids, attention_mask, token_type_ids).logits

            y_truth.extend(y.cpu().numpy())
            y_pred.extend(output.argmax(1).cpu().numpy())

            loss = criterion(output, y)

            acc = (output.argmax(1) == y).sum().item() / y.size(0)

            losses.update(loss.item(), y.size(0))
            accs.update(acc, y.size(0))

            tk.set_postfix(loss=losses.avg, acc=accs.avg)
    F1,GOLD,PRED,intersection,F11,GOLD1,PRED1,intersection1 = macro_f1(pred=y_pred,gold=y_truth)
    print('GOLD:{} {}'.format(GOLD,GOLD1))
    print('PRED:{} {}'.format(PRED,PRED1))
    print('intersection:{} {}'.format(intersection,intersection1))
    print('F1:{} {}'.format(F1,F11))
    return losses.avg, accs.avg,F11

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name1='bert.embeddings.word_embeddings.weight',emb_name2='bert.embeddings.position_embeddings.weight',emb_name3='bert.embeddings.token_type_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
            if param.requires_grad and emb_name2 in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
            if param.requires_grad and emb_name3 in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name1='bert.embeddings.word_embeddings.weight',emb_name2='bert.embeddings.position_embeddings.weight',emb_name3='bert.embeddings.token_type_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                assert name in self.backup
                param.data = self.backup[name]
            if param.requires_grad and emb_name2 in name:
                assert name in self.backup
                param.data = self.backup[name]
            if param.requires_grad and emb_name3 in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name1='bert.embeddings.word_embeddings.weight',emb_name2='bert.embeddings.position_embeddings.weight',emb_name3='bert.embeddings.token_type_embeddings.weight', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)
            if param.requires_grad and emb_name2 in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)
            if param.requires_grad and emb_name3 in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name1='bert.embeddings.word_embeddings.weight',emb_name2='bert.embeddings.position_embeddings.weight',emb_name3='bert.embeddings.token_type_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
            if param.requires_grad and emb_name2 in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
            if param.requires_grad and emb_name3 in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            targets = torch.eye(2)[targets.reshape(-1)].to(device)
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            targets = torch.eye(2)[targets.reshape(-1)].to(device)
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


train = train_df.loc[:]


train_set = MyDataset(train)


train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                          num_workers=CFG['num_workers'])

best_acc = 0

model = BertForSequenceClassification.from_pretrained(CFG['model'],num_labels = 2).to(device)  # 模型

scaler = GradScaler()
optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])  # AdamW优化器
criterion = CE_with_rdrop()
scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                            CFG['epochs'] * len(train_loader) // CFG['accum_iter'])
# get_cosine_schedule_with_warmup策略，学习率先warmup一个epoch，然后cos式下降
fgm = FGM(model)
pgd = PGD(model)
best_F1 = 0
for md in ['3_mac_fgm','2_ro_fgm','6_mrcmac_error_fgm','7_mrcmac_error_fgm','8_mrcmac_error_fgm']:
    model.load_state_dict(torch.load(md + '.pt'))
    train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                              num_workers=CFG['num_workers'])
    time.sleep(0.2)

    train_loss, train_acc = train_model(model, fgm, pgd, train_loader)
    torch.save(model.state_dict(), md)

