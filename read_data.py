# -*- coding = utf-8 -*-
# 2023/3/10 22:27

import h5py
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, DataCollatorWithPadding, AutoTokenizer


class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # 使用 tokenizer 对文本进行编码
        encoding = self.tokenizer(text, truncation=True, padding=True)
        return {"input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "token_type_ids": encoding["token_type_ids"],
                "labels": label}

# 加载模型数据
def loadData(filename):
    contents = []
    labels = []
    with open(filename, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            # 划分文本内容和分类标签
            content, label = lin.split('\t')
            contents.append(content)
            labels.append(int(label))
    return contents, labels

# 加载模型数据(前面增加标签词版本)
def loadDataAddLabel(filename):
    contents = []
    labels = []
    class_list = [x.strip() for x in open('message/data/class_zh.txt', encoding='UTF-8').readlines()]
    with open(filename, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            # 划分文本内容和分类标签
            content, label = lin.split('\t')
            contents.append(class_list[int(label)]+content)
            labels.append(int(label))
    return contents, labels

# 返回dataloader
def makeDataset(batch_size, texts, labels, tokenizer):
    # 创建一个 DataCollatorWithPadding
    collator = DataCollatorWithPadding(padding=True, tokenizer=tokenizer)

    # 构造 Dataset 对象
    train_dataset = MyDataset(texts, labels, tokenizer)

    # 创建 DataLoader 对象
    data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    return data_loader

def retLoader(filename, tokenizer, batch_size, add_label=False):
    if add_label:
        contents, labels = loadDataAddLabel(filename)
    else:
        contents, labels = loadData(filename)
    loader = makeDataset(batch_size, contents, labels, tokenizer)
    return loader