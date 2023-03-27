# -*- coding = utf-8 -*-
# 2023/3/10 22:27

import h5py
import torch
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, DataCollatorWithPadding, AutoTokenizer


class DataGeneLabel(Dataset):
    def __init__(self, texts, labels, tokenizer, sentence):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.sentence = sentence

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx] + self.sentence + '[MASK]'
        label = self.labels[idx]
        # inputs = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=128,
        #                         return_tensors='pt')
        inputs = self.tokenizer(text, truncation=True, padding=True)
        labels = self.tokenizer.encode_plus(label, add_special_tokens=False)
        return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
                'labels': labels['input_ids']}

# 加载模型数据
def load_data(filename, labeldict):
    contents = []
    labels = []
    with open(filename, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            # 划分文本内容和类别内容
            content, label = lin.split('\t')
            contents.append(content)
            labels.append(labeldict[label])
    return contents, labels

# 返回dataloader
def makeDataset(batch_size, texts, labels, tokenizer, sentence):
    # 创建一个 DataCollatorWithPadding
    collator = DataCollatorWithPadding(padding=True, tokenizer=tokenizer)

    # 构造 Dataset 对象
    train_dataset = DataGeneLabel(texts, labels, tokenizer, sentence)

    # 创建 DataLoader 对象
    data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    return data_loader

def ret_loader(filename, tokenizer, batch_size, sentence, labeldict):
    contents, labels = load_data(filename, labeldict)
    loader = makeDataset(batch_size, contents, labels, tokenizer, sentence)
    return loader