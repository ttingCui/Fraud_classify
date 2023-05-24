# -*- coding = utf-8 -*-
# 2023/3/10 22:27

from read_data import *

class DataGeneLabel(Dataset):
    def __init__(self, texts, labels, tokenizer, sentence):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.sentence = sentence

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.sentence[0] + self.texts[idx] + self.sentence[1] + '[MASK]'
        label = self.labels[idx]
        # inputs = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=128,
        #                         return_tensors='pt')
        inputs = self.tokenizer(text, truncation=True, padding=True)
        return {'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'labels': label}

# 返回dataloader
def makeDatasetLabel(batch_size, texts, labels, tokenizer, sentence):
    # 创建一个 DataCollatorWithPadding
    collator = DataCollatorWithPadding(padding=True, tokenizer=tokenizer)

    # 构造 Dataset 对象
    train_dataset = DataGeneLabel(texts, labels, tokenizer, sentence)

    # 创建 DataLoader 对象
    data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    return data_loader

def retLoaderLabel(filename, tokenizer, batch_size, sentence):
    contents, labels = loadData(filename)
    loader = makeDatasetLabel(batch_size, contents, labels, tokenizer, sentence)
    return loader