# -*- coding = utf-8 -*-
# 2023/3/11 15:07

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        # 少样本版
        self.train_path = dataset + '/fewshotdata/train.txt'                                # 训练集
        self.dev_path = dataset + '/fewshotdata/dev.txt'                                    # 验证集
        self.test_path = dataset + '/fewshotdata/test.txt'                                  # 测试集
        # self.train_path = dataset + '/data/train.txt'                                # 训练集
        # self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        # self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.pt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 32                                            # mini-batch大小
        self.learning_rate = 5e-5                                       # 学习率
        # self.bert_path = './bert_pretrain'
        self.bert_path = 'bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.freeze_bert = False


class BertClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        if config.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits
