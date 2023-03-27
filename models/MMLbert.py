# -*- coding = utf-8 -*-
# 2023/3/19 19:20

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForMaskedLM

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bertprompt'
        self.train_path = dataset + '/train.txt'                                # 训练集
        self.dev_path = dataset + '/dev.txt'                                    # 验证集
        self.test_path = dataset + '/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open('message/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.pt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 8                                             # mini-batch大小
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = 'bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.mask_id = 103
        self.vocab_size = 21142
        self.freeze_bert = False
        self.classifier_indices = [21128, 21129, 21130, 21131, 21132, 21133, 21134, 21135, 21136, 21137, 21138, 21139, 21140, 21141]
        self.special_tokens = ["[MASK0]", "[MASK1]", "[MASK2]", "[MASK3]", "[MASK4]", "[MASK5]", "[MASK6]", "[MASK7]",
                          "[MASK8]", "[MASK9]", "[MASK10]", "[MASK11]", "[MASK12]", "[MASK13]"]





class BertClassifier(nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.bert = BertForMaskedLM.from_pretrained(config.bert_path)
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.classifier = Decoder(config.hidden_size, config.vocab_size)
        self.mask_id = config.mask_id

    def forward(self, input_ids, attention_mask):
        _mask = input_ids.eq(self.mask_id)
        # output: MaskedLMOutput [bsize, sql, hidden]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs.last_hidden_state
        # outputs = outputs.logits
        # polled_output: [batchsize, hidden]
        pooled_output = outputs[_mask]
        # logits: [batchsize, classvocabnum]
        logits = self.classifier(pooled_output, word_prediction=True)
        return logits

    def resize_token_embeddings(self, lens):
        self.bert.resize_token_embeddings(lens)


class Decoder(nn.Module):

    def __init__(self, isize, nwd):
        super().__init__()
        self.classifier = nn.Linear(isize, nwd)
        self.lsm = nn.LogSoftmax(-1)

    def forward(self, inpute, word_prediction=False):
        # out [bsize, classnum]
        out = self.classifier(inpute)
        if word_prediction:
            out = self.lsm(out)
        return out

    def update_classifier(self, indices):
        _nwd = indices.numel()
        _classifier = nn.Linear(self.classifier.weight.size(-1), _nwd, bias=self.classifier.bias is not None)

        with torch.no_grad():
            _classifier.weight.copy_(self.classifier.weight.index_select(0, indices))
            if self.classifier.bias is not None:
                _classifier.bias.copy_(self.classifier.bias.index_select(0, indices))
        self.classifier = _classifier