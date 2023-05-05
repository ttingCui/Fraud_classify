# -*- coding = utf-8 -*-
# 2023/3/30 16:05
# -*- coding = utf-8 -*-
# 2023/3/19 19:20

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform, BertLMPredictionHead
from transformers import BertModel, BertTokenizer, BertForPreTraining

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'robertaprompt'
        self.train_path = dataset + '/train.txt'                                # 训练集
        self.dev_path = dataset + '/dev.txt'                                    # 验证集
        self.test_path = dataset + '/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open('message/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.pt'        # 模型训练结果
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.device = torch.device('cpu')   # 设备

        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 4                                             # mini-batch大小
        self.learning_rate = 5e-5                                       # 学习率
        # self.learning_rate = 1e-5                                     # 学习率
        self.bert_path = 'hfl/chinese-roberta-wwm-ext'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.mask_id = 103
        self.freeze_bert = False

        self.attention_probs_dropout_prob = 0.1,
        self.directionality = "bidi",
        self.hidden_act = "gelu",
        self.hidden_dropout_prob = 0.1,
        self.initializer_range = 0.02,
        self.intermediate_size = 3072,
        self.layer_norm_eps = 1e-12,
        self.max_position_embeddings = 512,
        self.model_type = "bert",
        self.num_attention_heads = 12,
        self.num_hidden_layers = 12,
        self.pad_token_id = 0,
        self.pooler_fc_size = 768,
        self.pooler_num_attention_heads = 12,
        self.pooler_num_fc_layers = 3,
        self.pooler_size_per_head = 128,
        self.pooler_type = "first_token_transform",
        self.type_vocab_size = 2,
        self.vocab_size = 21128


class MyBertModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.cls = classify(config)
        self.mask_id = config.mask_id

    def forward(self, input_ids, attention_mask):
        _mask = input_ids.eq(self.mask_id)
        # output: MaskedLMOutput [bsize, sql, hidden]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs.last_hidden_state
        # polled_output: [batchsize, hidden]
        pooled_output = outputs[_mask]
        # logits: [batchsize, classvocabnum]
        logits = self.cls(pooled_output)
        return logits


class classify(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.lsm = nn.LogSoftmax(-1)
        # [0.213, 0.323, ..., 0.9897]
    def forward(self, inpute):
        # out [bsize, classnum]
        hidden_states = self.transform(inpute)
        hidden_states = self.decoder(hidden_states)
        hidden_states = self.lsm(hidden_states)
        return hidden_states

    # 加载之前的预训练模型的权重
    def load_cls(self, config):
        bert = BertForPreTraining.from_pretrained(config.bert_path)
        bert_decoder = bert.cls.predictions.decoder
        bert_transform = bert.cls.predictions.transform
        with torch.no_grad():
            self.decoder.weight.copy_(bert_decoder.weight)
            self.decoder.bias.copy_(bert_decoder.bias)
        self.transform = bert_transform

    # 更新classifier
    def update_classifier(self, indices, labellen=4):
        # _nwd = len(indices_list)
        _nwd = indices.numel() // labellen
        _classifier = nn.Linear(self.decoder.weight.size(-1), _nwd, bias=self.decoder.bias is not None)

        with torch.no_grad():
            weight = self.decoder.weight.index_select(0, indices)
            weight = weight.view(_nwd, labellen, self.decoder.weight.size(-1)).mean(dim=1)
            _classifier.weight.copy_(weight)
            if self.decoder.bias is not None:
                bias = self.decoder.bias.index_select(0, indices)
                bias = bias.view(_nwd, labellen).mean(dim=1)
                _classifier.bias.copy_(bias)
        self.decoder = _classifier