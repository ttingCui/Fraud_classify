# -*- coding = utf-8 -*-
# 2023/3/10 22:14

import sys
import torch
import torch.nn as nn

from read_data_prompt import ret_loader
from models.MMLbert import Config, BertClassifier
from logger import logger

from torch.utils.data import Dataset



def train(model, train_loader, val_loader, test_loader, optimizer, criterion, config, log_interval=20, eval_interval=200):
    device = config.device
    model.to(device)
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(config.num_epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            # bsize, classnum
            outputs = model(input_ids, attention_mask=attention_mask)
            # pred = torch.argmax(outputs, dim=-1)
            labels = (labels-int(config.classifier_indices[0])).squeeze()
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if (i + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                train_losses.append(avg_loss)
                logger.info(f'Epoch [{epoch + 1}/{config.num_epochs}] Batch [{i + 1}] Loss: {avg_loss:.4f}')
                running_loss = 0.0

            if (i + 1) % eval_interval == 0:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                if val_acc > best_val_acc:
                    logger.info(f'New best validation accuracy: {val_acc:.4f}')
                    best_val_acc = val_acc
                    best_model_params = model.state_dict()
                    torch.save(best_model_params, config.save_path)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    logger.info(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')

    return train_losses, val_losses, val_accs


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            # logits = outputs.logits
            labels = (labels-int(config.classifier_indices[0])).squeeze()

            loss = criterion(outputs, labels)

            predicted = torch.argmax(outputs, dim=-1)
            total_correct += (predicted == labels).sum().item()
            total_count += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total_count
    accuracy = total_correct / total_count

    return avg_loss, accuracy

# labeldict = {
#     "0": "贷款广告",
#     "1": "网络广告",
#     "2": "其他广告",
#     "3": "房产广告",
#     "4": "零售广告",
#     "5": "金融欺诈",
#     "6": "其他欺诈",
#     "7": "银行钓鱼",
#     "8": "非法钓鱼",
#     "9": "非法陪护",
#     "10": "假证假票",
#     "11": "非法赌博",
#     "12": "非法政治",
#     "13": "私人交流",
# }
labeldict = {
    "0": "[MASK0]",
    "1": "[MASK1]",
    "2": "[MASK2]",
    "3": "[MASK3]",
    "4": "[MASK4]",
    "5": "[MASK5]",
    "6": "[MASK6]",
    "7": "[MASK7]",
    "8": "[MASK8]",
    "9": "[MASK9]",
    "10": "[MASK10]",
    "11": "[MASK11]",
    "12": "[MASK12]",
    "13": "[MASK13]",
}
sentence = "该短信涉及"
# 加载数据集位置 获取相关配置信息
# dataset = sys.argv[1]
dataset = "message/fewshotdata_8"
config = Config(dataset)
# 将特殊token列表添加到tokenizer的词汇表中
config.tokenizer.add_tokens(config.special_tokens)
# 加载预训练模型
model = BertClassifier(config)
model.resize_token_embeddings(len(config.tokenizer))
model.classifier.load_cls(config)
model.classifier.update_classifier(torch.as_tensor(config.classifier_indices, dtype=torch.long))
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 将模型移到对应设备上
model.to(config.device)
# 将数据传递给DataLoader
train_dataloader = ret_loader(config.train_path, config.tokenizer, config.batch_size, sentence, labeldict)
dev_dataloader = ret_loader(config.dev_path, config.tokenizer, config.batch_size, sentence, labeldict)
test_dataloader = ret_loader(config.test_path, config.tokenizer, config.batch_size, sentence, labeldict)
# 训练
train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, criterion, config)