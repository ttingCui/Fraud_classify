# -*- coding = utf-8 -*-
# 2023/3/30 16:05
# -*- coding = utf-8 -*-
# 2023/3/10 22:14

import sys
import os
import torch
import torch.nn as nn

from read_data_prompt import retLoaderLabel
from models.prompt_bert import Config, MyBertModel
from logger import logger

from torch.utils.data import Dataset



def train(model, train_loader, val_loader, test_loader, optimizer, criterion, config, log_interval=5, eval_interval=10):
    device = config.device
    model.to(device)
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accs = []
    stop_training = False

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
            # labels = (labels-int(config.classifier_indices[0])).squeeze()
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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
                    config.counter = 0
                else:
                    config.counter = config.counter + 1

            if config.counter >= config.patience:
                print("Validation performance did not improve for {} eval. Training stopped.".format(config.patience))
                stop_training = True
                break

                # ...

        if stop_training:
            break

    if os.path.exists(config.save_path):
        model_state_dict = torch.load(config.save_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_state_dict)
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
            # labels = (labels-int(config.classifier_indices[0])).squeeze()

            loss = criterion(outputs, labels)

            predicted = torch.argmax(outputs, dim=-1)
            total_correct += (predicted == labels).sum().item()
            total_count += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total_count
    accuracy = total_correct / total_count

    return avg_loss, accuracy
# labeldict = {
#     "0": "贷款",
#     "1": "网络",
#     "2": "广告",
#     "3": "房产",
#     "4": "零售",
#     "5": "金融",
#     "6": "欺诈",
#     "7": "银行",
#     "8": "钓鱼",
#     "9": "陪护",
#     "10": "证票",
#     "11": "赌博",
#     "12": "政治",
#     "13": "私人",
# }
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

sentence = "该短信涉及"
# 加载数据集位置 获取相关配置信息
dataset = sys.argv[1]
# dataset = "message/new_data/fewshot_4"
config = Config(dataset)

# 加载预训练模型
model = MyBertModel(config)
model.cls.load_cls(config)
indices_list = []
with open("label_id_2.txt", 'r', encoding="UTF-8") as f:
    for line in f.readlines():
        # indices = []
        line = line.strip()
        # indices.extend([int(i) for i in line.split(" ")])
        indices_list.extend(int(i) for i in line.split(" "))
        # indices_list.append(indices)


# model.cls.update_classifier(indices_list)
model.cls.update_classifier(torch.as_tensor(indices_list, dtype=torch.long), labellen=2)
# 定义优化器
# Adam betas=(0.9, 0.98,), eps=1e-9
optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), config.learning_rate, betas=(0.9, 0.98,), eps=1e-9)
# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 将模型移到对应设备上
model.to(config.device)
# 将数据传递给DataLoader
train_dataloader = retLoaderLabel(config.train_path, config.tokenizer, config.batch_size, sentence)
dev_dataloader = retLoaderLabel(config.dev_path, config.tokenizer, config.batch_size, sentence)
test_dataloader = retLoaderLabel(config.test_path, config.tokenizer, config.batch_size, sentence)
# 训练
train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, criterion, config)