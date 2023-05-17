# -*- coding = utf-8 -*-
# 2023/3/10 22:14

import sys
import os
import torch
import torch.nn as nn

from read_data import retLoader
from models.bert import Config, BertClassifier
from logger import logger

from torch.utils.data import Dataset



def train(model, train_loader, val_loader, test_loader, optimizer, criterion, config, log_interval=5, eval_interval=10):
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

            outputs = model(input_ids, attention_mask=attention_mask)

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
            loss = criterion(outputs, labels)

            # torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_count += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total_count
    accuracy = total_correct / total_count

    return avg_loss, accuracy


# 加载数据集位置 获取相关配置信息
dataset = sys.argv[1]
# dataset = "message/new_data/fewshot_4"
config = Config(dataset)
add_label = True
# 加载预训练模型
model = BertClassifier(config)
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 将模型移到对应设备上
model.to(config.device)
# 将数据传递给DataLoader
train_dataloader = retLoader(config.train_path, config.tokenizer, config.batch_size, add_label)
dev_dataloader = retLoader(config.dev_path, config.tokenizer, config.batch_size, add_label)
test_dataloader = retLoader(config.test_path, config.tokenizer, config.batch_size, add_label)
# 训练
train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, criterion, config)