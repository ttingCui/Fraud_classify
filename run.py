# -*- coding = utf-8 -*-
# 2023/3/10 22:14


import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

from read_data import ret_loader
from bert import Config, BertClassifier
from importlib import import_module

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.texts = [d[0] for d in data]
        self.labels = [d[1] for d in data]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=128,
                                return_tensors='pt')
        return {'input_ids': inputs['input_ids'][0], 'attention_mask': inputs['attention_mask'][0],
                'labels': torch.tensor(self.labels[idx])}


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

            outputs = model(input_ids, attention_mask=attention_mask)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if (i + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                train_losses.append(avg_loss)
                print(f'Epoch [{epoch + 1}/{config.num_epochs}] Batch [{i + 1}] Loss: {avg_loss:.4f}')
                running_loss = 0.0

            if (i + 1) % eval_interval == 0:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                if val_acc > best_val_acc:
                    print(f'New best validation accuracy: {val_acc:.4f}')
                    best_val_acc = val_acc
                    best_model_params = model.state_dict()
                    torch.save(best_model_params, config.save_path)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')

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
            logits = outputs.logits
            loss = criterion(logits, labels)

            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_count += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total_count
    accuracy = total_correct / total_count

    return avg_loss, accuracy


# 加载数据集位置 获取相关配置信息
dataset = "message"
config = Config(dataset)
# 加载预训练模型
model = BertClassifier(config)
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 将模型移到对应设备上
model.to(config.device)
# 将数据传递给DataLoader
train_dataloader = ret_loader(config.train_path, config.tokenizer, config.batch_size)
dev_dataloader = ret_loader(config.dev_path, config.tokenizer, config.batch_size)
test_dataloader = ret_loader(config.test_path, config.tokenizer, config.batch_size)
# 训练
train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, criterion, config)