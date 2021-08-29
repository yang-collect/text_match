import numpy as np
import torch
from torch import nn
from transformers import AdamW, get_scheduler, BertTokenizer

import os
import sys

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)
from dsm_bert import config
from dsm_bert import DataProcess, modeling


def compute_loss(model, val_data, loss_func):
    """Evaluate the loss and f1 score for an epoch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_data (dataset.PairDataset): The evaluation data set.

    Returns:
        numpy ndarray: The average loss of the dev set.
    """
    print('validating')
    # metric = load_metric("f1")
    # metric = load_metric("f1")
    val_loss = []
    with torch.no_grad():
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for batch, data in enumerate(val_data):
            batch_query = {k: v[0].to(device) for k, v in data.items() if k != 'labels'}
            batch_title = {k: v[1].to(device) for k, v in data.items() if k != 'labels'}
            prob = model(batch_query, batch_title)
            batch_label = data['labels'].to(device)
            loss = loss_func(prob, batch_label)
        val_loss.append(loss.item())

    return np.mean(val_loss)


def train(train_path=config.train_path, model_path=config.model_path, epochs=config.epochs,
          save_path=config.save_path, test_path=config.test_path):
    # 加载预训练模型和tokenizer
    model = modeling.Dsm_Bert.from_pretrained(config.model_path)
    # model = BertModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 加载训练和测试集
    train_dataloder = DataProcess.DataLoad(tokenizer, train_path)
    test_dataloder = DataProcess.DataLoad(tokenizer, test_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 实例化优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_training_steps = epochs * len(train_dataloder)
    # warm up
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 模型训练
    loss_func = nn.CrossEntropyLoss()  # 损失函数
    val_loss = np.inf
    model.to(device)
    model.train()

    for epoch in range(epochs):
        batch_loss = []
        for num, batch in enumerate(train_dataloder):
            # 分别加载 query 、title、 label
            batch_query = {k: v[0].to(device) for k, v in batch.items() if k != 'labels'}
            batch_title = {k: v[1].to(device) for k, v in batch.items() if k != 'labels'}
            prob = model(batch_query, batch_title)
            batch_label = batch['labels'].to(device)
            loss = loss_func(prob, batch_label)
            batch_loss.append(loss.item())
            # 梯度更新
            loss.backward()
            # 优化器和学习率更新
            optimizer.step()
            lr_scheduler.step()
            # 梯度清零
            optimizer.zero_grad()
            # 每100个打印一次结果
            if num % 1000 == 0:
                print(f'epoch:{epoch},batch :{num} ,train_loss :{loss} !')
        #
        epoch_loss = np.mean(batch_loss)
        avg_val_loss = compute_loss(model, test_dataloder, loss_func)
        print(f'epoch:{epoch},tran_loss:{epoch_loss},valid loss;{avg_val_loss}')
        print('*' * 100)
        # Update minimum evaluating loss.
        if avg_val_loss < val_loss:
            tokenizer.save_pretrained(config.save_path)
            model.save_pretrained(config.save_path)
            val_loss = avg_val_loss

    print(val_loss)


if __name__ == '__main__':
    train()
