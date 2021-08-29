from datasets import load_metric
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
import os
import sys

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)

from dsm_bert import config
from dsm_bert.modeling import Dsm_Bert
from dsm_bert.DataProcess import DataLoad


def evaluate(model, eval_dataloader):
    """
    根据传入模型以及数据集计算accuracy
    """
    # 加载accuracy评估器
    metric = load_metric(config.score_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 模型评估过程
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            batch_query = {k: v[0].to(device) for k, v in batch.items() if k != 'labels'}
            batch_title = {k: v[1].to(device) for k, v in batch.items() if k != 'labels'}
            prob = model(batch_query, batch_title)
            prob = F.softmax(prob, dim=-1)
            # 对模型输出取argmax
            predictions = torch.argmax(prob, dim=-1)
            # 将当前批次数据的预测结果和原始结果传递给评估器
            metric.add_batch(predictions=predictions, references=batch['labels'].to(device))

    # 返回评估其计算结果
    return metric.compute()


if __name__ == '__main__':
    # print(config.score_path)
    # metric = load_metric(config.score_path)
    # print(metric)
    # 模型保存路径
    model_path = config.cust_path
    # 加载预训练模型
    model = Dsm_Bert.from_pretrained(model_path)
    # tokenizer实例化
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # 训练即测试数据加载
    train_dataloder = DataLoad(tokenizer, config.train_path)
    test_dataloder = DataLoad(tokenizer, config.test_path)
    # 打印训练的accuracy
    print('train data:', evaluate(model, train_dataloder))
    # 打印测试数据上的accuracy
    print('test data:', evaluate(model, test_dataloder))
    # 加载评估数据
    eval_dataloader = DataLoad(tokenizer, config.dev_path)
    # 打印在评估数据上的accuracy
    print('dev data', evaluate(model, eval_dataloader))
