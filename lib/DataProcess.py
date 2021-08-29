import torch
from torch.utils.data import DataLoader, Dataset
from jieba import cut
import os
import sys

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)

from lib import config


def read_data(path=config.train_path):
    """ 读取数据并将query, title, label提取出来
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        query = []
        title = []
        label = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            line = line.strip().split('\t')
            query.append(list(cut(line[0])))
            title.append(list(cut(line[1])))
            if len(line) == 2:
                continue
            else:
                label.append(int(line[2]))
    return query, title, label


class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        if len(data) == 3:
            query, title, label = data
        else:
            query, title = data
        self.tokenizer = tokenizer
        self.length = len(query)
        # 获取input_ids,attention,由于每个模型只打算输入一句，故toke_type_ids必然全部为0，可选择不输入
        self.input_ids, self.mask_attention, self.token_type_ids = self.encode(query, title)
        self.label = None
        if len(label) > 0:
            self.label = torch.tensor(label)

    def encode(self, text_list, title_list):

        token = self.tokenizer(
            text_list, title_list,
            add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
            max_length=config.max_length,  # 设定最大文本长度
            padding=True,  # padding
            truncation=True,  # truncation
            is_split_into_words=True,  # 是否分词
            return_tensors='pt'  # 返回的类型为pytorch tensor
        )

        return token['input_ids'], token['attention_mask'], token['token_type_ids']

    def __getitem__(self, index):
        if self.label is not None:
            return {'input_ids': self.input_ids[index],
                    'attention_mask': self.mask_attention[index],
                    'token_type_ids': self.token_type_ids[index],
                    'labels': self.label[index]}
        else:
            return {'input_ids': self.input_ids[index],
                    'attention_mask': self.mask_attention[index],
                    'token_type_ids': self.token_type_ids[index]}

    def __len__(self):
        return self.length


def DataLoad(tokenizer, path=config.train_path):
    """ 根据数据构建dataloder
    """
    data = read_data(path)
    # 构建dataset
    dataset = MyDataset(data, tokenizer)

    # 构建dataloder
    dataloder = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    return dataloder

# if __name__ == '__main__':
#     from transformers import BertModel, BertTokenizer
#
#     model_path = config.model_path
#     model = BertModel.from_pretrained(model_path)
#     tokenizer = BertTokenizer.from_pretrained(model_path)
#     data = DataLoad(tokenizer, config.test_path)
#     test = next(iter(data))
#     # print(test['input_ids'].shape)
#     # print(test['token_type_ids'].shape)
#     # print(test['attention_mask'].shape)
#     print(test)