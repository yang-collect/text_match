from pathlib import Path
import os
import sys

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)

path = Path(__file__).parent.parent
train_path = path.joinpath('./data/lcqmc/train.tsv')
test_path = path.joinpath('./data/lcqmc/test.tsv')
dev_path = path.joinpath('./data/lcqmc/dev.tsv')

model_path = path.parent.joinpath('./model_file/ernie-1.0')  # 预训练模型路径

save_path = path.parent.joinpath('./model_file/doc_match')  # fine-tune后的模型路径

cust_path = path.parent.joinpath('./model_file/dsm_bert')

max_length = 50  # 由于训练集中文本长度的长度最大的第二个为49，而测试集的最大文本长度为29，故假定该段文本数据集的文本长度最大值为50

batch_size = 128

epochs = 6

num_warmup_steps = 500

score_path = str(path.parent.joinpath('./metrics/accuracy.py'))
# print(f1_path)
