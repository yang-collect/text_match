# 数据集介绍

数据集采用的是lcqmc，是一个开源的文本匹配数据，其中每行数据有`txet_a`、`text_b`和`label`，也就是两段短文本和一个是否匹配的标签，`0表示不匹配，1表示匹配`。

# 模型思路

将两段文本当作两个输入的特征，将任务视为一个二分类任务，构建单塔模型和双塔模型。

单塔模型是直接将两个特征文本输入模型；双塔模型是分别对两个文本特征用bert进行embedding，其中embedding为bert的最后一个hidden state，并对embedding结果计算cosine相似度、线性变换和softmax，得到预测概率。

# 结果

`train.py` 输出的loss日志详见 `core/loss.txt 和 dsm_bert/loss.txt` 最终 

单塔模型的loss： 

`epoch:5,tran_loss:0.04637531887324783,valid loss;0.19695040583610535`

双塔模型的loss：

`epoch:5,tran_loss:0.37612259440689166,valid loss;0.40758877992630005`

`eval.py`长时间未计算出结果

其中，计算评分的函数为`accurarcy` 脚本见 `accurarcy.py`

`sever.py`测试时间消耗为`90ms`

![image-20210830063604686](C:\Users\wie\AppData\Roaming\Typora\typora-user-images\image-20210830063604686.png)

双塔模型的`sever.py`测试时间消耗为`69ms`

![image-20210830040314926](C:\Users\wie\AppData\Roaming\Typora\typora-user-images\image-20210830040314926.png)

# 文件结构

```
│  readme.md
│  requirements.txtnja# 依赖文件
│
├─.idea
│  │  .gitignore
│  │  misc.xml
│  │  modules.xml
│  │  workspace.xml
│  │  文本语义相似度计算.iml
│  │
│  └─inspectionProfiles
│          profiles_settings.xml
│          Project_Default.xml
│
├─core # 主要运行脚本
│      sever.py # 部署脚本
│      eval.py # 计算score脚本
│      predict.py # 单条预测脚本
│      train.py # 训练脚本
|	   loss.txt
|	   eval.txt
│
├─data# 数据文件
│  │  data_explode.ipynb
│  │
│  └─lcqmc
│          dev.tsv
│          test.tsv
│          train.tsv
│
├─dsm_bert # 双塔模型完整脚本
│  │  config.py 
│  │  DataProcess.py
│  │  eval.py
│  │  modeling.py # 自定义双塔模型
│  │  loss.txt # 记录的loss日志
│  │  sever.py # 部署服务脚本
│  │  train.py # 训练
│  │
│  └─__pycache__
│          config.cpython-37.pyc
│          DataProcess.cpython-37.pyc
│          modeling.cpython-37.pyc
└─lib # 配置以及数据处理
    │  config.py
    │  DataProcess.py
    │
    └─__pycache__
            config.cpython-37.pyc
            DataProcess.cpython-37.pyc
            modeling.cpython-37.pyc
```