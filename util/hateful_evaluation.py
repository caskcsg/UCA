from types import DynamicClassAttribute
import pandas as pd
import json
import csv
import torchmetrics
from sklearn.metrics import accuracy_score
import torch
import numpy as np


def cul(pre_path, label_path):
    tmp_dic = {}
    true_label = []
    pre_label = []
    pre_prob = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tmp = line.split(' ') 
            tmp_dic[tmp[0]] = int(tmp[1].replace('\n',''))
        
    with open(pre_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tmp = line.split(' ')
            pre_label.append(int(tmp[1]))
            true_label.append(tmp_dic[tmp[0]])
            pre_prob.append(float(tmp[2].replace('\n','')))
    acc = accuracy_score(true_label, pre_label)
    auc_roc = torchmetrics.AUROC(task="binary")(torch.tensor(pre_prob), torch.tensor(true_label))
    print("acc:", acc)
    print("auc_roc:", auc_roc.cpu().numpy())


if __name__ == '__main__':
    #cul('./hate_result.txt', './hate_test.txt') ## 预测结果文件   测试集真实标签文件
    # with open('/workspace/twitter/jsonl/test.jsonl','r',encoding='utf8') as fp:
    #     f = open('/workspace/twitter/twitter_label.txt', 'w')
    #     for line in fp.readlines():
    #         dic = json.loads(line)
    #         id = dic["id"]
    #         label = dic["label"]
    #         f.writelines(str(id) + " "+ str(label)+"\n")
    #     f.close()

    # with open('/workspace/hateful_memes/test_seen.jsonl','r',encoding='utf8') as fp:
    #     f = open('/workspace/hateful_memes/test.txt', 'w')
    #     for line in fp.readlines():
    #         dic = json.loads(line)
    #         dicc = {}
    #         id = dic["id"]
    #         label = dic["label"]
    #         f.writelines(id + " "+ str(label)+"\n")
    #     f.close()

    # with open('/workspace/twitter/test.jsonl','r',encoding='utf8') as fp:
    #     f = open('/workspace/twitter/test.txt', 'w')
    #     for line in fp.readlines():
    #         dic = json.loads(line)
    #         dicc = {}
    #         id = dic["id"]
    #         label = dic["label"]
    #         f.writelines(str(id) + " "+ str(label)+"\n")
    #     f.close()

    with open('/workspace/Harm_C/test.jsonl','r',encoding='utf8') as fp:
        f = open('/workspace/Harm_C/harmC_label.txt', 'w')
        for line in fp.readlines():
            dic = json.loads(line)
            id = dic["id"]
            if dic["labels"] == ["not harmful"]:
                label = 0
            else :
                label = 1
            f.writelines(str(id) + " "+ str(label)+"\n")
        f.close()

# with open('/workspace/hateful_memes/TLM/train_hateful.jsonl','r',encoding='utf8')as fp:
#     a = []
#     keys = []
#     writer = csv.writer('/workspace/twitter/ALBEF/train.csv')
#     for line in fp.readlines():
#         dic = json.loads(line)
#         dicc = {}
#         dicc["image"] = dic["id"]
#         dicc["sentence"] = dic["text"]
#         dicc["label"] = dic["label"]
#         a.append(dicc)
#     with open('/workspace/twitter/ALBEF/train.csv','w',encoding='utf8')as fp:
#             csv.writer(a, fp)

# with open('/workspace/hateful_memes/TLM/train_twitter.jsonl','r',encoding='utf8')as fp:
#     a = []
#     for line in fp.readlines():
#         dic = json.loads(line)
#         dicc = {}
#         dicc["image"] = dic["id"]
#         dicc["sentence"] = dic["text"]
#         dicc["label"] = dic["label"]
#         a.append(dicc)
#     with open('/workspace/hateful_memes/TLM/train_twitter.csv','w',encoding='utf8')as fp:
#             json.dump(a, fp)