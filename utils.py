import os
import copy
import torch 
import torch.nn.functional as F
import math
import json
import logging
import argparse
import datetime
import random
import numpy as np
from transformers import BertTokenizer, BertModel, AdamW

from tqdm import tqdm

cla2ds = {
    0:0, 1:0, 2:0, 3:0,
    4:1, 5:1, 6:1, 7:1, 8:1,
    9:2, 10:2, 11:2, 12:2, 13:2, 14:2, 15:2,
    16:2, 17:2, 18:2, 19:2, 20:2, 21:2, 22:2,
    23:3, 24:3, 25:3, 26:3, 27:3,
    28:3, 29:3, 30:3, 31:3, 32:3
}

def load_data(filepath, ds_name):
    if ds_name == 'agnews':
        offset = 0
    elif ds_name == 'amazon' or ds_name == 'yelp':
        offset = 4
    elif ds_name == 'dbpedia':
        offset = 9
    elif ds_name == 'yahoo':
        offset = 23
    else:
        assert 'Dataset should be AGNews, Amazon, DBPedia, Yahoo, or Yelp' == None

    dataset = json.load(open(filepath, encoding='utf-8'))
    dataset = [(i[1], int(i[0]) + offset) for i in dataset]

    return dataset

def data_gen(args, data_list, is_shuffle=False):
    if is_shuffle:
        random.shuffle(data_list)

    padding_len = args['padding_len']
    batch_size = args['batch_size']

    idx = 0
    while idx < len(data_list):
        ide = min(idx + batch_size, len(data_list))

        sent = [
            torch.LongTensor([i[0] + [args['pad_token']] * (padding_len - len(i[0]))] ) \
            if len(i[0]) < padding_len else \
            torch.LongTensor([i[0][:padding_len]])
            for i in data_list[idx:ide]
        ]

        mask = [
            torch.cat(
                [
                    torch.ones(1, len(i[0]), dtype=torch.int64),
                    torch.zeros(1, padding_len - len(i[0]), dtype=torch.int64)
                ],
                dim=1
            ) if len(i[0]) < padding_len else \
            torch.ones(1, padding_len, dtype=torch.int64)
            for i in data_list[idx:ide]
        ]

        sent = torch.cat(sent, dim=0)
        mask = torch.cat(mask, dim=0)

        label = torch.LongTensor([[i[1]] for i in data_list[idx:ide]])
        one_hot = torch.zeros(ide - idx, 33).scatter_(1, label, 1)

        ds_lbl = torch.LongTensor([[cla2ds[i[1]]] for i in data_list[idx:ide]])
        ds_one = torch.zeros(ide - idx, 4).scatter_(1, ds_lbl, 1)

        if len(data_list[idx]) > 2:
            prototype = torch.cat(
                [i[2] for i in data_list[idx:ide]],
                dim=0
            )
            idx += batch_size
            yield sent, mask, one_hot, prototype
        
        else:
            idx += batch_size
            yield sent, mask, one_hot, ds_one

def sample_mini_batch(args, data_list, start_index, batch_size):
    padding_len = args['padding_len']
    sample_list = []
    for i in range(batch_size):
        if start_index == 0:
            random.shuffle(data_list)

        sample_list.append(data_list[start_index])
        start_index += 1
        if start_index == len(data_list):
            start_index = 0

    sent = [
        torch.LongTensor([i[0] + [0] * (padding_len - len(i[0]))] ) \
        if len(i[0]) < padding_len else \
        torch.LongTensor([i[0][:padding_len]])
        for i in sample_list
    ]

    mask = [
        torch.cat(
            [
                torch.ones(1, len(i[0]), dtype=torch.int64),
                torch.zeros(1, padding_len - len(i[0]), dtype=torch.int64)
            ],
            dim=1
        ) if len(i[0]) < padding_len else \
        torch.ones(1, padding_len, dtype=torch.int64)
        for i in sample_list
    ]

    sent = torch.cat(sent, dim=0)
    mask = torch.cat(mask, dim=0)

    label = torch.LongTensor([[i[1]] for i in sample_list])
    one_hot = torch.zeros(batch_size, 33).scatter_(1, label, 1)

    return sent, mask, one_hot, start_index

class Summary:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.tp = [0] * n_classes
        self.p = [0] * n_classes
        self.t = [0] * n_classes
        self.count = 0

    def clear(self):
        self.tp = [0] * self.n_classes
        self.p = [0] * self.n_classes
        self.t = [0] * self.n_classes
        self.count = 0

    def update(self, pred, gold):
        assert pred.numel() == gold.numel()
        self.count += pred.numel()
        for i in range(len(pred)):
            self.p[pred[i].item()] += 1
            self.t[gold[i].item()] += 1
            if pred[i] == gold[i]:
                self.tp[gold[i].item()] += 1

    def get_result(self, eva_tags=None):
        if eva_tags:
            self.tp = [self.tp[i] for i in range(self.n_classes) if i in eva_tags]
            self.p = [self.p[i] for i in range(self.n_classes) if i in eva_tags]
            self.t = [self.t[i] for i in range(self.n_classes) if i in eva_tags]
            nc = len(eva_tags)
        else:
            nc = self.n_classes

        macro_p = sum(self.tp) / (sum(self.p) + 1e-8)
        macro_q = sum(self.tp) / (sum(self.t) + 1e-8)
        macro_f1 = 2. * macro_p * macro_q / (macro_p + macro_q + 1e-8)

        micro_p = [self.tp[i] / (self.p[i] + 1e-8) for i in range(nc)]
        micro_q = [self.tp[i] / (self.t[i] + 1e-8) for i in range(nc)]
        micro_f1 = [2. * micro_p[i] * micro_q[i] / \
            (micro_p[i] + micro_q[i] + 1e-8) for i in range(nc)]

        """print(macro_f1)
        print(' '.join(['%.4f'%i for i in micro_p]))
        print(' '.join(['%.4f'%i for i in micro_q]))
        print(' '.join(['%.4f'%i for i in micro_f1]))"""

        return macro_f1, sum(micro_f1) / len(micro_f1), sum(self.tp) / self.count

def test_model(args, model, test_data, report_tags=None):
    model.eval()
    v_count, v_corr = 0, 0
    summary = Summary(33)
    summary.clear()
    for v_batch in tqdm(data_gen(args, test_data, is_shuffle=True)):
        sent  = v_batch[0].to(args['device'])
        mask  = v_batch[1].to(args['device'])
        label = v_batch[2].to(args['device'])

        if args['model'] == 'layer' or args['model'] == 'layersum':
            y_hat, _ = model(sent, mask)
        else:
            y_hat = model(sent, mask)
        y_hat = torch.argmax(y_hat, dim=1)
        label = torch.argmax(label, dim=1)
        summary.update(y_hat, label)
        result = (y_hat == label)
        
        v_corr = v_corr + torch.sum(result.float())
        v_count = v_count + sent.shape[0]

    acc = v_corr / v_count
    args['logger'].info('acc=%d/%d=%.4f' % (v_corr, v_count, acc))

    if report_tags == None:
        args['logger'].info('tp: ' + '|'.join(['%4d' % i for i in summary.tp]))
        args['logger'].info(' p: ' + '|'.join(['%4d' % i for i in summary.p]))
        args['logger'].info(' t: ' + '|'.join(['%4d' % i for i in summary.t]))
    elif len(report_tags) > 0:
        args['logger'].info('tp: ' + '|'.join(['%4d' % s for i,s in enumerate(summary.tp) if i in report_tags]))
        args['logger'].info(' p: ' + '|'.join(['%4d' % s for i,s in enumerate(summary.p)  if i in report_tags]))
        args['logger'].info(' t: ' + '|'.join(['%4d' % s for i,s in enumerate(summary.t)  if i in report_tags]))

    return acc
