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

import utils
import models

# 0 - AGNews
# 1 - Amazon
# 2 - DBPedia
# 3 - Yahoo
# 4 - Yelp
order_list = [
    [4, 0, 2, 1, 3],
    [2, 3, 0, 1, 4],
    [4, 3, 1, 2, 0],
    [0, 4, 1, 3, 2],
    [2, 0, 1, 3, 4],
    [0, 2, 1, 3, 4]
]
dataset_name = ['AGNews', 'Amazon', 'DBPedia', 'Yahoo', 'Yelp']

report_tags = [
    [0, 1, 2, 3],
    [4, 5, 6, 7, 8],
    [i for i in range(9, 23)],
    [i for i in range(23, 33)],
    [4, 5, 6, 7, 8]
]


def get_cross_entropy(ori_score, gold_label):
    sm_score = torch.softmax(ori_score, dim=1)
    return gold_label * torch.log(sm_score) + (1. - gold_label) * torch.log(1. - sm_score)

def get_memory_and_prototype(args, model, data_list):
    s_l, e_l = args['start_layer'], args['end_layer'] + 1

    model.eval()
    proto_list = []
    for batch in utils.data_gen(args, data_list, is_shuffle=False):
        sent  = batch[0].to(args['device'])
        mask  = batch[1].to(args['device'])
        _, hidden_states = model(sent, mask, ret_layer_fea=True)
        hidden_states = [i.detach().to('cpu') for i in hidden_states]

        for i in range(sent.shape[0]):
            item = torch.cat([j[i:i+1, :] for j in hidden_states[s_l:e_l]], dim=0)
            proto_list.append(item)

    ret_list = []
    for i in range(len(proto_list)):
        ret_list.append(data_list[i] + (proto_list[i].reshape(1, -1, 768), ) )

    return ret_list


def training_by_epoch(args, model, optimizer, train_data, test_data, layer_optimizer=None, rep=1, ds_reg=-1.):
    count = 0
    loss_sum = 0.
    for batch in utils.data_gen(args, train_data, is_shuffle=True):
        sent  = batch[0].to(args['device'])
        mask  = batch[1].to(args['device'])
        label = batch[2].to(args['device'])
        dslbl = batch[-1].to(args['device'])
        
        if args['model'] == 'layer':
            y_hat, layer_scores = model(sent, mask)
            loss = get_cross_entropy(y_hat, label)
            loss = -torch.sum(loss) / sent.shape[0]
            for i in range(len(layer_scores)):
                loss_i = get_cross_entropy(layer_scores[i], label)
                loss_i = -torch.sum(loss_i) / sent.shape[0]
                layer_optimizer[i].zero_grad()
                loss_i.backward(retain_graph=True)
                layer_optimizer[i].step()

        elif args['model'] == 'layersum':
            y_hat, layer_scores = model(sent, mask)
            loss = get_cross_entropy(y_hat, label)
            loss = -torch.sum(loss) / sent.shape[0]
            for i in range(4, len(layer_scores)):
                loss_i = get_cross_entropy(layer_scores[i], label)
                loss_i = -0.05 * torch.sum(loss_i) / sent.shape[0]
                loss += loss_i

        else:
            y_hat = model(sent, mask)
            if ds_reg > 0:
                ds_y = [y_hat[:, 0:4], y_hat[:, 4:9], y_hat[:, 9:23], y_hat[:, 23:33]]
                ds_y = [torch.max(y_item, dim=1).values for y_item in ds_y]
                ds_y = torch.cat([y_item.reshape(-1, 1) for y_item in ds_y], dim=1)
                loss = -torch.sum(get_cross_entropy(y_hat, label)) - \
                        torch.sum(get_cross_entropy(ds_y, dslbl)) * ds_reg
            else:
                loss = -torch.sum(get_cross_entropy(y_hat, label))
            loss = loss / sent.shape[0] * rep
            
        print(loss, end='\r')
        loss_sum += loss

        optimizer.zero_grad()
        loss.backward()
        flag = False
        for p in model.parameters():
            if type(p.grad) != type(None):
                if p.grad[p.grad != p.grad].size(0):
                    flag = True
                    break
        if flag:
            optimizer.zero_grad()
        else:
            optimizer.step()

        count += 1
        if 'steps' in args:
            args['steps'] += int(sent.shape[0])
            if args['steps'] >= args['ck_counter']:
                args['ck_counter'] += 5000
                #args['ck_counter'] += 5000
                torch.save(model.state_dict(), './save/%s-%s-%d-%d.pt' % (args['plm_type'], args['now_time'], args['steps'], count))
        if count % args['eval_step'] == 0:
            if not test_data:
                continue
            if len(test_data) < 1:
                continue
            
            args['logger'].info("step %d: loss=%f" % (count, loss_sum / args['eval_step']))
            loss_sum = 0.0
            
            acc = utils.test_model(args, model, test_data)
            model.train()

    return model


def multiclass_train(args, model, optimizer, train_data, test_data, layer_optimizer=None):
    train_data = [j for i in train_data for j in i]
    test_data  = [j for i in test_data  for j in i]

    args['steps'] = 0
    args['ck_counter'] = 5000

    for ep in range(1, args['epoch'] + 1):
        args['logger'].info('------  Epoch %2d  ------' % ep)
        model = training_by_epoch(args, model, optimizer, train_data, test_data, layer_optimizer)

        args['logger'].info("Test after training on the whole epoch:")
        acc = utils.test_model(args, model, test_data)
        model.train()

        torch.save(model.state_dict(), './save/MultiClass-%s-%d.pt' % (args['now_time'], ep))


def sequential_train(args, model, optimizer, train_data, test_data, layer_optimizer=None):
    assert args['order'] >= 0 and args['order'] < 5

    order = order_list[args['order']]
    test_data  = [j for i in test_data  for j in i]

    args['steps'] = 0
    args['ck_counter'] = 5000

    for i in range(5):
        now_train_data = train_data[order[i]]

        for ep in range(1, args['epoch'] + 1):
            args['logger'].info('---- Epoch %2d on %s ----' % (ep, dataset_name[order[i]]))
            model = training_by_epoch(args, model, optimizer, now_train_data, test_data, layer_optimizer)

            args['logger'].info("Test after training on the whole epoch:")
            acc = utils.test_model(args, model, test_data)
            model.train()

            torch.save(model.state_dict(), './save/%s-order%d-EP%d-Final.pt' % (dataset_name[order[i]], args['order'], ep))

def sequential_train_2(args, model, optimizer, train_data, test_data, layer_optimizer=None):
    order_ori = [0, 1, 2, 3, 4]
    order = [args['order'] // 4]
    order_ori = [i for i in order_ori if not i == order[0]]
    order = order + [order_ori[args['order'] % 4]]

    print(order)

    #test_data  = [j for i in test_data  for j in i]
    test_data = test_data[order[0]] + test_data[order[1]]

    args['steps'] = 0
    args['ck_counter'] = 10000

    for i in range(2):
        now_train_data = train_data[order[i]]

        for ep in range(1, args['epoch'] + 1):
            args['logger'].info('---- Epoch %2d on %s ----' % (ep, dataset_name[order[i]]))
            model = training_by_epoch(args, model, optimizer, now_train_data, test_data, layer_optimizer)

            args['logger'].info("Test after training on the whole epoch:")
            acc = utils.test_model(args, model, test_data)
            model.train()

            torch.save(model.state_dict(), './save/%s-order%d-EP%d-Final.pt' % (dataset_name[order[i]], args['order'], ep))


def original_replay_train(args, model, optimizer, train_data, test_data, layer_optimizer=None):
    assert args['order'] >= 0 and args['order'] < 5
    order = order_list[args['order']]
    args['steps'] = 0
    args['ck_counter'] = 5000

    seen_test_data = []
    memory_list = []
    for i in range(5):
        now_train_data = train_data[order[i]]
        seen_test_data += test_data[order[i]]

        for ep in range(1, args['epoch'] + 1):
            random.shuffle(now_train_data)
            args['logger'].info('---- Epoch %2d on %s ----' % (ep, dataset_name[order[i]]))
            print(len(now_train_data))   ###
            train_data_split = list(range(0, len(now_train_data), args['rep_itv'])) + [len(now_train_data)]
            for ts_idx in range(len(train_data_split) - 1):
                start_idx, end_idx = train_data_split[ts_idx], train_data_split[ts_idx + 1]
                model = training_by_epoch(args, model, optimizer, now_train_data[start_idx:end_idx], None, ds_reg=-1.)
                if len(memory_list):
                    random.shuffle(memory_list)
                    model = training_by_epoch(args, model, optimizer, memory_list[:args['rep_num']], None, ds_reg=-1)
                    args['steps'] = args['steps'] - args['rep_num']
                    torch.save(model.state_dict(), './save/%s-%s-%d-rep.pt' % (args['plm_type'], args['now_time'], args['steps']))

            acc = utils.test_model(args, model, seen_test_data)
            model.train()


        memory_list += now_train_data[:args['memory_save']]