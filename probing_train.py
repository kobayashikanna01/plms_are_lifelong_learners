import os
import copy
import time
import pynvml
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
import trainers

def main(args, logger):
    if not os.path.exists('probing_save'):
        os.system('mkdir probing_save')


    tok_name_tmp = args['tok_name'].replace('//', '_').replace('-', '_')
    agnews_train  = utils.load_data(
        'data/agnews.train.%s.json' % (tok_name_tmp),  'agnews')
    amazon_train  = utils.load_data(
        'data/amazon.train.%s.json' % (tok_name_tmp),  'amazon')
    dbpedia_train = utils.load_data(
        'data/dbpedia.train.%s.json' % (tok_name_tmp), 'dbpedia')
    yahoo_train   = utils.load_data(
        'data/yahoo.train.%s.json' % (tok_name_tmp),   'yahoo')
    yelp_train    = utils.load_data(
        'data/yelp.train.%s.json' % (tok_name_tmp),    'yelp')
    
    agnews_test  = utils.load_data(
        'data/agnews.test.%s.json' % (tok_name_tmp),   'agnews')
    amazon_test  = utils.load_data(
        'data/amazon.test.%s.json' % (tok_name_tmp),   'amazon')
    dbpedia_test = utils.load_data(
        'data/dbpedia.test.%s.json' % (tok_name_tmp),  'dbpedia')
    yahoo_test   = utils.load_data(
        'data/yahoo.test.%s.json' % (tok_name_tmp),    'yahoo')
    yelp_test    = utils.load_data(
        'data/yelp.test.%s.json' % (tok_name_tmp),     'yelp')

    ori_train_data = [
        agnews_train,
        amazon_train,
        dbpedia_train,
        yahoo_train,
        yelp_train
    ]

    ori_test_data = [
        agnews_test,
        amazon_test,
        dbpedia_test,
        yahoo_test,
        yelp_test
    ]

    dsname = [
        'agnews',
        'amazon',
        'dbpedia',
        'yahoo',
        'yelp'
    ]

    args['logger'].info('Load %d items from training dataset, %d items from test dataset.' \
        % (sum([len(i) for i in ori_train_data]), sum([len(i) for i in ori_test_data])))

    #train_data = [j for i in train_data for j in i]
    #test_data  = [j for i in test_data  for j in i]

    model = models.MultiClassModel(args)
    #layer_optimizer = None

    ck_name = list(os.walk('./save'))[0][2]
    ck_name = [i for i in ck_name if args['train_time'] in i]
    #ck_name = json.load(open('checkpoint.json'))

    if args['gn'] >= 0:
        pynvml.nvmlInit()
        while True:
            handle = pynvml.nvmlDeviceGetHandleByIndex(args['gn'])
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if meminfo.free > 1.8e10:
                break
            else:
                print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), meminfo.free)
                time.sleep(120)

    for dn in [0, 1, 2, 3, 4]: #range(5):
        for ck in ck_name:
            finished_list = list(os.walk('probing_save'))[0][2]
            if '%s_best_%s' % (dsname[dn], ck) in finished_list:
                continue

            train_data = ori_train_data[dn]
            test_data  = ori_test_data[dn]

            args['logger'].info('==== Dataset: %s, Model: %s ====' % (dsname[dn], ck))
            model.load_state_dict(torch.load('./save/' + ck))
            for p in model.bert.parameters():
                p.requires_grad = False
            model = model.to(args['device'])
            parameters = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = AdamW(parameters, lr=args['learning_rate'])

            args['bs'] = args['batch_size']
            args['batch_size'] = args['testbs']
            best_acc = utils.test_model(args, model, test_data, report_tags=[])
            torch.save(model.state_dict(), './probing_save/%s_best_%s' % (dsname[dn], ck))
            args['batch_size'] = args['bs']
            if best_acc > 0.:
                ep_silence = -1
            else:
                ep_silence = -1 #2

            for ep in range(1, args['epoch'] + 1):
                args['logger'].info('------  Epoch %2d  ------' % ep)
                count = 0
                loss_sum = 0.
                for batch in utils.data_gen(args, train_data, is_shuffle=True):
                    sent  = batch[0].to(args['device'])
                    mask  = batch[1].to(args['device'])
                    label = batch[2].to(args['device'])
                    y_hat = model(sent, mask)
                    loss = trainers.get_cross_entropy(y_hat, label)
                    loss = -torch.sum(loss) / sent.shape[0]
                        
                    print(loss, end='\r')
                    loss_sum += loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (loss == loss) == False:
                        break

                    count += 1
                    if ep > ep_silence and count % args['eval_step'] == 0:
                        if not test_data:
                            continue
                        if len(test_data) < 1:
                            continue
                        
                        args['logger'].info("step %d: loss=%f" % (count, loss_sum / args['eval_step']))
                        loss_sum = 0.0
                        
                        args['bs'] = args['batch_size']
                        args['batch_size'] = args['testbs']
                        acc = utils.test_model(args, model, test_data, report_tags=[])
                        args['batch_size'] = args['bs']
                        if acc > best_acc:
                            best_acc = acc
                            torch.save(model.state_dict(), './probing_save/%s_best_%s' % (dsname[dn], ck))
                            args['logger'].info("New best model.")

                        model.train()

                args['logger'].info("Test after training on the whole epoch:")
                args['bs'] = args['batch_size']
                args['batch_size'] = args['testbs']
                acc = utils.test_model(args, model, test_data, report_tags=[])
                args['batch_size'] = args['bs']
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), './probing_save/%s_best_%s' % (dsname[dn], ck))
                    args['logger'].info("New best model.")
                model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plm_name', default='roberta-base', type=str)
    parser.add_argument('--tok_name', default='', type=str)
    parser.add_argument('--pad_token', default=1, type=int)
    parser.add_argument('--plm_type', default='roberta', type=str)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=1023, type=int)
    parser.add_argument('--padding_len', default=128, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--eval_step', default=360, type=int)
    parser.add_argument('--train_time', default='', type=str)
    parser.add_argument('--trainer', default='multiclass', type=str)
    parser.add_argument('--model', default='multiclass', type=str)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--order', default=0, type=int)
    parser.add_argument('--memory_save', default=1150, type=int)
    #parser.add_argument('--reg_rate', default=1e-6, type=float)
    #parser.add_argument('--start_layer', default=12, type=int)
    #parser.add_argument('--end_layer', default=12, type=int)
    parser.add_argument('--testbs', default=128, type=int)
    parser.add_argument('--gn', default=-1, type=int)

    args = parser.parse_args()
    if args.device != 'cuda':
        args.device = 'cpu'
    args = args.__dict__

    assert args['train_time'] != ''

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logger = logging.getLogger()
    fh = logging.FileHandler('./logs/SEPFinetune-%s.log' % now_time, encoding='utf-8')
    args['now_time'] = now_time
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(10)
    logger.info('Configuration:\n' + json.dumps(args, indent=2))
    args['logger'] = logger

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(args['seed'])

    main(args, logger)

