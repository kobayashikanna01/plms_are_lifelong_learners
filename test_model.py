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
import trainers

def add_grad(dic, nm, gten):
    if not nm in dic:
        dic[nm] = torch.zeros(gten.shape)

    dic[nm] += gten.cpu()
    return dic

def main(args, logger):
    try:
        ck_files = json.load(open('checkpoint.json'))
    except:
        ck_files = list(os.walk('./probing_save/'))[0][2]
        ck_files = [i for i in ck_files if args['train_time'] in ck_files]

    if not os.path.exists('logs'):
        os.system('mkdir logs')

    if not os.path.exists('logs/F1_result'):
        os.system('mkdir logs/F1_result')


    tok_name_tmp = args['tok_name'].replace('//', '_').replace('-', '_')
    
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

    test_data = [
        agnews_test,
        amazon_test,
        dbpedia_test,
        yahoo_test,
        yelp_test
    ]

    dataname = [
        'agnews',
        'amazon',
        'dbpedia',
        'yahoo',
        'yelp'
    ]

    report_tags = [
        [0, 1, 2, 3],
        [4, 5, 6, 7, 8],
        [i for i in range(9, 23)],
        [i for i in range(23, 33)],
        [4, 5, 6, 7, 8]
    ]

    split_pos = [
        [0, 4],
        [4, 9],
        [9, 23],
        [23, 33],
        [4, 9]
    ]

    #train_data = train_data[args['order']]
    #test_data  = test_data[args['order']]

    args['logger'].info('Load %d items from training dataset, %d items from test dataset.' \
        % (sum([len(i) for i in train_data]), sum([len(i) for i in test_data])))

    try:
        f = open('./logs/F1_result/' + args['prefix'] + '.json')
        results = json.load(f)
    except:
        results = {}
        json.dump(results, open('./logs/F1_result/Results-' + args['now_time'] + '.json', 'w'), indent=2)

    data_rev_map = {
        'agnews' : 0,
        'amazon' : 1,
        'dbpedia' : 2,
        'yahoo' : 3,
        'yelp' : 4,
    }

    model = models.MultiClassModel(args)
    for ck in ck_files:
        ck = ck.split('_')
        dn = data_rev_map[ck[0]]
        ck = '_'.join(ck[1:])
        args['logger'].info('---- %s : %s ----' % (dataname[dn], ck))
        if ck in results and dataname[dn] in results[ck]:
            continue
        else:
            #try:
            pt_1 = torch.load('./save/%s' % (ck.replace('best_', '')))
            pt_2 = torch.load('./probing_save/%s_%s' % (dataname[dn], ck))
            for p_name in pt_2:
                pt_1[p_name] = pt_2[p_name]
            model.load_state_dict(pt_1)
            #except:
            #    continue
            model = model.to(args['device'])
            model.eval()

            if not ck in results:
                results[ck] = {}
            
            score_sum = utils.test_model(args, model, test_data[dn], report_tags[dn])
            score_sum = score_sum.item()

            results[ck][dataname[dn]] = score_sum

            json.dump(results, open('./logs/F1_result/Results-' + args['now_time'] + '.json', 'w'), indent=2)

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
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--eval_step', default=2000, type=int)
    parser.add_argument('--train_time', default='', type=str)
    parser.add_argument('--trainer', default='multiclass', type=str)
    parser.add_argument('--model', default='multiclass', type=str)
    parser.add_argument('--epoch', default=6, type=int)
    parser.add_argument('--order', default=1, type=int)
    parser.add_argument('--memory_save', default=1150, type=int)

    args = parser.parse_args()
    if args.device != 'cuda':
        args.device = 'cpu'
    args = args.__dict__

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logger = logging.getLogger()
    fh = logging.FileHandler('./logs/F1_result/TestResult-%s.log' % now_time, encoding='utf-8')
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

