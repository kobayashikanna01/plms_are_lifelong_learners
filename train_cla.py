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

def main(args, logger):
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

    train_data = [
        agnews_train,
        amazon_train,
        dbpedia_train,
        yahoo_train,
        yelp_train
    ]

    test_data = [
        agnews_test,
        amazon_test,
        dbpedia_test,
        yahoo_test,
        yelp_test
    ]

    args['logger'].info('Load %d items from training dataset, %d items from test dataset.' \
        % (sum([len(i) for i in train_data]), sum([len(i) for i in test_data])))

    trainer_select = {
        'multiclass' : trainers.multiclass_train,
        'sequential' : trainers.sequential_train,
        'replay'     : trainers.original_replay_train,
    }
    
    if args['model'] == 'multiclass':
        model = models.MultiClassModel(args)
        layer_optimizer = None
    elif args['model'] == 'noptr':
        model = models.NoPretrainedModel(args)
        layer_optimizer = None
    '''
    elif args['model'] == 'layer':
        model = models.LayerSymModel(args)
        layer_optimizer = []
        for i in range(12):
            l_param = \
                [p for p in model.layer_pooler[i].parameters() if p.requires_grad] + \
                [p for p in model.layer_classifier[i].parameters() if p.requires_grad]

            l_optim = torch.optim.Adadelta(l_param, lr=1e-5)
            layer_optimizer.append(l_optim)
    elif args['model'] == 'layersum':
        model = models.LayerSymModel(args)
        layer_optimizer = None'''

    model = model.to(args['device'])
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(parameters, lr=args['learning_rate'])

    trainer = trainer_select[args['trainer']]
    trainer(args, model, optimizer, train_data, test_data, layer_optimizer)

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
    parser.add_argument('--learning_rate', default=1.5e-5, type=float)
    parser.add_argument('--eval_step', default=500000, type=int)
    parser.add_argument('--load_path', default='', type=str)
    parser.add_argument('--trainer', default='', type=str)
    parser.add_argument('--model', default='multiclass', type=str)
    parser.add_argument('--epoch', default=2, type=int)
    parser.add_argument('--order', default=0, type=int)
    parser.add_argument('--memory_save', default=1150, type=int)
    #parser.add_argument('--reg_rate', default=1e-6, type=float)
    #parser.add_argument('--start_layer', default=12, type=int)
    #parser.add_argument('--end_layer', default=12, type=int)
    parser.add_argument('--rep_itv', default=10000, type=int)
    parser.add_argument('--rep_num', default=100, type=int)
    #parser.add_argument('--cla_num', default=50, type=int)
    #parser.add_argument('--prox_rate', default=1e-4, type=float)
    #parser.add_argument('--cla_lr', default=5e-5, type=float)
    #parser.add_argument('--rep_lr', default=3e-5, type=float)

    args = parser.parse_args()
    if args.device != 'cuda':
        args.device = 'cpu'
    args = args.__dict__
    if args['tok_name'] == '':
        args['tok_name'] = args['plm_name']

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logger = logging.getLogger()
    fh = logging.FileHandler('./logs/TrainModel-%s.log' % now_time, encoding='utf-8')
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

