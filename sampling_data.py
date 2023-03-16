import pandas as pd
import numpy as np
import argparse
import random
import json

from tqdm import tqdm

def sample_from_dframe(data, n_sample, label_pos, text_pos):
    sample_id = list(range(len(data)))
    random.shuffle(sample_id)

    ret_list = []
    for item_id in tqdm(sample_id[:n_sample]):
        ret_list.append(
            [
                int(data.iloc[item_id, label_pos]),
                ' '.join(str(data.iloc[item_id, tid]) for tid in text_pos)
            ]
        )

    label_list = list(set([i[0] for i in ret_list]))
    label_list.sort()
    label_map = {j:i for i,j in enumerate(label_list)}

    for i in range(len(ret_list)):
        ret_list[i][0] = label_map[ ret_list[i][0] ]

    return ret_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    print('Random seed:', args.seed)

    if args.seed != None:
        np.random.seed(seed)
        random.seed(seed)
    
    print('Sampling from AGNews...')
    agnews_train = pd.read_csv('data/ag_news_csv/train.csv', header=None)
    agnews_train = sample_from_dframe(
        agnews_train,
        115000,
        0,
        [1, 2]
    )
    agnews_test = pd.read_csv('data/ag_news_csv/test.csv', header=None)
    agnews_test = sample_from_dframe(
        agnews_test,
        7600,
        0,
        [1, 2]
    )

    print('Sampling from Amazon...')
    amazon_train = pd.read_csv('data/amazon_review_full_csv/train.csv', header=None)
    amazon_train = sample_from_dframe(
        amazon_train,
        115000,
        0,
        [1, 2]
    )
    amazon_test = pd.read_csv('data/amazon_review_full_csv/test.csv', header=None)
    amazon_test = sample_from_dframe(
        amazon_test,
        7600,
        0,
        [1, 2]
    )

    print('Sampling from DBPedia...')
    dbpedia_train = pd.read_csv('data/dbpedia_csv/train.csv', header=None)
    dbpedia_train = sample_from_dframe(
        dbpedia_train,
        115000,
        0,
        [2]
    )
    dbpedia_test = pd.read_csv('data/dbpedia_csv/test.csv', header=None)
    dbpedia_test = sample_from_dframe(
        dbpedia_test,
        7600,
        0,
        [2]
    )

    print('Sampling from Yahoo...')
    yahoo_train = pd.read_csv('data/yahoo_answers_csv/train.csv', header=None)
    yahoo_train = sample_from_dframe(
        yahoo_train,
        115000,
        0,
        [1, 2]
    )
    yahoo_test = pd.read_csv('data/yahoo_answers_csv/test.csv', header=None)
    yahoo_test = sample_from_dframe(
        yahoo_test,
        7600,
        0,
        [1, 2]
    )

    print('Sampling from Yelp...')
    yelp_train = pd.read_csv('data/yelp_review_full_csv/train.csv', header=None)
    yelp_train = sample_from_dframe(
        yelp_train,
        115000,
        0,
        [1]
    )
    yelp_test = pd.read_csv('data/yelp_review_full_csv/test.csv', header=None)
    yelp_test = sample_from_dframe(
        yelp_test,
        7600,
        0,
        [1]
    )
    

    dataset_name = ['agnews', 'amazon', 'dbpedia', 'yahoo', 'yelp']
    split_name = ['train', 'test']

    for ds in dataset_name:
        for s in split_name:
            json.dump(eval('%s_%s' % (ds, s)), open('data/%s.%s.raw.json' % (ds, s), 'w'))

