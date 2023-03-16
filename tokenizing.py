from transformers import AutoTokenizer
import argparse
import os

#tok_dict = {
#    # 'roberta-base' : 'roberta',
#    # 'google/electra-base-discriminator' : 'electra',
#    # 'facebook/bart-base' : 'bart',
#    # 'bert-base-uncased' : 'bert',
#    # 'xlnet-base-cased' : 'xlnet'
#    # 'gpt2' : 'gpt2'
#    'microsoft/deberta-v3-base' : 'deberta'
#}

import os, json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--max_token_num', type=int, default=128)

    args = parser.parse_args()

    data = [i for i in list(os.walk('data'))[0][2] if 'raw.json' == i[-8:]]
    data = {i:json.load(open( os.path.join(args.data_dir, i) )) for i in data}

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    for d in data:
        print(args.tokenizer, d)
        sents = [' '.join(i[1].split()) for i in data[d]]
        token_ids = tokenizer(sents, max_length=args.max_token_num)['input_ids']
        # token_ids = tokenizer(sents)['input_ids']
        assert len(token_ids) == len(data[d])
        tmp_data = [[data[d][i][0], token_ids[i]] for i in range(len(data[d]))]

        tok_name = args.tokenizer.replace('//', '_').replace('-', '_')
        json.dump(tmp_data, open('data/' + d.replace('raw.json', tok_name + '.json'), 'w'))

