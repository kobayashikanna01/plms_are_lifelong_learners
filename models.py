import torch
import torch.nn as nn
import numpy as np
import random
#from transformers import BertModel, BertTokenizer, BertConfig
from transformers import AutoModel, AutoConfig

class MultiClassModel(nn.Module):
    def __init__(self, args):
        super(MultiClassModel, self).__init__()
        self.bert = AutoModel.from_pretrained(args['plm_name'])
        self.classifier = nn.Linear(args['hidden_size'], 33, bias=False)

        """nn.init.normal_(
            self.classifier.weight.data, 
            mean=0, 
            std=np.sqrt(1. / 768)
        )"""
        self.plm = args['plm_type']

    def forward(self, sent, mask, ret_layer_fea=False):
        if ret_layer_fea:
            feature = self.bert(
                input_ids=sent,
                attention_mask=mask,
                output_hidden_states=True
            )
            score = self.classifier(feature.pooler_output)
            return score, [i[:, 0, :] for i in feature.hidden_states]

        feature = self.bert(
            input_ids=sent, 
            attention_mask=mask
        )

        if self.plm == 'xlnet' or self.plm == 'gpt2':
            bs, sl = sent.shape
            seq_len = torch.sum(mask.detach(), dim=1).cpu().tolist()
            feature = torch.cat(
                [feature.last_hidden_state[ib, pos-1:pos, :] for ib, pos in enumerate(seq_len)],
                dim=0
            )
            score = self.classifier(feature)
            return score
        
        if 'pooler_output' in feature.keys():
            feature = feature.pooler_output
        else:
            feature = feature.last_hidden_state[:, 0, :]

        #score = self.classifier(feature[:,0,:])
        score = self.classifier(feature)

        return score

class NoPretrainedModel(nn.Module):
    def __init__(self, args):
        super(NoPretrainedModel, self).__init__()
        cfg = AutoConfig.from_pretrained(args['plm_name'])
        self.bert = AutoModel.from_config(cfg)
        self.classifier = nn.Linear(args['hidden_size'], 33, bias=False)

        """nn.init.normal_(
            self.classifier.weight.data, 
            mean=0, 
            std=np.sqrt(1. / 768)
        )"""
        self.plm = args['plm_prefix']

    def forward(self, sent, mask, ret_layer_fea=False):
        if ret_layer_fea:
            feature = self.bert(
                input_ids=sent,
                attention_mask=mask,
                output_hidden_states=True
            )
            score = self.classifier(feature.pooler_output)
            return score, [i[:, 0, :] for i in feature.hidden_states]

        feature = self.bert(
            input_ids=sent, 
            attention_mask=mask
        )

        if self.plm == 'xlnet' or self.plm == 'gpt2':
            bs, sl = sent.shape
            seq_len = torch.sum(mask.detach(), dim=1).cpu().tolist()
            feature = torch.cat(
                [feature.last_hidden_state[ib, pos-1:pos, :] for ib, pos in enumerate(seq_len)],
                dim=0
            )
            score = self.classifier(feature)
            return score
        
        if 'pooler_output' in feature.keys():
            feature = feature.pooler_output
        else:
            feature = feature.last_hidden_state[:, 0, :]

        #score = self.classifier(feature[:,0,:])
        score = self.classifier(feature)

        return score