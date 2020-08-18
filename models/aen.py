# -*- coding: utf-8 -*-
# file: aen.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention, NoQueryAttention
from layers.point_wise_feed_forward import PositionwiseFeedForward
import torch
import torch.nn as nn
import torch.nn.functional as F


# CrossEntropyLoss for Label Smoothing Regularization
class CrossEntropyLoss_LSR(nn.Module):
    def __init__(self, device, para_LSR=0.2):
        super(CrossEntropyLoss_LSR, self).__init__()
        self.para_LSR = para_LSR
        self.device = device
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def _toOneHot_smooth(self, label, batchsize, classes):
        prob = self.para_LSR * 1.0 / classes
        one_hot_label = torch.zeros(batchsize, classes) + prob
        for i in range(batchsize):
            index = label[i]
            one_hot_label[i, index] += (1.0 - self.para_LSR)
        return one_hot_label

    def forward(self, pre, label, size_average=True):
        b, c = pre.size()
        one_hot_label = self._toOneHot_smooth(label, b, c).to(self.device)
        loss = torch.sum(-one_hot_label * self.logSoftmax(pre), dim=1)
        if size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class AEN_GloVe(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(AEN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()

        self.attn_k = Attention(opt.embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_q = Attention(opt.embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)

        self.dense = nn.Linear(opt.hidden_dim*3, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, target_indices = inputs[0], inputs[1]
        context_len = torch.sum(text_raw_indices != 0, dim=-1)
        target_len = torch.sum(target_indices != 0, dim=-1)
        context = self.embed(text_raw_indices)
        context = self.squeeze_embedding(context, context_len)
        target = self.embed(target_indices)
        target = self.squeeze_embedding(target, target_len)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)

        s1, _ = self.attn_s1(hc, ht)

        context_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)

        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))

        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        out = self.dense(x)
        return out
    def get_Optimizer(self):
        if self.optimizer != None:
            return self.optimizer
        return None

    def set_Optimizer(self, newoptimizer):
        self.optimizer = newoptimizer


class AEN_BERT(nn.Module): #Attentional Encoder Network for Targeted Sentiment Classiﬁcation
    def __init__(self, bert, opt):
        super(AEN_BERT, self).__init__()
        #print(" 1 In AEN_BERT  ")
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)
        #print(" 2 In AEN_BERT  ")
        self.attn_k = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_q = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        #print(" 3 In AEN_BERT  ")
        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.hat = False
        self.last = torch.nn.ModuleList()
        for t in range(self.opt.taskcla):
            self.last.append(nn.Linear(opt.hidden_dim*3, opt.polarities_dim))

    def get_Optimizer(self):
        if self.optimizer != None:
            return self.optimizer
        return None

    def set_Optimizer(self, newoptimizer):
        self.optimizer = newoptimizer

    def forward(self, t, inputs, s):
        context, target = inputs[0], inputs[1]
        context_len = torch.sum(context != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1)
        context = self.squeeze_embedding(context, context_len)
        # context, _ = self.bert(context, output_all_encoded_layers=False)

        #Bert Trasformer
        context, _,  _ = self.bert(context)

        #Bert pretrained (Old version)
        #context, _ = self.bert(context, output_all_encoded_layers=False)

        target = self.squeeze_embedding(target, target_len)
        # , output_all_encoded_layers = False

        # Bert Trasformer
        target, _,  _ = self.bert(target)

        # Bert pretrained (Old version)
        #target, _ = self.bert(target, output_all_encoded_layers=False)

        target = self.dropout(target)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)

        s1, _ = self.attn_s1(hc, ht)

        context_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)

        #Verify, because there are problems whit Tensor type Longtensor and more down with FloatTensor
        #context_len = context_len.clone().detach().to(self.opt.device)

        target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)

        #Verify, because there are problems whit Tensor type Longtensor and more down with FloatTensor
        #target_len = target_len.clone().detach().to(self.opt.device)

        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))

        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        y = []
        for i, _ in enumerate(range(self.opt.taskcla)):
            y.append(self.last[i](x))
        return y

    def get_bert_model_parameters(self):
        variable_name = ["attn_k", "attn_q", "ffn_c", "ffn_t","attn_s1","last"]
        modelVariables = []

        for i, (name, var) in enumerate(self.named_parameters()):
            for iname in variable_name:
                if name.find(iname) != -1:
                    modelVariables.append((name, var))
                    break

        return modelVariables