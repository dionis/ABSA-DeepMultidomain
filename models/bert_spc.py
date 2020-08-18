# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from layers.squeeze_embedding import SqueezeEmbedding


class BERT_SPC(nn.Module): #Basic BERT-based model
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        # self.squeeze_embedding = SqueezeEmbedding()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.opt = opt
        #self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.hat = False
        self.last = torch.nn.ModuleList()
        for t in range(self.opt.taskcla):
            self.last.append(nn.Linear(opt.bert_dim, opt.polarities_dim))

    def forward(self, t, inputs, s):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        # text_bert_len = torch.sum(text_bert_indices != 0, dim=-1)
        # text_bert_indices = self.squeeze_embedding(text_bert_indices, text_bert_len)
        # bert_segments_ids = self.squeeze_embedding(bert_segments_ids, text_bert_len)
        #, output_all_encoded_layers=False
        _, pooled_output,_ = self.bert(text_bert_indices, bert_segments_ids)
        pooled_output = self.dropout(pooled_output)

        y = []
        for i, _ in enumerate(range(self.opt.taskcla)):
            y.append(self.last[i](pooled_output))
        return y

    def get_bert_model_parameters(self):
        variable_name = ["last"]
        modelVariables = []

        for i, (name, var) in enumerate(self.named_parameters()):
              for iname in variable_name:
                  if name.find(iname) != -1:
                      modelVariables.append((name,var))
                      break

        return modelVariables

    def get_Optimizer(self):
        if self.optimizer != None:
            return self.optimizer
        return None

    def set_Optimizer(self, newoptimizer):
        self.optimizer = newoptimizer