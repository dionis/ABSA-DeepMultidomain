# -*- coding: utf-8 -*-
# file: lcf_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.


# The code is based on repository: https://github.com/yangheng95/LCF-ABSA


import torch
import torch.nn as nn
import copy
import numpy as np

#from pytorch_pretrained_bert import BertPooler, BertSelfAttention
from pytorch_transformers.modeling_bert import BertPooler, BertSelfAttention


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len),
                                            dtype=np.float32), dtype=torch.float32).to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])

class LCF_BERT_HAT(nn.Module):
    def __init__(self, bert, opt):
        super(LCF_BERT_HAT, self).__init__()

        self.bert_spc = bert
        self.opt = opt
        # self.bert_local = copy.deepcopy(bert)  # Uncomment the line to use dual Bert
        self.bert_local = bert   # Default to use single Bert and reduce memory requirements
        self.dropout = nn.Dropout(opt.dropout)


        self.bert_SA = SelfAttention(bert.config, opt)
        self.linear_double = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        #self.linear_single = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.bert_pooler = BertPooler(bert.config)
        #self.dense =

        self.last = torch.nn.ModuleList()
        for t  in range(self.opt.taskcla):
            self.last.append(nn.Linear(opt.bert_dim, opt.polarities_dim))

        self.gate = torch.nn.Sigmoid()

        self.ec1 = torch.nn.Embedding(self.opt.taskcla, opt.bert_dim)
        self.ec2 = torch.nn.Embedding(self.opt.taskcla, bert.config.hidden_size)
        self.ec3 = torch.nn.Embedding(self.opt.taskcla, bert.config.hidden_size)

        self.hat = True

    def feature_dynamic_mask(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        mask_len = self.opt.SRD
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
            except:
                continue
            if asp_begin >= mask_len:
                mask_begin = asp_begin - mask_len
            else:
                mask_begin = 0
            for i in range(mask_begin):
                masked_text_raw_indices[text_i][i] = np.zeros((self.opt.bert_dim), dtype=np.float)
            for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len):
                masked_text_raw_indices[text_i][j] = np.zeros((self.opt.bert_dim), dtype=np.float)
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
                asp_avg_index = (asp_begin * 2 + asp_len) / 2
            except:
                continue
            distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
            for i in range(1, np.count_nonzero(texts[text_i])-1):
                if abs(i - asp_avg_index) + asp_len / 2 > self.opt.SRD:
                    distances[i] = 1 - (abs(i - asp_avg_index)+asp_len/2
                                        - self.opt.SRD)/np.count_nonzero(texts[text_i])
                else:
                    distances[i] = 1
            for i in range(len(distances)):
                masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def forward(self,t, inputs, s):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2]
        aspect_indices = inputs[3]

        bert_spc_out, _ ,_= self.bert_spc(text_bert_indices, bert_segments_ids)
        bert_spc_out = self.dropout(bert_spc_out)

        bert_local_out, _ , _ = self.bert_local(text_local_indices)
        bert_local_out = self.dropout(bert_local_out)

        masks = self.mask(t, s=s)
        gc1, gc2, gc3 = masks

        if self.opt.local_context_focus == 'cdm':
            masked_local_text_vec = self.feature_dynamic_mask(text_local_indices, aspect_indices)
            bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)

        elif self.opt.local_context_focus == 'cdw':
            weighted_text_local_features = self.feature_dynamic_weighted(text_local_indices, aspect_indices)
            bert_local_out = torch.mul(bert_local_out, weighted_text_local_features)


        out_cat = torch.cat((bert_local_out, bert_spc_out), dim=-1)


        # out_cat =   out_cat * gc2.view(1, 1, -1).expand_as(out_cat)
        mean_pool = self.linear_double(out_cat)

        # First hat apply
        mean_pool = mean_pool*gc1.view(1, 1, -1).expand_as(mean_pool)

        self_attention_out = self.bert_SA(mean_pool)

        #Second hat apply
        self_attention_out = self_attention_out* gc2.view(1, 1, -1).expand_as(self_attention_out)

        pooled_out = self.bert_pooler(self_attention_out)

        # Thrid hat apply
        pooled_out = pooled_out * gc3.view(1,-1).expand_as(pooled_out)
        #dense_out = self.dense(pooled_out)
        y = []
        for i, _ in enumerate(range(self.opt.taskcla)):
            y.append(self.last[i](pooled_out))
        return y,masks


    def mask(self,t,s=1):
        gc1=self.gate(s*self.ec1(t))
        gc2=self.gate(s*self.ec2(t))
        gc3=self.gate(s*self.ec3(t))

        return [gc1,gc2,gc3]

    def get_bert_model_parameters(self):
        variable_name = ["bert_SA","bert_pooler","linear_double","last"]
        modelVariables = []

        for i, (name, var) in enumerate(self.named_parameters()):
              for iname in variable_name:
                  if name.find(iname) != -1:
                      modelVariables.append((name,var))
                      break

        return modelVariables

    def get_view_for(self, n, masks):
        gc1, gc2, gc3 = masks

        # Variable == > bert_SA.SA.query.weight
        # Variable == > bert_SA.SA.query.bias
        # Variable == > bert_SA.SA.key.weight
        # Variable == > bert_SA.SA.key.bias
        # Variable == > bert_SA.SA.value.weight
        # Variable == > bert_SA.SA.value.bias
        # Variable == > linear_double.weight
        # Variable == > linear_double.bias
        # Variable == > bert_pooler.dense.weight
        # Variable == > bert_pooler.dense.bias
        # Variable == > last.0.weight
        # Variable == > last.0.bias
        # Variable == > last.1.weight
        # Variable == > last.1.bias

        variablesList = ["linear_double","bert_SA","bert_pooler"]
        if n == "linear_double.weight":
           return gc1.data.view(-1, 1).expand_as(self.linear_double.weight)
        elif n == 'linear_double.bias':
            return gc1.data.view(-1)

        elif n == 'bert_SA.SA.query.weight':
            post = gc2.data.view(-1, 1).expand_as(self.bert_SA.SA.query.weight)
            pre = gc2.data.view(1, -1).expand_as(self.bert_SA.SA.query.weight)
            return torch.min(post, pre)
        elif n == 'bert_SA.SA.query.bias':
            return gc2.data.view(-1)

        elif n == 'bert_SA.SA.key.weight':
            post = gc2.data.view(-1, 1).expand_as(self.bert_SA.SA.query.weight)
            pre = gc2.data.view(1, -1).expand_as(self.bert_SA.SA.query.weight)
            return torch.min(post, pre)
        elif n == 'bert_SA.SA.key.bias':
            return gc2.data.view(-1)

        elif n == 'bert_SA.SA.value.weight':
            post = gc2.data.view(-1, 1).expand_as(self.bert_SA.SA.value.weight)
            pre = gc2.data.view(1, -1).expand_as(self.bert_SA.SA.value.weight)
            return torch.min(post, pre)
        elif n == 'bert_SA.SA.value.bias':
            return gc2.data.view(-1)

        elif n == 'bert_pooler.dense.weight':
            post = gc3.data.view(-1, 1).expand_as(self.bert_pooler.dense.weight)
            pre = gc3.data.view(1, -1).expand_as(self.bert_pooler.dense.weight)
            return torch.min(post, pre)
        elif n == 'bert_pooler.dense.bias':
            return gc3.data.view(-1)


        return None

    def get_Optimizer(self):
        if self.optimizer != None:
            return self.optimizer
        return None

    def set_Optimizer(self, newoptimizer):
        self.optimizer = newoptimizer