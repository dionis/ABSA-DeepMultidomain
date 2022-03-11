# -*- coding: utf-8 -*-
# file: memnet.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import numpy as np
from layers.attention import Attention
import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from layers.squeeze_embedding import SqueezeEmbedding
import torch.nn.functional as F
import re

class LifelongABSA(nn.Module):
    
    def locationed_memory(self, memory, memory_len):
        # here we just simply calculate the location vector in Model2's manner
        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        memory_len = memory_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for idx in range(memory_len[i]):
                weight[i].append(1-float(idx+1)/memory_len[i])
            for idx in range(memory_len[i], seq_len):
                weight[i].append(1)
        weight = torch.tensor(weight).to(self.opt.device)
        memory = weight.unsqueeze(2)*memory
        return memory

    def __init__(self, embedding_matrix, opt):
        super(LifelongABSA, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.attention = Attention(opt.embed_dim, score_function='mlp')
        #self.x_linear = nn.Linear(opt.embed_dim, opt.embed_dim)
        # self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)
        self.polarities_dim = opt.polarities_dim
        self.last = torch.nn.ModuleList()
        self.hat = False
        self.word_in_domain = torch.zeros(1, dtype=torch.int64)

        self.context_attention = dict()
        self.aspect_context_attention = dict()
        self.currenttask = -1
        self.currentSentence = 0

        for t in range(self.opt.taskcla):
            self.last.append(nn.Linear(opt.polarities_dim, opt.polarities_dim))
            self.context_attention[t] = dict()
            self.aspect_context_attention[t] = dict()

        #Parameter(torch.Tensor(out_features, in_features))
        self.W =   torch.nn.Parameter(torch.randn(opt.embed_dim, opt.polarities_dim ))

        self.opt.initializer(self.W)

        #where the negative, neutral, and positive classes are denoted
        # as [1, 0 ,0], [0, 1 ,0] and [0, 0 ,1] respectively

        self.An = torch.tensor( np.array([[1,-1,-1]]), dtype=torch.float32, requires_grad=False )
        self.Bn = torch.tensor(np.array([[1, 0, 0]]), dtype=torch.float32, requires_grad=False)

        self.Ap = torch.tensor(np.array([[-1, -1, 1]]), dtype=torch.float32, requires_grad=False)
        self.Bp = torch.tensor(np.array([[0, 0, 1]]), dtype=torch.float32, requires_grad=False)

        self.L2MN = False

    def forward(self, t, inputs, s):
        if self.currenttask  != t:
            self.currenttask = t
            self.currentSentence = 0
        text_raw_without_aspect_indices, aspect_indices = inputs[0], inputs[1]
        memory_len = torch.sum(text_raw_without_aspect_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)



        memory = self.embed(text_raw_without_aspect_indices)
        memory = self.squeeze_embedding(memory, memory_len)
        # memory = self.locationed_memory(memory, memory_len)
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)

        #memory_sentence =  torch.sum(memory, dim=1)

        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))
        x = aspect.unsqueeze(dim=1)
        out_at, score_sentence = self.attention(memory, x)

        #Save sentence context attention
        o_lifelog =  torch.matmul (score_sentence,memory)

        s_output = torch.matmul((x + o_lifelog), self.W)
        if self.L2MN == True:
           #print ("Execute L2MN algorithm ")
           #Only obtein the real index without 80
           faPosVector = self.getFaPositiveVector(text_raw_without_aspect_indices, aspect_indices)
           faPosVectorTensor = torch.tensor(faPosVector, dtype=torch.float32 , requires_grad=False)

           faNegVector = self.getFaPositiveVector(text_raw_without_aspect_indices, aspect_indices)
           faNegVectorTensor = torch.tensor(faNegVector, dtype=torch.float32, requires_grad=False)

           hQMatrix = self.getHqVectorMatrix(text_raw_without_aspect_indices, aspect_indices)
           hQMatrixTensor = torch.tensor(hQMatrix, dtype=torch.float32, requires_grad=False)

           o_lifelogPositive = torch.matmul (faPosVectorTensor,memory)

           o_lifelogNegative = torch.matmul(faNegVectorTensor, memory)

           score_sentence_Hq = (score_sentence + faNegVectorTensor + faPosVectorTensor)

           multScoreSenteceHq = torch.matmul(score_sentence_Hq,hQMatrixTensor)


           #Positive actions
           parcialPositive = torch.matmul(self.Ap , torch.transpose(self.Bp, 0, 1))
           s_output_positive = torch.matmul( o_lifelogPositive, self.W)


           parcialPositive = torch.matmul(parcialPositive, s_output_positive)

           #Negative actions
           parcialNegative = torch.matmul(self.An , torch.transpose(self.Bn, 0, 1))
           s_output_negative = torch.matmul(o_lifelogNegative, self.W)
           parcialNegative = torch.matmul(parcialNegative, s_output_negative)


           sjoin = s_output + parcialPositive + parcialNegative + multScoreSenteceHq


        # for _ in range(self.opt.hops):
        #     x = self.x_linear(x)
        #     out_at, _ = self.attention(memory, x)
        #     x = out_at + x
        # x = x.view(x.size(0), -1)
        # out = self.dense
        y = []
        #y.append(self.last[i](s_output).view(-1, self.opt.polarities_dim))
        for i, _ in enumerate(range(self.opt.taskcla)):
            y.append(self.last[i](s_output).view(-1,self.opt.polarities_dim))


        for ielement in range(text_raw_without_aspect_indices.shape[0]):
            self.context_attention[t.item()][self.currentSentence] = zip(text_raw_without_aspect_indices[ielement],
                                                                  score_sentence[ielement][0])

            self.aspect_context_attention[t.item()][self.currentSentence] = (text_raw_without_aspect_indices[ielement]
                                                                                              ,aspect_indices[ielement])
            self.currentSentence += 1



        #Update all word index for each sentence
        self.word_in_domain = torch.unique(torch.cat((self.word_in_domain, text_raw_without_aspect_indices.view(-1))))


    # for iBatch in range(score_sentence.size(0)):
        #     sentencePos = iBatch + t
        #
        #     attentionContext = dict()
        #     for i, iattention in score_sentence:
        #         attentionContext[text_raw_without_aspect_indices[sentencePos]] = iattention
        #     self.context_attention[sentencePos].update(attentionContext)

        return y


    def getEmbeddingMatrixEx(self,vocabulary):

        if self.embed == None or type(vocabulary) == type(None):
            return None
        t = torch.unique(vocabulary)
        memory = self.embed(t)
        # Where M in R t.q V * K
        # M = WC
        return torch.matmul(memory,self.W)


    def buildASA(self, task, dataset, word_contex_domain, aspect_domain):
        exdomain_context_sentiment = dict()
        domain_context_sentiment = dict()

        #Build data structure
        for ivalue, iword_index in enumerate(word_contex_domain):
            exdomain_context_sentiment[iword_index] = dict()
            domain_context_sentiment[iword_index] = dict()

            for ivalue, iaspect in enumerate(aspect_domain):
                exdomain_context_sentiment[iword_index][iaspect] = {0:(0,0),1:(0,0),2:(0,0)}
                domain_context_sentiment[iword_index][iaspect] = {0: 0, 1: 0, 2: 0}

        #Count exist context words
        for iValue, isentences in enumerate (dataset):
           contexIndexAttention =  self.context_attention[task][iValue]
           text_raw_without_aspect_indices, aspect_indices =  self.aspect_context_attention[task][iValue]
           targets = isentences['polarity']
           ### Polarity convenction
           # 0 negative
           # 1 neutral
           # 2 positive
           ###
           for idex, (index, score) in enumerate(contexIndexAttention):
               if index.item() in domain_context_sentiment:
                   #print("Index = " + str(text_raw_without_aspect_indices[idex])  + " == " + str(index.item()))
                   #print ("Score = " + str(score.item()))
                   for iaspect in aspect_indices:
                       if iaspect.item() in domain_context_sentiment[index.item()]:
                           word_numerator, word_denominator = exdomain_context_sentiment[index.item()][iaspect.item()][targets]
                           word_denominator += 1
                           word_numerator +=1*score.item()
                           exdomain_context_sentiment[index.item()][iaspect.item()][targets] = (word_numerator,word_denominator)

        #Compute probabilities
        for ivalue, iword_index in enumerate(word_contex_domain):
            for ivalue, iaspect in enumerate(aspect_domain):
                for iopinion in range(3):
                  word_numerator, word_denominator = exdomain_context_sentiment[iword_index.item()][iaspect.item()][iopinion]
                  if word_denominator != 0:
                     domain_context_sentiment[iword_index.item()][iaspect.item()][iopinion] = word_numerator/word_denominator

        return domain_context_sentiment


    def insertKnowBase(self, ASAt, CSEt):
        self.currentASAt = ASAt
        self.currentCSEt = CSEt
        self.L2MN = True

    def getFAvector(self, type, sent_index_word, list_aspect):
        if self.currentASAt == None:
            return None

        #index = 0 negative
        #index = 1 neutral
        #index = 2 positive
        index = 0
        fAList = []
        if type == "positive":
            index = 1
        rowSize = sent_index_word.shape[0]
        asaWordKeys = self.currentASAt.keys()

        memory_len = torch.sum(list_aspect != 0, dim=-1)
        index_len = torch.sum(sent_index_word != 0, dim=-1)

        sent_index_word =  self.squeeze_embedding(sent_index_word, index_len)
        nlist_aspect = self.squeeze_embedding(list_aspect, memory_len)

        for iRow in range(rowSize):
            fA = []
            list_index_word = sent_index_word[iRow]
            for word in list_index_word:
                if not ( word.item() in asaWordKeys ):
                    fA.append(0)
                else: #Exist in Knowldge Base
                   aspectDict = self.currentASAt[word.item()]
                   aspectDictKeys = aspectDict.keys()
                   list_aspect_row = nlist_aspect[iRow]

                   aspectToCompare = set([ival.item() for ival in list_aspect_row if ival != 0])
                   aspectInterset = set(aspectDictKeys)
                   aspectInterset = aspectInterset &  aspectToCompare
                   if len (aspectInterset)== 0 :
                       fA.append(0)
                   else:
                       for iaspect in aspectInterset:
                           scoreattention =aspectDict[iaspect][index]
                           fA.append(scoreattention)
                           break

            fAList.append([fA])

        return np.array(fAList)

    def getFaPositiveVector(self,sent_index_word, list_aspect):
        return self.getFAvector("positive",sent_index_word, list_aspect)

    def getFaNegativeVector(self,sent_index_word, list_aspect):
        return self.getFAvector("negative",sent_index_word, list_aspect)

    def getHqVectorMatrix(self,sent_index_word, list_aspect):
        if self.currentCSEt == None:
            return None
        resultHqList = list()
        rowSize = sent_index_word.shape[0]
        index_len = torch.sum(sent_index_word != 0, dim=-1)
        sent_index_word = self.squeeze_embedding(sent_index_word, index_len)

        currentCSEtKeys =  self.currentCSEt.keys()
        for iRow in range(rowSize):
            resultHq = list()
            list_index_word = sent_index_word[iRow]
            for word in list_index_word:
                if not (word.item() in currentCSEtKeys):
                    resultHq.append([0, 0, 0])
                else:
                    aspectDict = self.currentCSEt[word.item()]
                    resultHq.append(aspectDict.numpy())

            resultHqList.append(resultHq)

        return  np.array(resultHqList)

    def get_Optimizer(self):
        if self.optimizer != None:
            return self.optimizer
        return None

    def set_Optimizer(self, newoptimizer):
        self.optimizer = newoptimizer

