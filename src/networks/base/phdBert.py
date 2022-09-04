import sys
import torch
import os
from transformers import BertModel, BertConfig
from torch import nn
import torch.nn.functional as F
import utils

from .bert_spc import BERT_SPC


class Net(torch.nn.Module):

    def __init__(self,taskcla,args):
        super(Net,self).__init__()

        #ncha,size,_=inputsize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.taskcla=taskcla
        config = BertConfig.from_pretrained(args.bert_model, cache_dir = ".." + os.path.sep + "Transformer" + os.path.sep,  local_files_only=False)
        config.return_dict = False
        self.args = args

        self.args.taskcla = len(self.taskcla)
        #Model atributte to assoaciated to BERT pre-trained
        self.model =  BERT_SPC(BertModel.from_pretrained(args.bert_model,config=config, cache_dir = ".." + os.path.sep + "Transformer" + os.path.sep,  local_files_only=False), self.args)

        self.last = self.model.last

        ###
        ### Inicial todos los pesos a 0 segun algoritmo del paper
        ###
        self.tm = torch.nn.Linear(self.args.nclasses, self.args.nclasses)

        # Initialice 0 because reset_parametes input other standar initialization

        neuronsize = self.tm.weight.shape[0]
        input_neuron_size = self.tm.weight.shape[1]

        for i in range(neuronsize):
            for j in range(input_neuron_size):
                self.tm.weight.data[i, j] = 0.0

            self.tm.bias.data[i] = 0.0

        self.hard = True  #Identify if inner model use Hard algoritm or not
        # """
        return


    def forward(self,t,x,s=1):
        if self.model != None:
           bert_output = None
           masks = None
           try:
               if self.model.hat != None and self.model.hat == True:
                  bert_output, masks = self.model.forward(t,x,s)
               else:
                   bert_output = self.model.forward(t,x,s)
           except (AttributeError):
               bert_output = self.model.forward(t, x, s)

           bert_output = self.tm(bert_output[t.to(self.device )])
           return bert_output,masks
        return None, None
        # h=x.view(x.size(0),-1)
        # h=self.drop(self.relu(self.fc1(h)))
        # h=self.drop(self.relu(self.fc2(h)))
        # h=self.drop(self.relu(self.fc3(h)))
        # y=[]
        # for t,i in self.taskcla:
        #     y.append(self.last[t](h))
        # return y

    def mask(self,t,s=1):
        if self.model.hat != None:
            return self.model.mask(t,s)

        gc1=self.gate(s*self.ec1(t))
        gc2=self.gate(s*self.ec2(t))
        gc3=self.gate(s*self.ec3(t))
        gfc1=self.gate(s*self.efc1(t))
        gfc2=self.gate(s*self.efc2(t))
        return [gc1,gc2,gc3,gfc1,gfc2]



    def get_view_for(self, n, masks):
        if self.model.hat != None:
            return self.model.get_view_for(n, masks)
        return 0.0

    def set_Model(self, newmodel):
        self.model = newmodel
        self.last = self.model.last
        ####
        #  Used only in hat approach
        #  each base algortihm  knowes how compute get_view_for
        ###

        ####
        #  Used only in hat approach
        #  each base algortihm  knowes how compute get_view_for
        ###

    def get_Model(self):
        return self.model

    def set_ModelOptimizer(self, optimizer):
        if self.model != None:
            self.model.set_Optimizer(optimizer)

    def get_Optimizer(self):
        if self.model != None and  hasattr(self.model, 'optimizer') and self.model.optimizer != None:
            return self.model.optimizer;
        return None

    def get_bert_model_parameters(self):
        if self.model != None:
            return self.model.get_bert_model_parameters()
        return None


    def getLastLayer(self):
        if self.model == None or self.model.last == None:
            return None
        return self.model.last