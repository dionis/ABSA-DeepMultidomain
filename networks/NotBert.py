import sys
import torch
import utils

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla, opt):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla

        #Model atributte to assoaciated to BERT pre-trained
        self.model = None

        #Only learn parameters in these structure in AR1 algorithm
        #self.owner_structure = torch.nn.Linear(opt.bert_dim, opt.polarities_dim )


        #Create a Linear output layer for each domain or task
        #in regularization concept only fix wheigts in output layer for each domain

        #self.tm = torch.nn.Linear( opt.polarities_dim, opt.polarities_dim)

        # self.gate = torch.nn.Sigmoid()
        # # All embedding stuff should start with 'e'
        # self.ec1 = torch.nn.Embedding(len(self.taskcla), 64)
        # self.ec2 = torch.nn.Embedding(len(self.taskcla), 128)
        # self.ec3 = torch.nn.Embedding(len(self.taskcla), 256)
        # self.efc1 = torch.nn.Embedding(len(self.taskcla), 2048)
        # self.efc2 = torch.nn.Embedding(len(self.taskcla), 2048)
        # # """ (e.g., used in the compression experiments)
        # lo, hi = 0, 2
        # self.ec1.weight.data.uniform_(lo, hi)
        # self.ec2.weight.data.uniform_(lo, hi)
        # self.ec3.weight.data.uniform_(lo, hi)
        # self.efc1.weight.data.uniform_(lo, hi)
        # self.efc2.weight.data.uniform_(lo, hi)
        self.hard = True  #Identify if inner model use Hard algoritm or not
        # """
        return

    def forward(self,t,x,s=1):
        if self.model != None:
           bert_output = None
           masks = None
           try:
               if self.model.hat != None and self.model.hat == True:
                  bert_output, masks = self.model.forward(t, x,s)
               else:
                   bert_output = self.model.forward(t,x,s)
           except (AttributeError):
               bert_output = self.model.forward(t, x)

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
        # if self.model.hat != None:
        #     return self.model.get_view_for(n, masks)
        return 0.0

    def set_Model(self, newmodel):
        self.model = newmodel
        self.last = self.model.last
        ####
        #  Used only in hat approach
        #  each base algortihm  knowes how compute get_view_for
        ###

    def get_Model(self):
        return self.model


    def getWordInDomain(self):
        if self.model == None:
            return None
        return self.model.word_in_domain

    def clearWordInDomain(self):
        if self.model == None:
            return False
        self.model.word_in_domain.clear()
        return True

    def getEmbeddingMatrix(self, vocabulary):
        if self.model == None:
            return None
        return self.model.getEmbeddingMatrixEx(vocabulary)

    def buildASA(self, t, dataset, word_contex_domain, aspect_domain ):
        if self.model == None:
            return None
        return self.model.buildASA(t, dataset,word_contex_domain, aspect_domain)


    def insertKnowBase(self, ASAt, CSEt):
        if self.model == None:
            return None
        self.model.insertKnowBase(ASAt, CSEt)

    def get_bert_model_parameters(self):
        if self.model != None:
            return self.model.named_parameters()
        return None

    def set_ModelOptimizer(self, optimizer):
        if self.model != None:
            self.model.set_Optimizer(optimizer)

    def get_Optimizer(self):
        if self.model != None:
            return self.model.optimizer;
        return None