import sys,time
import numpy as np
import os
from datetime import datetime
import psutil
import torch
from copy import deepcopy
import utils
from sklearn import metrics
from torch.autograd import Variable
import re

########################################################################################################################
##
#   Knowledge Base for each Domain
##
class RK(object):
    def __init__(self,cSM, aSA, wordVocabulary, aspectVocabulary):
        self.wordContext_CSM =  torch.nn.Embedding.from_pretrained(cSM)
        self.aspect_ASA = aSA
        self.word_vocabulary = wordVocabulary
        self.aspect_vocabulary = aspectVocabulary

        self.orderedVocabulary = list(self.word_vocabulary)
        self.orderedVocabulary.sort()



    def getIndexInWordVocabulary(self, word):
        if word == None or word == "" or   self.orderedVocabulary == None:
            return -1
        else:
            try:
                if len( self.orderedVocabulary ) == 0:
                    return -1
                return self.orderedVocabulary.index(word)
            except(ValueError):
                return -1



class Appr(object):


    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,lamb=0.75,smax=400,args=None):
        self.model=model
        self.opt = args
        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.ce=torch.nn.CrossEntropyLoss()


        self.lamb=lamb
        self.smax=smax
        self.logpath = None
        self.single_task = False
        self.logpath = args.parameter

        # Synaptic Implementatio development
        self.small_omega_var = {}
        self.previous_weights_mu_minus_1 = {}
        self.big_omega_var = {}
        self.aux_loss = 0.0

        self.reset_small_omega_ops = []
        self.update_small_omega_ops = []

        # Parameters for the intelligence synapses model.
        self.param_c = 0.1
        self.param_xi = 0.1

        self.learning_rate = 0.001
        self.exp_pow = torch.tensor(2)
        self.exp_pow = 2

        if self.model != None:
            self.task_size = 1 if self.model.taskcla == None else len(self.model.taskcla)


        if self.model != None:
          self.task_size = len(self.model.taskcla)
          self.wordInDomVocabulary = dict()
          self.aspectInDomVocabulary = dict()

          #Values taken from ""


        optimizer = self.model.get_Optimizer()
        if optimizer != None:
            self._set_optimizer(optimizer)


        self.current_task = -1

        self.cse = dict()
        self.asa = dict()

        self.rkList = dict()

        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            if len(params)>1:
                if utils.is_number(params[0]):
                    self.lamb=float(params[0])
                else:
                    self.logpath = params[0]
                if utils.is_number(params[1]):
                    self.smax=float(params[1])
                else:
                    self.logpath = params[1]
                if len(params)>2 and not utils.is_number(params[2]):
                    self.logpath = params[2]
                if len(params)>3 and utils.is_number(params[3]):
                    self.single_task = int(params[3])
            else:
                self.logpath = args.parameter

        if self.logpath is not None:
            self.logs={}
            self.logs['train_loss'] = {}
            self.logs['train_acc'] = {}
            self.logs['train_reg'] = {}
            self.logs['valid_loss'] = {}
            self.logs['valid_acc'] = {}
            self.logs['valid_reg'] = {}
            self.logs['mask'] = {}
            self.logs['mask_pre'] = {}
        else:
            self.logs = None

        self.mask_pre=None
        self.mask_back=None

        return

    def find_noleaf(self, list_variables):
        print("Parameters")
        for i, (name, var) in enumerate(list_variables):
            if var.is_leaf == False:
                print("Leaf tensor False")
                break
        return

    def _set_optimizer(self, _new_optimize):
        if _new_optimize != None: self.optimizer = _new_optimize

    def _get_optimizer(self,lr=None):
        if lr is None: lr = self.lr

        print("!!!!New optmization!!!!!")
        # if self.optimizer != None:
        #     print("--------Optmization---------")
        #     return self.optimizer

        # return torch.optim.SGD(self.tensorVariables, lr=lr)
        # return torch.optim.SGD(self.model.parameters(),lr=lr)
        return self.opt.optimizer(self.model.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

    def train(self, t, train_data_loader, test_data_loader, val_data_loader):
        best_loss=np.inf
        #best_model=utils.get_model(self.model)
        lr=self.lr
        patience=self.lr_patience


        task = torch.autograd.Variable(torch.LongTensor([t]).cuda(),
                                       volatile=False, requires_grad=False) if torch.cuda.is_available() \
                                     else torch.autograd.Variable(torch.LongTensor([t]), volatile=False, requires_grad=False)

        if t != self.current_task:
            ###It need that al weights in last output layer are inicialized in zero
            ###Optimization in original paper
            ###no usal la inicializacion Gaussiana y de Xavier. Aunque se conoce que los pesos de las
            ###redes no deben inicializarce a 0 pero esto es para niveles intermedios y no para los niveles
            ###de salida

            self.current_task = t

        ##
        ##  LA VARIABLE tm se coloca entre los valores a optimizar??????
        ##

        print(
            " ###### Update status of last layer weight in current task(domain) AVOID Stocastic Gradient ########")

        for name, var in self.model.named_parameters():
            if name.find("model.last.") != -1:
                var.requires_grad_(False);
                if re.match("model.last." + str(t), name) != None:
                    print("Variable " + name + " update to SGD")
                    var.requires_grad_(True);
        self.optimizer = self._get_optimizer(lr)
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()

            print("----- Optimizer -----")
            print(self.optimizer)
            print("----------------------")
            #print("1")
            self.train_epochesi(t, train_data_loader)

            clock1 = time.time()

            #print("2")
            train_loss, train_acc, train_recall, train_f1, train_cohen_kappa = self.eval_withregsi(t, val_data_loader)

            #print("3")
            clock2 = time.time()

            dataset_size = len(val_data_loader.dataset)
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train-Val: loss={:.3f}, acc={:5.1f}, f1={:5.1f}, cohen_kappa={:5.1f}%|'.format(
                e + 1,
                1000 * self.sbatch * (
                    clock1 - clock0) / dataset_size,
                1000 * self.sbatch * (
                    clock2 - clock1) / dataset_size,
                train_loss,
                100 * train_acc,
                100 * train_f1,
                100 * train_cohen_kappa),
                end='')

            # Valid
            #print("4")
            valid_loss, valid_acc , valid_recall, valid_f1, valid_cohen_kappa = self.eval_withregsi(t, test_data_loader)
            print(' Test: loss={:.3f}, acc={:5.1f}, f1={:5.1f}, cohen_kappa={:5.1f}%|'.format(valid_loss, 100 * valid_acc, 100 * valid_f1,100*valid_cohen_kappa),
                  end='')

            #print("5")
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                #best_model=utils.get_model(self.model)
                patience=self.lr_patience
                print(' *',end='')
            else:
                patience-=1
                if patience<=0:
                    lr/=self.lr_factor
                    print(' lr={:.1e}'.format(lr),end='')
                    if lr<self.lr_min:
                        print()
                        break
                    patience=self.lr_patience
                    self.optimizer=self._get_optimizer(lr)
            print()
            #print("6")
            #self.find_noleaf(self.model.named_parameters())


        print(" ###### Show status of last layer weight in current task(domain) ########")
        toViewLasLayer = []
        for name, var in self.model.named_parameters():
            if name.find("model.last.") != -1:
                print("Requiere Grand ==> " + str(var.requires_grad))
                print("Variable name " + name + " == " + str(var.data))

                toViewLasLayer.append((name, var))


        return

    def train_epochesi(self, t, train_data_loader, thres_cosh=50,thres_emb=6):
        self.model.train()
        # Loop batches

        loop_size = 0

        # Task domain:
        if t >= 2:
            CSEt, _ = self.getCSMNewDomain(t, self.wordInDomVocabulary[t], self.aspectInDomVocabulary[t])
            ASAt, similarAspectList = self.getASANewDomain(t, self.wordInDomVocabulary[t], self.aspectInDomVocabulary[t])

            self.model.insertKnowBase(ASAt, CSEt)

        loop_size = 0
        global_step = 0
        n_correct, n_total, loss_total = 0, 0, 0

        for i_batch, sample_batched in enumerate(train_data_loader):
            print("Batch size: " + str (sample_batched.__len__()))
            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            # outputs = self.model(inputs)
            targets = sample_batched['polarity'].to(self.opt.device)


            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False, requires_grad=False) \
                if torch.cuda.is_available() \
                else torch.autograd.Variable(torch.LongTensor([t]), volatile=False, requires_grad=False)


            # Forward current model

            startDateTime = datetime.now()
            outputs,_=self.model.forward( task, inputs)
            #print('Train DataTime', datetime.now() - startDateTime)
            #print("Train forward")
            self.getMemoryRam()


            output = outputs[t]
            loss=self.criterion(t,output,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            n_correct += (torch.argmax(output, -1) == targets).sum().item()
            n_total += len(output)
            loss_total += loss.item() * len(outputs)
            if global_step % self.opt.log_step == 0:
                # train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                # print('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
                print('loss: {:.4f}'.format(train_loss))


            self.optimizer.step()


        #Build Context-Sentiment-Effec Matrix by t domain
        self.cse[t] = self.buildCSE(self.model.getWordInDomain())
        self.asa[t] = self.buildASA(t, train_data_loader.dataset)

        self.setDomaiVocabulary(t, self.cse[t],  self.asa[t] ,  self.wordInDomVocabulary[t],  self.aspectInDomVocabulary[t] )


        # Mean
        # 1 4 7
        # r = torch.mean(v, 1)  # Size 3: Mean in dim 1
        #
        # r = torch.mean(v, 1, True)  # Size 3x1 since keep dimension = True

        # ERRROR
        #
        # File
        # "E:/___Dionis_MO/Articulos/IMPLEMENTACION/SOURCE/Inoid_ABSA_DL/ABSA-PyTorch-master\approaches\ar1.py", line
        # 355, in train_epochesi
        # self.model.last[t][i_output] = self.model.tm[i_output] - torch.mean(self.model.tm)
        #  TypeError: 'Linear' object does not support indexing
        print("1.8 ")
        return

    def eval_withregsi(self, t, val_data_loader):
        total_loss = 0
        total_acc = 0
        total_num = 0

        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None


        self.model.eval()

        total_reg = 0

        for i_batch, sample_batched in enumerate(val_data_loader):
            # clear gradient accumulators

            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            # outputs = self.model(inputs)
            targets = sample_batched['polarity'].to(self.opt.device)

            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False, requires_grad=False) \
                if torch.cuda.is_available() \
                else torch.autograd.Variable(torch.LongTensor([t]), volatile=False,requires_grad=False)

            # Forward
            startDateTime = datetime.now()
            outputs,_ = self.model.forward(task, inputs)
            #print('Eval DataTime', datetime.now() - startDateTime)
            #print("Eval forward")
            self.getMemoryRam()


            output = outputs[t]
            loss = self.criterion(t, output, targets)
            _, pred = output.max(1)
            hits = (pred == targets).float()

            n_correct += (torch.argmax(output, -1) == targets).sum().item()

            # Log
            current_batch_size = len(pred)
            total_loss += loss.data.cpu().numpy() * current_batch_size
            total_acc += hits.sum().data.cpu().numpy()
            total_num += current_batch_size

            if t_targets_all is None:
                t_targets_all = targets.detach().numpy()
                t_outputs_all = output.detach().numpy()
            else:
                t_targets_all =  np.concatenate((t_targets_all, targets.detach().numpy()), axis=0)
                t_outputs_all =  np.concatenate((t_outputs_all, output.detach().numpy()), axis=0)

        #OJOOOO DEBEMOS REVISAR LAS LABELS [0,1,2] Deben corresponder a como las pone la implementacion
        ##### FALTA LA ETIQUETA PARA CUANDO NO ES ASPECTO
        f1 = metrics.f1_score(t_targets_all, np.argmax(t_outputs_all, -1), labels=[0, 1, 2], average='macro')
        recall = metrics.recall_score(t_targets_all, np.argmax(t_outputs_all, -1), labels=[0, 1, 2],
                              average='macro')


        cohen_kappa = metrics.cohen_kappa_score(t_targets_all, np.argmax(t_outputs_all, -1))

        return total_loss / total_num, total_acc / total_num, recall, f1, cohen_kappa

###-------------------------------------------------------------------------------------------------------------
    def eval(self, t, test_data_loader):
        return self.eval_withregsi(t, test_data_loader)




    def criterion(self,t,output,targets):
        # Regularization for all previous tasks
        loss_reg=0

        #
        # for name, var in self.model.named_parameters():
        #     print ("Variable: ", name)

        # if t>0:
        #     for name, var in self.tensorVariablesTuples:
        #         loss_reg += torch.sum(torch.mul( self.big_omega_var[name], (self.previous_weights_mu_minus_1[name] - var.data).pow(self.exp_pow)))

            # for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
            #     loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
        # + self.param_c * loss_reg
        return self.ce(output,targets)


########################################################################################################################
    def setAllAspect(self, all_aspect):
        self.all_aspect = all_aspect

    def setAllWord(self, all_word_vocabulary):
        self.all_word_vocabulary = all_word_vocabulary


    def setAspectInDomain(self, task, aspect_vocabulary):
        self.aspectInDomVocabulary[task] = aspect_vocabulary

    def setWordInDomain(self, task, word_vocabulary):
        self.wordInDomVocabulary[task] = word_vocabulary

        # Serialize model, optimizer and other parameters to file

    def saveModel(self, topath):
        torch.save({
            'epoch': self.nepochs,
            'model_state_dict': self.model.get_Model().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.ce,
            'learning_rate': self.lr,
            'batch': self.sbatch,
            'task_size': self.task_size
        }, topath)

        return True



        # Unserialize model, optimizer and other parameters from file

    def loadModel(self, frompath):
        if not os.path.exists(frompath):
            return False
        else:
            checkpoint = torch.load(frompath)
            self.model.get_Model().load_state_dict(checkpoint['model_state_dict'])

            self.optimizer = self.opt.optimizer(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.ce = checkpoint['loss']
            return True

    def buildCSE(self, vocabulary):
        if  type(vocabulary) == type(None):
            return None
        else:
            memory = self.model.getEmbeddingMatrix(vocabulary)
            return ( memory)

    def buildASA(self, t, dataset):
        return self.model.buildASA(t, dataset, self.wordInDomVocabulary[t], self.aspectInDomVocabulary[t])

    def updateGlobalCSE(self):
        pass

    def updateGlobalASA(self, current_domain_asa):
        pass

    """
     The Aspect-Sentiment Attention (ASA)
    """
    def getASANewDomain(self,t, word_context_vocabulary, aspect_vocabulary):
        global_vocabulary = set()
        global_aspect = set()

        for ikey, iRK in self.rkList.items():
          global_vocabulary = global_vocabulary | iRK.word_vocabulary
          global_aspect = global_aspect | iRK.aspect_vocabulary
        asaDictDomain = dict()

        similarWord = global_vocabulary.intersection(word_context_vocabulary)
        similarAspect =  global_aspect.intersection(aspect_vocabulary)

        for wordindex in similarWord:
           if not (wordindex in asaDictDomain):
              asaDictDomain[wordindex] = list()
           for ikey, iRK in self.rkList.items():
                 if wordindex in iRK.word_vocabulary: #Obtain all values in each domain if in all domains
                     asaDictDomain[wordindex].append(iRK.aspect_ASA[wordindex])

        #Average each iqual aspect bucket
        resultAspectAttnDist = dict()
        for keyWord, attentionList in asaDictDomain.items():
            aspectSentAttentDistribution = dict()
            for item in attentionList:
              for attentionKey, polaritiesvalues in item.items():
                if attentionKey in aspectSentAttentDistribution:
                    negative, neutral, positive = polaritiesvalues
                    oldnegative, oldneutral, oldpositive,counter = aspectSentAttentDistribution[attentionKey]
                    aspectSentAttentDistribution[attentionKey] = (negative + oldnegative, neutral + oldneutral, positive+oldpositive, counter+1)
                else:
                    negative, neutral, positive = polaritiesvalues
                    aspectSentAttentDistribution[attentionKey]= (negative,neutral,positive,1)

            resultAspect = dict()
            for iAspect, opinionValueAndCounter in aspectSentAttentDistribution.items():
                oldnegative, oldneutral, oldpositive, counter = opinionValueAndCounter
                #Specifically, we average the distribution values (j)
                #learned from past domains, where stands for a frequent word under aspect t and sentiment r
                resultAspect[iAspect] = (oldnegative/counter, oldneutral/counter,oldpositive/counter)

            resultAspectAttnDist[keyWord] = resultAspect

        return resultAspectAttnDist,similarAspect

    """
    The Context-Sentiment Effect (CSE) knowledge
    """
    def getCSMNewDomain(self,t, word_context_vocabulary, aspect_vocabulary):
        global_vocabulary = set()
        global_aspect = set()

        for ikey, iRK in self.rkList.items():
            global_vocabulary = global_vocabulary | iRK.word_vocabulary
            global_aspect = global_aspect | iRK.aspect_vocabulary
        asaDictDomain = dict()

        similarWord = global_vocabulary.intersection(word_context_vocabulary)
        similarAspect = global_aspect.intersection(aspect_vocabulary)
        aspectSentAttentDistribution = dict()
        for wordindex in similarWord:
            if not (wordindex in asaDictDomain):
                asaDictDomain[wordindex] = list()
            for ikey, iRK in self.rkList.items():
                if wordindex in iRK.word_vocabulary:  # Obtain all values in each domain if in all domains
                    indexInVocabulary =  iRK.getIndexInWordVocabulary(wordindex)
                    if indexInVocabulary == -1: #No exits in Domain Vocabulary
                        raise Exception("No exist word index " + wordindex  + " in domain")

                    tensorEmbedding = torch.tensor(np.array(indexInVocabulary),dtype=torch.int64 , requires_grad=False )

                    asaDictDomain[wordindex].append(iRK.wordContext_CSM(tensorEmbedding))


        # Average each iqual aspect bucket
        resultAspectAttnDist = dict()
        for keyWord, attentionList in asaDictDomain.items():
            averagePattern = len(attentionList)
            vectorSum = torch.zeros(self.opt.polarities_dim)
            for item in attentionList:
                vectorSum += item

            resultAspectAttnDist[keyWord] = vectorSum/averagePattern

        return resultAspectAttnDist, similarAspect

    def setDomaiVocabulary(self, t, wordCSM, aspectASA, wordVocabularyDomain, aspectVocabularyDomain):
        self.rkList[t]= RK(wordCSM, aspectASA,wordVocabularyDomain,aspectVocabularyDomain)


    def getMemoryRam(self):
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        #print('memory use:', memoryUse)