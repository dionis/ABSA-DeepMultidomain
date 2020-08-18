import sys,time, os
import numpy as np
from datetime import datetime
import psutil
import torch
from sklearn import metrics
from copy import deepcopy

from setuptools.command.test import test

import utils

class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """


    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=100,lamb=5000,args=None):
        self.model=model
        self.opt = args
        self.model_old=None
        self.fisher=None

        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.ce=torch.nn.CrossEntropyLoss()
        optimizer = self.model.get_Optimizer()
        if optimizer != None:
            self._set_optimizer(optimizer)

        # self.optimizer=self._get_optimizer()
        self.lamb=lamb                      # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000
        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])

        # modelVariables = [(name, var) for i, (name, var) in enumerate(self.model.named_parameters()) if
        #                   name.find("bert") == -1]



        modelVariables = [(name, var) for i, (name, var) in enumerate(self.model.named_parameters())]
        self.tensorVariables = []
        self.tensorVariablesTuples = []
        for name, var in modelVariables:
            #print("Variable ==> " + name)
            self.tensorVariables.append(var)
            self.tensorVariablesTuples.append((name, var))

        return

    def _set_optimizer(self, _new_optimize):
        if _new_optimize != None: self.optimizer = _new_optimize

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr

        print("!!!!New optmization!!!!!")
        if self.optimizer != None:
            print("--------Optmization---------")
            return self.optimizer

        modelVariables = [(name, var) for i, (name, var) in enumerate(self.model.named_parameters())]

        # modelVariables = [(name, var) for i, (name, var) in enumerate(self.model.named_parameters()) if
        #                   name.find("bert") == -1]
        tensorVariables = []
        for name, var in modelVariables:
            #print("Variable ==> " + name)
            tensorVariables.append(var)

        return torch.optim.SGD(tensorVariables,lr=lr)

    def train(self, t, train_data_loader, test_data_loader, val_data_loader):
        best_loss=np.inf
        best_model=utils.get_model(self.model)
        lr=self.lr
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr)

        task = torch.autograd.Variable(torch.LongTensor([t]).cuda(),
                                       volatile=False) if torch.cuda.is_available() else torch.autograd.Variable(
            torch.LongTensor([t]), volatile=False)
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()

            self.train_epochewc(t, train_data_loader)

            clock1 = time.time()

            train_loss, train_acc,train_recall, train_f1 = self.eval_withregx(t, test_data_loader)

            clock2 = time.time()

            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e + 1,
                                                                                                        1000 * self.sbatch * (
                                                                                                            clock1 - clock0) / train_data_loader.__len__(),
                                                                                                        1000 * self.sbatch * (
                                                                                                            clock2 - clock1) / train_data_loader.__len__(),
                                                                                                        train_loss,
                                                                                                        100 * train_acc),
              end='')

            # Valid
            valid_loss, valid_acc, valid_recall, valid_f1 = self.eval_withregx(t, val_data_loader)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
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

        # Restore best
        utils.set_model_(self.model,best_model)

        # Update old
        self.model_old=deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old) # Freeze the weights

        # Fisher ops
        if t>0:
            fisher_old={}

            startDateTimeOldLast = datetime.now()
            for n,_ in self.model.named_parameters():

                fisher_old[n]=self.fisher[n].clone()

            print('DataTime OldLast', datetime.now() - startDateTimeOldLast)
            print("Analysis compute memory waste in Old Task")

        # self.fisher=utils.fisher_matrix_diag(t,xtrain,ytrain,self.model,self.criterion)
        self.fisher = utils.fisher_matrix_diag_nlp(t, train_data_loader, self.model, self.criterion, opt=self.opt )
        if t>0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            startDateTime = datetime.now()
            for n,_ in self.model.named_parameters():

                self.fisher[n]=(self.fisher[n]+fisher_old[n]*t)/(t+1)       # Checked: it is better than the other option
                #self.fisher[n]=0.5*(self.fisher[n]+fisher_old[n])
            print("Analysis compute memory waste")
            print('DataTime OldLast', datetime.now() - startDateTime)
        return

    def trainewc(self,t,xtrain,ytrain,xvalid,yvalid):
        best_loss=np.inf
        best_model=utils.get_model(self.model)
        lr=self.lr
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()
            self.train_epoch(t,xtrain,ytrain)
            clock1=time.time()
            train_loss,train_acc=self.eval(t,xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
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

        # Restore best
        utils.set_model_(self.model,best_model)

        # Update old
        self.model_old=deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old) # Freeze the weights

        # Fisher ops
        if t>0:
            fisher_old={}
            for n,_ in self.model.named_parameters():
                fisher_old[n]=self.fisher[n].clone()
        self.fisher=utils.fisher_matrix_diag(t,xtrain,ytrain,self.model,self.criterion)
        if t>0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            for n,_ in self.model.named_parameters():
                self.fisher[n]=(self.fisher[n]+fisher_old[n]*t)/(t+1)       # Checked: it is better than the other option
                #self.fisher[n]=0.5*(self.fisher[n]+fisher_old[n])

        return

    def train_epochewc(self, t, train_data_loader, thres_cosh=50,thres_emb=6):
        self.model.train()

        # r = np.arange(x.size(0))
        # np.random.shuffle(r)
        # r = torch.LongTensor(r).cuda()

        # Loop batches
        for i_batch, sample_batched in enumerate(train_data_loader):
            self.optimizer.zero_grad()

            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            # outputs = self.model(inputs)
            targets = sample_batched['polarity'].to(self.opt.device)

            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False) \
                if torch.cuda.is_available() \
                else torch.autograd.Variable(torch.LongTensor([t]), volatile=False)


            # Forward current model
            startDateTime = datetime.now()
            outputs,_=self.model.forward( task, inputs)
            print('Train DataTime', datetime.now() - startDateTime)
            print("Train forward")
            self.getMemoryRam()

            output=outputs[t]

            startDateTimeLoss = datetime.now()
            loss=self.criterion(t,output,targets)
            print('DataTime loss', datetime.now() - startDateTimeLoss)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return


    # def train_epoch(self,t,x,y):
    #     self.model.train()
    #
    #     r=np.arange(x.size(0))
    #     np.random.shuffle(r)
    #     r=torch.LongTensor(r).cuda()
    #
    #     # Loop batches
    #     for i in range(0,len(r),self.sbatch):
    #         if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
    #         else: b=r[i:]
    #         images=torch.autograd.Variable(x[b],volatile=False)
    #         targets=torch.autograd.Variable(y[b],volatile=False)
    #
    #         # Forward current model
    #         outputs=self.model.forward(images)
    #         output=outputs[t]
    #         loss=self.criterion(t,output,targets)
    #         print('loss computed =====> ')
    #         # Backward
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
    #         self.optimizer.step()
    #
    #     return


    def eval_withregx(self, t, val_data_loader):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        total_reg = 0

        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None

        for i_batch, sample_batched in enumerate(val_data_loader):
            # clear gradient accumulators
            self.optimizer.zero_grad()

            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            # outputs = self.model(inputs)
            targets = sample_batched['polarity'].to(self.opt.device)

            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False) \
                if torch.cuda.is_available() \
                else torch.autograd.Variable(torch.LongTensor([t]), volatile=False)

            # Forward

            startDateTime = datetime.now()
            outputs,_ = self.model.forward( task, inputs)
            print('Eval DataTime', datetime.now() - startDateTime)
            print ("Eval forward")
            self.getMemoryRam()

            output = outputs[t]
            loss = self.criterion(t, output, targets)
            _, pred = output.max(1)
            hits = (pred == targets).float()

            # Log
            total_loss += loss.data.cpu().numpy() * sample_batched.__len__()
            total_acc += hits.sum().data.cpu().numpy()
            total_num += sample_batched.__len__()

            if t_targets_all is None:
                t_targets_all = targets.detach().numpy()
                t_outputs_all = output.detach().numpy()
            else:
                t_targets_all =  np.concatenate((t_targets_all, targets.detach().numpy()), axis=0)
                t_outputs_all =  np.concatenate((t_outputs_all, output.detach().numpy()), axis=0)

        #global_output = t_outputs_all
        f1 = metrics.f1_score(t_targets_all, np.argmax(t_outputs_all, -1), labels=[0, 1, 2],
                              average='macro')
        recall = metrics.recall_score(t_targets_all, np.argmax(t_outputs_all, -1), labels=[0, 1, 2],
                                      average='macro')
        return total_loss / total_num, total_acc / total_num, recall, f1


    def eval(self, t, val_data_loader):
        return self.eval_withregx(t, val_data_loader)

    def evalx(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r=np.arange(x.size(0))
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=torch.autograd.Variable(x[b],volatile=True)
            targets=torch.autograd.Variable(y[b],volatile=True)

            # Forward
            outputs=self.model.forward(images)
            output=outputs[t]
            loss=self.criterion(t,output,targets)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy()[0]*len(b)
            total_acc+=hits.sum().data.cpu().numpy()[0]
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num

    def criterion(self,t,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        if t>0:
            startDateTimeOldLast = datetime.now()
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):

                loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
            print('DataTime OldLast', datetime.now() - startDateTimeOldLast)
            print("Compute loss for last model")

        print("Compute loss function")
        return self.ce(output,targets)+self.lamb*loss_reg


    #Serialize model, optimizer and other parameters to file
    def saveModel(self, topath):
         torch.save({
                'epoch': self.nepochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.ce,
                'learning_rate': self.lr,
                'batch':self.sbatch
            }, topath)

         return True


    #Unserialize model, optimizer and other parameters from file
    def loadModel(self, frompath):

        if not os.path.exists(frompath):
            return False
        else:
            checkpoint = torch.load(frompath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.ce = checkpoint['loss']
            return True


    def getMemoryRam(self):
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory use:', memoryUse)