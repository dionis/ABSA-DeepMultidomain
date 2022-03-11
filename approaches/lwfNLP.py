import sys,time,os
import numpy as np
from datetime import datetime

import psutil
import torch
from sklearn import metrics
from copy import deepcopy
import re
import utils

class Appr(object):
    """ Class implementing the Learning Without Forgetting approach described in https://arxiv.org/abs/1606.09282 """

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=100,lamb=2,T=1,args=None):
        self.model=model
        self.opt = args
        self.model_old=None

        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        ##modelVariables = self.model.get_bert_model_parameters()
        if self.model != None:
            self.task_size = 1 if self.model.taskcla == None else len(self.model.taskcla)

        modelVariables = self.model.named_parameters()

        self.tensorVariables = []
        self.tensorVariablesTuples = []
        for name, var in modelVariables:
            # print("Variable ==> " + name)
            self.tensorVariables.append(var)
            self.tensorVariablesTuples.append((name, var))


        self.ce=torch.nn.CrossEntropyLoss()
        print("!!!!New optmization!!!!!")

        optimizer = model.get_Optimizer()
        if optimizer != None:
            self._set_optimizer(optimizer)

        print("------New optmization--------")


        self.lamb=lamb          # Grid search = [0.1, 0.5, 1, 2, 4, 8, 10]; best was 2
        self.T=T                # Grid search = [0.5,1,2,4]; best was 1

        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])
            self.T=float(params[1])



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

        #return torch.optim.SGD(self.model.parameters(),lr=lr)



    def train(self, t, train_data_loader, test_data_loader, val_data_loader):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)

        task = torch.autograd.Variable(torch.LongTensor([t]).cuda(),
                                       volatile=False) if torch.cuda.is_available() else torch.autograd.Variable(
            torch.LongTensor([t]), volatile=False)
        # Loop epochs
        print(" ###### Update status of last layer weight in current task(domain) AVOID Stocastic Gradient ########")

        for name, var in self.model.named_parameters():
            if name.find("model.last.") != -1:
                var.requires_grad_(False);
                if re.match("model.last." + str(t), name) != None:
                    print("Variable " + name + " update to SGD")
                    var.requires_grad_(True);

        for e in range(self.nepochs):
            # Train
            clock0 = time.time()

            self.train_epochlwf(t, train_data_loader)

            clock1 = time.time()

            train_loss, train_acc, train_recall, train_f1, train_cohen_kappa  = self.evallwf(t,val_data_loader )

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
                100*train_cohen_kappa),
                  end='')

            # Valid
            valid_loss, valid_acc, valid_recall, valid_f1,valid_cohen_kappa= self.evallwf(t, test_data_loader )
            print(' Test: loss={:.3f}, acc={:5.1f}, f1={:5.1f}, cohen_kappa={:5.1f}%|'.format(valid_loss, 100 * valid_acc, 100 * valid_f1, 100*valid_cohen_kappa),
                  end='')

            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                        break
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()

        # Restore best
        utils.set_model_(self.model, best_model)

        # Update old
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old)  # Freeze the weights

        print(" ###### Show status of last layer weight in current task(domain) ########")
        toViewLasLayer = []
        for name, var in self.model.named_parameters():
            if name.find("model.last.") != -1:
                print("Requiere Grand ==> " + str(var.requires_grad))
                print("Variable name " + name + " == " + str(var.data))

                toViewLasLayer.append((name, var))

        return

    def train_epochlwf(self, t, train_data_loader, thres_cosh=50, thres_emb=6):
        self.model.train()

        # r = np.arange(x.size(0))
        # np.random.shuffle(r)
        # r = torch.LongTensor(r).cuda()
        loop_size = 0
        global_step = 0
        n_correct, n_total, loss_total = 0, 0, 0

        # Loop batches
        for i_batch, sample_batched in enumerate(train_data_loader):
            self.optimizer.zero_grad()

            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            # outputs = self.model(inputs)
            targets = sample_batched['polarity'].to(self.opt.device)

            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False) \
                if torch.cuda.is_available() \
                else torch.autograd.Variable(torch.LongTensor([t]), volatile=False)

            # Forward old model
            targets_old = None
            if t > 0:
                startDateTimeOld = datetime.now()
                targets_old,_ = self.model_old.forward(task, inputs)
                print('DataTime Old', datetime.now() - startDateTimeOld)


            # Forward current model
            startDateTime = datetime.now()
            outputs, _ = self.model.forward(task, inputs)
            print('Train DataTime', datetime.now() - startDateTime)
            print("Train forward")
            self.getMemoryRam()

            output = outputs[t]
            loss = self.criterion(t, targets_old, output, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)

            n_correct += (torch.argmax(output, -1) == targets).sum().item()
            n_total += len(output)
            loss_total += loss.item() * len(outputs)
            if global_step % self.opt.log_step == 0:
                # train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                # print('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
                print('loss: {:.4f}'.format(train_loss))

            self.optimizer.step()

        return

    def evallwf(self, t, val_data_loader):
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

            # Forward old model
            targets_old = None
            if t > 0:
                startDateTimeOld = datetime.now()
                targets_old,_ = self.model_old.forward(task, inputs)
                print('Train DataTime old task', datetime.now() - startDateTimeOld)

            # Forward
            startDateTime = datetime.now()
            outputs, _ = self.model.forward(task, inputs)
            print('Eval DataTime', datetime.now() - startDateTime)
            print ("Eval forward")
            self.getMemoryRam()

            output = outputs[t]
            loss = self.criterion(t, targets_old, output, targets)
            _, pred = output.max(1)
            hits = (pred == targets).float()

            n_correct += (torch.argmax(output, -1) == targets).sum().item()

            # Log
            current_batch_size = len(pred)
            total_loss += loss.data.cpu().numpy()  * current_batch_size
            total_acc += hits.sum().data.cpu().numpy()
            total_num += current_batch_size

            if t_targets_all is None:
                t_targets_all = targets.detach().numpy()
                t_outputs_all = output.detach().numpy()
            else:
                t_targets_all =  np.concatenate((t_targets_all, targets.detach().numpy()), axis=0)
                t_outputs_all =  np.concatenate((t_outputs_all, output.detach().numpy()), axis=0)


        # OJOOOO DEBEMOS REVISAR LAS LABELS [0,1,2] Deben corresponder a como las pone la implementacion
        ### FALTA LA ETIQUETA PARA CUANDO NO ES ASPECTO
        #global_output = t_outputs_all

        f1 = metrics.f1_score(t_targets_all, np.argmax(t_outputs_all, -1), labels=[0, 1, 2],
                                  average='macro')
        recall = metrics.recall_score(t_targets_all, np.argmax(t_outputs_all, -1), labels=[0, 1, 2],
                                          average='macro')

        cohen_kappa = metrics.cohen_kappa_score(t_targets_all, np.argmax(t_outputs_all, -1))

        return total_loss / total_num, total_acc / total_num, recall, f1, cohen_kappa


        #return total_loss / total_num, total_acc / total_num, 0, 0


    def eval(self, t, val_data_loader):
        return self.evallwf(t, val_data_loader)

    def criterion(self,t,targets_old,outputs,targets):
        # TODO: warm-up of the new layer (paper reports that improves performance, but negligible)

        # Knowledge distillation loss for all previous tasks
        loss_dist=0


        # for t_old in range(0,t):
        #     loss_dist+=utils.cross_entropy(outputs[t_old],targets_old[t_old],exp=1/self.T)
        #
        # # Cross entropy loss
        # loss_ce=self.ce(outputs[t],targets)
        for t_old in range(0,t):
            loss_dist+=utils.cross_entropy(outputs,targets_old[t_old],exp=1/self.T)

        # Cross entropy loss
        loss_ce=self.ce(outputs,targets)

        # We could add the weight decay regularization mentioned in the paper. However, this might not be fair/comparable to other approaches
        print("Loss evaluation")
        return loss_ce+self.lamb*loss_dist

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



    def getMemoryRam(self):
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory use:', memoryUse)