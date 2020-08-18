import sys,time, os
import numpy as np
import torch
from copy import deepcopy
import utils
from datetime import datetime
import psutil
from sklearn import metrics
from torch.autograd import Variable

########################################################################################################################

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
        self.update_big_omega_ops = []

        # Parameters for the intelligence synapses model.
        self.param_c = 0.1
        self.param_xi = 0.1

        self.learning_rate = 0.001
        self.exp_pow = torch.tensor(2)
        self.exp_pow = 2

        # modelVariables = [(name, var) for i, (name, var) in enumerate(self.model.named_parameters()) if
        #                   name.find("bert") == -1]

        modelVariables = [(name, var) for i, (name, var) in enumerate(self.model.named_parameters()) ]

        self.tensorVariables = []
        self.tensorVariablesTuples = []
        for name, var in modelVariables:
            #print("Variable ==> " + name)
            self.tensorVariables.append( var)
            self.tensorVariablesTuples.append((name, var))

        list_variables = list(self.tensorVariablesTuples)
        for name, var in list_variables:
            self.small_omega_var[name] = Variable(torch.zeros(var.shape), requires_grad=False)
            self.previous_weights_mu_minus_1[name] = Variable(torch.zeros(var.shape), requires_grad=False)
            self.big_omega_var[name] = Variable(torch.zeros(var.shape), requires_grad=False)

        print("!!!!New optmization!!!!!")

        optimizer = model.get_Optimizer()
        if optimizer != None:
            self._set_optimizer(optimizer)

        print("------New optmization--------")

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

    def _set_optimizer(self, _new_optimize):
        if _new_optimize != None: self.optimizer = _new_optimize

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr

        print("!!!!New optmization!!!!!")
        if self.optimizer != None:
            print("--------Optmization---------")
            return self.optimizer

        return torch.optim.SGD(self.tensorVariables,lr=lr)

    def update_big_omega(self, list_variables, previous_weights_mu_minus_1, small_omega_var):
        big_omega_var = {}
        for name, var in list_variables:
            big_omega_var[name] = torch.div(self.small_omega_var[name], (self.param_xi +
                                                                    torch.pow(
                                                                        (var.data - self.previous_weights_mu_minus_1[name]),
                                                                        self.exp_pow)))

        return (big_omega_var)

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

            self.train_epochesi(t, train_data_loader)

            clock1 = time.time()

            train_loss, train_acc, train_recall, train_f1 = self.eval_withregsi(t, test_data_loader)

            clock2 = time.time()

            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e + 1,
                                                                                                        1000 * self.sbatch * (
                                                                                                            clock1 - clock0) / train_data_loader.__len__( ),
                                                                                                        1000 * self.sbatch * (
                                                                                                            clock2 - clock1) / train_data_loader.__len__( ),
                                                                                                        train_loss,
                                                                                                        100 * train_acc),
                  end='')

            # Valid
            valid_loss, valid_acc, valid_recall, valid_f1 = self.eval_withregsi(t, val_data_loader)
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
            self.big_omega_var = self.update_big_omega(self.tensorVariablesTuples, self.previous_weights_mu_minus_1,
                                                       self.small_omega_var)

            for i, (name, var) in enumerate(self.tensorVariablesTuples):
                self.previous_weights_mu_minus_1[name] = var.data
                self.small_omega_var[name] = 0.0
        return


    def train_epochesi(self, t, train_data_loader, thres_cosh=50,thres_emb=6):
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
            outputs,_=self.model.forward(task,inputs)
            print('Train DataTime', datetime.now() - startDateTime)
            print("Train forward")
            self.getMemoryRam()

            output=outputs[t]
            loss=self.criterion(t,output,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)

            # for name, var in self.tensorVariablesTuples:  Esto da error
            #     self.small_omega_var[name] -= self.lr * var.grad  # small_omega -= delta_weight(t)*gradient(t)

            self.optimizer.step()

        return


    def eval_withregsi(self, t, val_data_loader):
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
            outputs,_ = self.model.forward(task,inputs)

            print('Eval DataTime', datetime.now() - startDateTime)
            print("Eval forward")
            self.getMemoryRam()

            output = outputs[t]
            loss = self.criterion(t, output, targets)
            _, pred = output.max(1)
            hits = (pred == targets).float()

            # Log
            total_loss += loss.data.cpu().numpy() * sample_batched.__len__()
            total_acc += hits.sum().data.cpu().numpy()
            total_num += len(sample_batched)

            if t_targets_all is None:
                t_targets_all = targets.detach().numpy()
                t_outputs_all = output.detach().numpy()
            else:
                t_targets_all =  np.concatenate((t_targets_all, targets.detach().numpy()), axis=0)
                t_outputs_all =  np.concatenate((t_outputs_all, output.detach().numpy()), axis=0)

        #OJOOOO DEBEMOS REVISAR LAS LABELS [0,1,2] Deben corresponder a como las pone la implementacion
        #### FALTA LA ETIQUETA PARA CUANDO NO ES ASPECTO

        #global_output = t_outputs_all.detach()
        f1 = metrics.f1_score(t_targets_all, np.argmax(t_outputs_all, -1), labels=[0, 1, 2],
                              average='macro')

        recall = metrics.recall_score(t_targets_all, np.argmax(t_outputs_all, -1), labels=[0, 1, 2], average='macro')

        return total_loss / total_num, total_acc / total_num, recall, f1

    def eval(self, t, test_data_loader):
        return self.eval_withregsi(t, test_data_loader)

    def criterion(self,t,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        if t>0:
            for name, var in self.tensorVariablesTuples:
                loss_reg += torch.sum(torch.mul( self.big_omega_var[name], (self.previous_weights_mu_minus_1[name] - var.data).pow(self.exp_pow)))

            # for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
            #     loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2

        return self.ce(output,targets)+ self.param_c * loss_reg

########################################################################################################################
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