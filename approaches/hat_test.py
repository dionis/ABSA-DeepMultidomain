import sys,time, os
import numpy as np
from datetime import datetime
import psutil
import torch
from copy import deepcopy
from sklearn import metrics
import utils

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

        optimizer = self.model.get_Optimizer()
        if optimizer != None:
            self._set_optimizer(optimizer)

        self.lamb=lamb
        self.smax=smax
        self.logpath = None
        self.single_task = False
        self.logpath = args.parameter
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

        bert_variables_name = self.model.named_parameters()
        self.tensorVariables = []
        self.tensorVariablesTuples = []
        for name, var in bert_variables_name:
            #print("Variable ==> " + name)
            self.tensorVariables.append(var)
            self.tensorVariablesTuples.append((name, var))
        # return torch.optim.SGD(self.model.parameters(),lr=lr)

        return torch.optim.SGD(self.tensorVariables, lr=lr)

    def train_epochx(self,t, train_data_loader, thres_cosh=50,thres_emb=6):
        self.model.train()

        # r=np.arange(x.size(0))
        # np.random.shuffle(r)
        #r=torch.LongTensor(r).cuda()

        # Loop batches

        for i_batch, sample_batched in enumerate(train_data_loader):

            # clear gradient accumulators
            self.optimizer.zero_grad()

            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            #outputs = self.model(inputs)
            targets = sample_batched['polarity'].to(self.opt.device)


            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False)\
                                                           if torch.cuda.is_available() \
                                                           else torch.autograd.Variable(torch.LongTensor([t]),volatile=False)

            s=(self.smax-1/self.smax)*i_batch/len(inputs[0])+1/self.smax

            # Forward
            startDateTime = datetime.now()
            outputs,masks = self.model(task,inputs,s)
            print('Train DataTime', datetime.now() - startDateTime)
            print("Train forward")
            self.getMemoryRam()


            output = outputs[t]
            loss,_ = self.criterion(t,output,targets,masks)
            loss.backward()
            self.optimizer.step()

            # outputs,masks=self.model.forward(task,images,s=s)
            # output=outputs[t]
            # loss,_=self.criterion(output,targets,masks)

            # Backward
            # self.optimizer.zero_grad()
            # loss.backward()

            # Restrict layer gradients in backprop
            if t>0:
                for n,p in self.tensorVariablesTuples:
                     if n in self.mask_back:
                        p.grad.data*=self.mask_back[n]

            # Compensate embedding gradients
            for n,p in self.tensorVariablesTuples:
                if n.startswith('model.e'):
                    num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den

            # Apply step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

            # Constrain embeddings
            for n,p in self.model.named_parameters():
                if n.startswith('e'):
                    p.data=torch.clamp(p.data, -thres_emb, thres_emb)

            #print(masks[-1].data.view(1,-1))
            #if i>=5*self.sbatch: sys.exit()
            #if i==0: print(masks[-2].data.view(1,-1),masks[-2].data.max(),masks[-2].data.min())
        #print(masks[-2].data.view(1,-1))

        return

    def train(self, t, train_data_loader, test_data_loader, val_data_loader):
        best_loss = np.inf
        #best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)

        # log
        losses_train = []
        losses_valid = []
        acc_train = []
        acc_valid = []
        reg_train = []
        reg_valid = []
        self.logs['mask'][t] = {}
        self.logs['mask_pre'][t] = {}

        task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False) if torch.cuda.is_available() else torch.autograd.Variable(torch.LongTensor([t]), volatile=False)
        bmask = self.model.mask(task, s=self.smax)
        for i in range(len(bmask)):
            bmask[i] = torch.autograd.Variable(bmask[i].data.clone(), requires_grad=False)
            self.logs['mask'][t][i] = {}
            self.logs['mask'][t][i][-1] = deepcopy(bmask[i].data.cpu().numpy().astype(np.float32))
            if t == 0:
                self.logs['mask_pre'][t][i] = deepcopy((0 * bmask[i]).data.cpu().numpy().astype(np.float32))
            else:
                self.logs['mask_pre'][t][i] = deepcopy(self.mask_pre[i].data.cpu().numpy().astype(np.float32))

        if not self.single_task or (self.single_task and t == 0):
            # Loop epochs
            try:
                for e in range(self.nepochs):
                    # Train
                    clock0 = time.time()

                    self.train_epochx(t, train_data_loader)

                    clock1 = time.time()

                    train_loss, train_acc, train_recall, train_f1 = self.eval_withregx(t, test_data_loader)

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

                    # log
                    losses_train.append(train_loss)
                    acc_train.append(train_acc)
                    #reg_train.append(train_reg)
                    losses_valid.append(valid_loss)
                    acc_valid.append(valid_acc)
                    # reg_valid.append(valid_reg)

                    # Adapt lr
                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        #best_model = utils.get_model(self.model)
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

                    # Log activations mask
                    task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False) \
                        if torch.cuda.is_available() \
                        else torch.autograd.Variable(torch.LongTensor([t]), volatile=False)

                    bmask = self.model.mask(task, s=self.smax)
                    for i in range(len(bmask)):
                        self.logs['mask'][t][i][e] = deepcopy(bmask[i].data.cpu().numpy().astype(np.float32))

                # Log losses
                if self.logs is not None:
                    self.logs['train_loss'][t] = np.array(losses_train)
                    self.logs['train_acc'][t] = np.array(acc_train)
                    #self.logs['train_reg'][t] = np.array(reg_train)
                    self.logs['valid_loss'][t] = np.array(losses_valid)
                    self.logs['valid_acc'][t] = np.array(acc_valid)
                    self.logs['valid_reg'][t] = np.array(reg_valid)
            except KeyboardInterrupt:
                print()

        # Restore best validation model
        #utils.set_model_(self.model, best_model)

        # Activations mask
        task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False) \
            if torch.cuda.is_available() \
            else torch.autograd.Variable(torch.LongTensor([t]), volatile=False)

        mask = self.model.mask(task, s=self.smax)
        for i in range(len(mask)):
            mask[i] = torch.autograd.Variable(mask[i].data.clone(), requires_grad=False)
        if t == 0:
            self.mask_pre = mask
        else:
            for i in range(len(self.mask_pre)):
                self.mask_pre[i] = torch.max(self.mask_pre[i], mask[i])

        # Weights mask
        self.mask_back = {}
        for n, _ in self.tensorVariablesTuples:
            vals = self.model.get_view_for(n, self.mask_pre)
            if vals is not None:
                self.mask_back[n] = 1 - vals
        return

    # def train_epoch(self,t,x,y,thres_cosh=50,thres_emb=6):
    #     self.model.train()
    #
    #     r=np.arange(x.size(0))
    #     np.random.shuffle(r)
    #     r=torch.LongTensor(r)
    #
    #     # Loop batches
    #     for i in range(0,len(r),self.sbatch):
    #         if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
    #         else: b=r[i:]
    #         images=torch.autograd.Variable(x[b],volatile=False)
    #         targets=torch.autograd.Variable(y[b],volatile=False)
    #         task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
    #         s=(self.smax-1/self.smax)*i/len(r)+1/self.smax
    #
    #         # Forward
    #         outputs,masks=self.model.forward(task,images,s=s)
    #         output=outputs[t]
    #         loss,_=self.criterion(output,targets,masks)
    #
    #         # Backward
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #
    #         # Restrict layer gradients in backprop
    #         if t>0:
    #             for n,p in self.model.named_parameters():
    #                 if n in self.mask_back:
    #                     p.grad.data*=self.mask_back[n]
    #
    #         # Compensate embedding gradients
    #         for n,p in self.model.named_parameters():
    #             if n.startswith('e'):
    #                 num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
    #                 den=torch.cosh(p.data)+1
    #                 p.grad.data*=self.smax/s*num/den
    #
    #         # Apply step
    #         torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
    #         self.optimizer.step()
    #
    #         # Constrain embeddings
    #         for n,p in self.model.named_parameters():
    #             if n.startswith('e'):
    #                 p.data=torch.clamp(p.data,-thres_emb,thres_emb)
    #
    #         #print(masks[-1].data.view(1,-1))
    #         #if i>=5*self.sbatch: sys.exit()
    #         #if i==0: print(masks[-2].data.view(1,-1),masks[-2].data.max(),masks[-2].data.min())
    #     #print(masks[-2].data.view(1,-1))
    #
    #     return

    # def evalx(self,t,x,y):
    #     total_loss=0
    #     total_acc=0
    #     total_num=0
    #     self.model.eval()
    #
    #     total_reg=0
    #
    #     r=np.arange(x.size(0))
    #     r=torch.LongTensor(r).cuda()
    #
    #     # Loop batches
    #     for i in range(0,len(r),self.sbatch):
    #         if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
    #         else: b=r[i:]
    #         images=torch.autograd.Variable(x[b],volatile=True)
    #         targets=torch.autograd.Variable(y[b],volatile=True)
    #         task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=True)
    #
    #         # Forward
    #         factor=1
    #         if self.single_task: factor=10000
    #         outputs,masks=self.model.forward(task,images,s=factor*self.smax)
    #         output=outputs[t]
    #         loss,reg=self.criterion(output,targets,masks)
    #         _,pred=output.max(1)
    #         hits=(pred==targets).float()
    #
    #         # Log
    #         total_loss+=loss.data.cpu().numpy().item()*len(b)
    #         total_acc+=hits.sum().data.cpu().numpy().item()
    #         total_num+=len(b)
    #         total_reg+=reg.data.cpu().numpy().item()*len(b)
    #
    #     print('  {:.3f}  '.format(total_reg/total_num),end='')
    #
    #     return total_loss/total_num,total_acc/total_num

    def eval_withregx(self, t, val_data_loader):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        total_reg = 0

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
            factor = 1
            if self.single_task: factor = 10000

            startDateTime = datetime.now()
            outputs, masks = self.model.forward(task, inputs, s=factor * self.smax)
            print('Eval DataTime', datetime.now() - startDateTime)
            print ("Eval forward")
            self.getMemoryRam()

            output = outputs[t]
            #output = outputs
            loss, reg = self.criterion(t,output, targets, masks)
            _, pred = output.max(1)
            hits = (pred == targets).float()

            # Log
            total_loss += loss.data.cpu().numpy().item() * sample_batched.__len__()
            total_acc += hits.sum().data.cpu().numpy().item()
            total_num += len(sample_batched)
            total_reg += reg.data.cpu().numpy().item() * sample_batched.__len__()

            if t_targets_all is None:
                t_targets_all = targets.detach().numpy()
                t_outputs_all = output.detach().numpy()
            else:
                t_targets_all = np.concatenate((t_targets_all, targets.detach().numpy()), axis=0)
                t_outputs_all = np.concatenate((t_outputs_all, output.detach().numpy()), axis=0)

        # global_output = t_outputs_all.detach()
        f1 = metrics.f1_score(t_targets_all, np.argmax(t_outputs_all, -1), labels=[0, 1, 2], average='macro')
        recall = metrics.recall_score(t_targets_all, np.argmax(t_outputs_all, -1), labels=[0, 1, 2],
                                      average='macro')

        print('  {:.3f}  '.format(total_reg / total_num), end='')
        return total_loss / total_num, total_acc / total_num, recall, f1

    def eval(self, t, val_data_loader):
        return self.eval_withregx(t, val_data_loader)

    def eval_withreg(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        total_reg=0

        r=np.arange(x.size(0))
        r=torch.LongTensor(r)

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=torch.autograd.Variable(x[b],volatile=True)
            targets=torch.autograd.Variable(y[b],volatile=True)
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=True)

            # Forward
            factor=1
            if self.single_task: factor=10000
            outputs,masks=self.model.forward(task,images,s=factor*self.smax)
            output=outputs[t]
            loss,reg=self.criterion(t,output,targets,masks)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy().item()*len(b)
            total_acc+=hits.sum().data.cpu().numpy().item()
            total_num+=len(b)
            total_reg+=reg.data.cpu().numpy().item()*len(b)

        print('  {:.3f}  '.format(total_reg/total_num),end='')

        return total_loss/total_num,total_acc/total_num,total_reg/total_num

    def criterion(self,t,outputs,targets,masks):
        reg=0
        count=0
        if self.mask_pre is not None:
            for m,mp in zip(masks,self.mask_pre):
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m in masks:
                reg+=m.sum()
                count+=np.prod(m.size()).item()
        reg/=count
        return self.ce(outputs,targets)+self.lamb*reg,reg

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