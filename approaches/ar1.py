import sys,time,os
import numpy as np
import torch
from copy import deepcopy
import utils
from datetime import datetime
import psutil
import re
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


        if self.model != None:

          self.task_size = 1 if self.model.taskcla == None else len(self.model.taskcla)
          self.wpelaty = torch.Tensor(torch.zeros(self.task_size))

          #Values taken from ""

        #modelVariables = self.model.get_bert_model_parameters()

        self.tensorVariables = []
        self.tensorVariablesTuples = []
        # for name, var in modelVariables:
        #     #print("Variable ==> " + name)
        #     self.tensorVariables.append(var)
        #     self.tensorVariablesTuples.append((name, var))

        print("!!!!New optmization!!!!!")
        for name, var in self.model.named_parameters():
            #print("Variable ==> " + name)
            self.tensorVariables.append(var)
            self.tensorVariablesTuples.append((name, var))
        optimizer = model.get_Optimizer()
        if optimizer != None:
            self._set_optimizer(optimizer)

        print("------New optmization--------")

        list_variables = list(self.tensorVariablesTuples)
        for name, var in list_variables:
            self.small_omega_var[name] = Variable(torch.zeros(var.shape), requires_grad=False)
            self.previous_weights_mu_minus_1[name] = Variable(torch.zeros(var.shape), requires_grad=False)
            self.big_omega_var[name] = Variable(torch.zeros(var.shape), requires_grad=False)


        self.optimizer = self._get_optimizer()
        self.clear_tmp_outputlayer()
        self.current_task = -1

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
        if lr is None: lr=self.lr

        print("!!!!New optmization!!!!!")
        # if self.optimizer != None:
        #     print("--------Optmization---------")
        #     return self.optimizer

        #return torch.optim.SGD(self.tensorVariables, lr=lr)
        #return torch.optim.SGD(self.model.parameters(),lr=lr)
        return self.opt.optimizer(self.model.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

    def update_big_omega(self, list_variables, previous_weights_mu_minus_1, small_omega_var):
        big_omega_var = {}
        for i, (name, var) in enumerate(list_variables):
            big_omega_var[name] = torch.div(self.small_omega_var[name], (self.param_xi +

                                                                    torch.pow(
                                                                        (var.data - self.previous_weights_mu_minus_1[name]),
                                                                        self.exp_pow)))

        return (big_omega_var)

    def clear_tmp_outputlayer(self): #Set 0 in all parameters
        # reset_parameters()

        ###
        ### Inicial todos los pesos a 0 segun algoritmo del paper
        ###
        self.model.tm = torch.nn.Linear( self.opt.polarities_dim, self.opt.polarities_dim)

        #Initialice 0 because reset_parametes input other standar initialization

        neuronsize = self.model.tm.weight.shape[0]
        input_neuron_size = self.model.tm.weight.shape[1]

        for i in range(neuronsize):
            for j in range(input_neuron_size):
                self.model.tm.weight.data[i, j] = 0.0

            self.model.tm.bias.data[i] = 0.0

        return

    def train(self, t, train_data_loader, test_data_loader, val_data_loader):
        best_loss=np.inf
        #best_model=utils.get_model(self.model)
        lr=self.lr
        patience=self.lr_patience


        task = torch.autograd.Variable(torch.LongTensor([t]).cuda(),
                                       volatile=False, requires_grad=False) if torch.cuda.is_available() \
                                     else torch.autograd.Variable(torch.LongTensor([t]), volatile=False, requires_grad=False)

        print(" ###### Update status of last layer weight in current task(domain) AVOID Stocastic Gradient ########")

        for name, var in self.model.named_parameters():
          if name.find("model.last.") != -1:
                var.requires_grad_(False);
                if re.match("model.last." + str(t), name) != None:
                    print("Variable " + name + " update to SGD")
                    var.requires_grad_(True);





        if t != self.current_task:
            ###It need that al weights in last output layer are inicialized in zero
            ###Optimization in original paper
            ###no usal la inicializacion Gaussiana y de Xavier. Aunque se conoce que los pesos de las
            ###redes no deben inicializarce a 0 pero esto es para niveles intermedios y no para los niveles
            ###de salida
            self.clear_tmp_outputlayer()
            self.current_task = t

        ##
        ##  LA VARIABLE tm se coloca entre los valores a optimizar??????
        ##
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
            train_loss, train_acc, train_recall, train_f1, train_cohen_kappa = self.eval_withregsi(t,val_data_loader )

            #print("3")
            clock2 = time.time()

            dataset_size = len(val_data_loader.dataset)
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train-Val: loss={:.3f}, acc={:5.1f}, f1={:5.1f}, cohen_kappa={:5.1f}%|'.format(e + 1,
                                                                                                        1000 * self.sbatch * (
                                                                                                            clock1 - clock0) / dataset_size,
                                                                                                        1000 * self.sbatch * (
                                                                                                            clock2 - clock1) / dataset_size,
                                                                                                        train_loss,
                                                                                                        100 * train_acc,
                                                                                                        100*train_f1,
                                                                                                        100 * train_cohen_kappa ),

                  end='')

            # Valid
            #print("4")
            valid_loss, valid_acc , valid_recall, valid_f1, valid_cohen_kappa= self.eval_withregsi(t, test_data_loader)

            print(' Test: loss={:.3f}, acc={:5.1f}, f1={:5.1f}, cohen_kappa={:5.1f}%|'.format(valid_loss, 100 * valid_acc,100*valid_f1, 100*valid_cohen_kappa), end='')

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

        # Restore best
        #utils.set_model_(self.model,best_model)

        # Update old
        # self.model_old=deepcopy(self.model)
        # self.model_old.eval()
        # utils.freeze_model(self.model_old) # Freeze the weights

        # Fisher ops
        if t>0:
            # After each task is complete, call update_big_omega and reset_small_omega
            # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
            self.big_omega_var = self.update_big_omega(self.tensorVariablesTuples, self.previous_weights_mu_minus_1,
                                                       self.small_omega_var)

            for i, (name, var) in enumerate(self.tensorVariablesTuples):
                self.previous_weights_mu_minus_1[name] = var.data
                self.small_omega_var[name] = 0.0



         ###### Show status of last layer weight in current task(domain) ########
        print(" ###### Show status of last layer weight in current task(domain) ########")
        toViewLasLayer = []
        for name, var in self.model.named_parameters():
          if name.find("model.last.") != -1:
               print ("Requiere Grand ==> " + str(var.requires_grad))
               print ("Variable name " + name + " == " + str(var.data))

               toViewLasLayer.append((name,var))


        return

    def trainsi(self,t,xtrain,ytrain,xvalid,yvalid):
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


        # After each task is complete, call update_big_omega and reset_small_omega
        # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
        self.big_omega_var = self.update_big_omega(self.model.named_parameters(), self.previous_weights_mu_minus_1, self.small_omega_var)

        for i, (name, var) in enumerate(self.model.named_parameters()):
            self.previous_weights_mu_minus_1[name] = var.data
            self.small_omega_var[name] = 0.0

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

    def train_epochesi(self, t, train_data_loader, thres_cosh=50,thres_emb=6):
        self.model.train()
        # Loop batches

        loop_size = 0
        global_step = 0
        n_correct, n_total, loss_total = 0, 0, 0

        for i_batch, sample_batched in enumerate(train_data_loader):
            global_step += 1
            #print("Batch size: " + str (sample_batched.__len__()))
            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            # outputs = self.model(inputs)
            targets = sample_batched['polarity'].to(self.opt.device)


            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False, requires_grad=False) \
                if torch.cuda.is_available() \
                else torch.autograd.Variable(torch.LongTensor([t]), volatile=False, requires_grad=False)


            # Forward current model
            startDateTime = datetime.now()
            outputs,_=self.model.forward( task, inputs)
           # print('Train DataTime', datetime.now() - startDateTime)
           # print("Train forward")
            self.getMemoryRam()
            output= self.model.tm(outputs[t])
            #output = outputs[t]
            loss=self.criterion(t,output,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            #print("1.6 ")
            torch.nn.utils.clip_grad_norm_(self.tensorVariables,self.clipgrad)


            n_correct += (torch.argmax(output, -1) == targets).sum().item()
            n_total += len(output)
            loss_total += loss.item() * len(outputs)
            if global_step % self.opt.log_step == 0:
                # train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                #print('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
                print('loss: {:.4f}'.format(train_loss))

            # for i, (name, var) in enumerate(self.tensorVariables):
            #     self.small_omega_var[name] -= self.lr * var.grad  # small_omega -= delta_weight(t)*gradient(t)
            self.optimizer.step()

        self.update_global_layer(t)
        #print("1.8 ")
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
            #print ("Eval forward")
            self.getMemoryRam()
            #output = self.model.tm(outputs[t])
            output = outputs[t]
            loss = self.criterion(t, output, targets)
            _, pred = output.max(1)
            hits = (pred == targets).float()

            n_correct += (torch.argmax(output, -1) == targets).sum().item()

            # Log
            current_batch_size = len(pred)
            total_loss += loss.data.cpu().numpy()*current_batch_size
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
        #global_output = t_outputs_all
        f1 = metrics.f1_score(t_targets_all, np.argmax(t_outputs_all, -1), labels=[0, 1, 2], average='macro')
        recall = metrics.recall_score(t_targets_all, np.argmax(t_outputs_all, -1), labels=[0, 1, 2],
                              average='macro')

        #Reference https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
        #          https://scikit-learn.org/stable/modules/model_evaluation.html#cohen-kappa


        cohen_kappa = metrics.cohen_kappa_score(t_targets_all,np.argmax(t_outputs_all, -1))

        return total_loss / total_num, total_acc / total_num, recall, f1, cohen_kappa

    ##
    #   Clasify Aspect in Sentence/ Production method
    ##

    def eval_classify(self, t, val_data_loader):
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
            #print ("Eval forward")
            self.getMemoryRam()

            output = outputs[t]
            loss = self.criterion(t, output, targets)
            _, pred = output.max(1)
            hits = (pred == targets).float()

            n_correct += (torch.argmax(output, -1) == targets).sum().item()

            # Log
            current_batch_size = len(pred)
            total_loss += loss.data.cpu().numpy()*current_batch_size
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
        #global_output = t_outputs_all
        f1 = metrics.f1_score(t_targets_all, np.argmax(t_outputs_all, -1), labels=[0, 1, 2], average='macro')
        recall = metrics.recall_score(t_targets_all, np.argmax(t_outputs_all, -1), labels=[0, 1, 2],
                              average='macro')
        #return t_outputs_all , total_loss / total_num, total_acc / total_num, recall, f1
        return pred
###-------------------------------------------------------------------------------------------------------------
    def eval(self, t, test_data_loader):
        return self.eval_withregsi(t, test_data_loader)

    def criterion(self,t,output,targets):
        # Regularization for all previous tasks
        loss_reg=0

        #
        # for name, var in self.model.named_parameters():
        #     print ("Variable: ", name)

        if t>0:
            for name, var in self.tensorVariablesTuples:
                loss_reg += torch.sum(torch.mul( self.big_omega_var[name], (self.previous_weights_mu_minus_1[name] - var.data).pow(self.exp_pow)))

            # for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
            #     loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
        # + self.param_c * loss_reg
        return self.ce(output,targets) + self.param_c * loss_reg

    def tm_output_layer(self, param):
        return self.tm(param)

    def update_global_layer(self, t):
        neuronsize = self.model.tm.weight.shape[0]
        input_neuron_size = self.model.tm.weight.shape[1]

        scalar = torch.mean(self.model.tm.weight)
        for i in range(neuronsize):
            for j in range(input_neuron_size):
                self.model.last[t].weight.data[i, j] = self.model.tm.weight.data[i, j] - scalar.item()

            self.model.last[t].bias.data[i] = self.model.tm.bias.data[i] - scalar.item()

        return

########################################################################################################################
    # Serialize model, optimizer and other parameters to file
    def saveModel(self, topath):
        torch.save({
            'epoch': self.nepochs,
            'model_state_dict': self.model.get_Model().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.ce,
            'learning_rate': self.lr,
            'batch': self.sbatch,
            'task_size':self.task_size
        }, topath)

        return True



    # Unserialize model, optimizer and other parameters from file
    def loadModel(self, frompath):
        if not os.path.exists(frompath):
            return False
        else:
            checkpoint = torch.load(frompath)
            self.model.get_Model().load_state_dict(checkpoint['model_state_dict'])

            self.optimizer = self.opt.optimizer(filter(lambda p: p.requires_grad,  self.model.parameters()), lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.ce = checkpoint['loss']
            return True

    def getMemoryRam(self):
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        #print('memory use:', memoryUse)

    ####
    #
    #    Apply clasification methods in a sentences set
    #
    ####
    def classify(self, t,  test_data_loader):
        best_loss = np.inf
        # best_model=utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience

        task = torch.autograd.Variable(torch.LongTensor([t]).cuda(),
                                       volatile=False, requires_grad=False) if torch.cuda.is_available() \
            else torch.autograd.Variable(torch.LongTensor([t]), volatile=False, requires_grad=False)

        print(" ###### Update status of last layer weight in current task(domain) AVOID Stocastic Gradient ########")



        if t != self.current_task:
            ###It need that al weights in last output layer are inicialized in zero
            ###Optimization in original paper
            ###no usal la inicializacion Gaussiana y de Xavier. Aunque se conoce que los pesos de las
            ###redes no deben inicializarce a 0 pero esto es para niveles intermedios y no para los niveles
            ###de salida
            self.clear_tmp_outputlayer()
            self.current_task = t

        ##
        ##  LA VARIABLE tm se coloca entre los valores a optimizar??????
        ##
        self.optimizer = self._get_optimizer(lr)
        # Loop epochs

        t_targest_all, train_loss, train_acc, train_recall, train_f1 = self.eval_classify(t, test_data_loader)


        return  (t_targest_all, train_loss, train_acc, train_recall, train_f1)
