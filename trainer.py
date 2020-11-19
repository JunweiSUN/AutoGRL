import torch
import torch.nn.functional as F
from copy import deepcopy
from evaluator import Evaluator
import torch.optim.swa_utils
import numpy as np
from tensorboardX import SummaryWriter
import random

class EarlyStoppingTrainer:
    def __init__(self, args, model, eval_steps=1):
        self.epochs = args.epochs
        self.lr = args.lr
        self.eval_steps = eval_steps
        if args.optim == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError('Not supported optimizer')
        
        self.model = model
    
    def train(self, data):
        writer = SummaryWriter('earlystop')
        evaluator = Evaluator(metric='acc')
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        best_val_acc = 0
        best_model_parameters = None
        y_all = data.y.squeeze()
        y_train = y_all[data.train_mask]
        y_valid = y_all[data.val_mask]
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(data)[data.train_mask]
            loss = F.nll_loss(out, y_train)
            writer.add_scalar('train_loss', loss.item(), global_step=epoch)
            writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
            loss.backward()
            self.optimizer.step()
            # scheduler.step()

            if epoch % self.eval_steps == 0:
                logits = self.inference(data)
                val_pred = logits[data.val_mask].max(1)[1]
                acc = evaluator(y_valid, val_pred)
                writer.add_scalar('val_acc', acc, global_step=epoch)
                if acc > best_val_acc:
                    best_val_acc = acc
                    best_model_parameters = deepcopy(self.model.state_dict())

        if best_model_parameters:
            self.model.load_state_dict(best_model_parameters)
    
    @torch.no_grad()
    def inference(self, data):
        self.model.eval()
        return self.model(data)

class SWAEnsembleTrainer:
    def __init__(self, args, model):
        self.epochs = args.epochs
        self.lr = args.lr
        if args.optim == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError('Not supported optimizer')

        self.model = model
        self.swa_model = torch.optim.swa_utils.AveragedModel(model)
    
    def train(self, data):
        writer = SummaryWriter('swa')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        swa_scheduler = torch.optim.swa_utils.SWALR(self.optimizer, anneal_strategy="linear", anneal_epochs=5, swa_lr=self.lr)
        swa_start = self.epochs // 2 + 50

        self.model.train()
        y_train = data.y.squeeze()[data.train_mask]
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.model(data)[data.train_mask]
            loss = F.nll_loss(out, y_train)
            writer.add_scalar('train_loss', loss.item(), global_step=epoch)
            writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
            loss.backward()
            self.optimizer.step()

            if epoch > swa_start:
                self.swa_model.update_parameters(self.model)
                swa_scheduler.step()
            else:
                scheduler.step()

    @torch.no_grad()
    def inference(self, data):
        self.swa_model.eval()
        return self.swa_model(data)

class SnapshotEnsembleTrainer:
    def __init__(self, args, model, interval=100):
        self.epochs = args.epochs
        self.lr = args.lr
        self.interval = interval
        if args.optim == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError('Not supported optimizer')
        
        self.model = model
    
    def train(self, data):
        writer = SummaryWriter('snapshot')
        scheduler_fn = lambda epoch: 1/2 * (np.cos(np.pi * (epoch % self.interval) / self.interval) + 1) # epoch start from 0
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=scheduler_fn)

        self.snapshots = []
        self.model.train()
        
        y_train = data.y.squeeze()[data.train_mask]
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.model(data)[data.train_mask]
            loss = F.nll_loss(out, y_train)
            writer.add_scalar('train_loss', loss.item(), global_step=epoch)
            writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
            loss.backward()
            self.optimizer.step()
            scheduler.step()

            if (epoch + 1) % self.interval == 0:
                self.snapshots.append(deepcopy(self.model.state_dict()))
    
    @torch.no_grad()
    def inference(self, data):
        logits = []
        for snapshot in self.snapshots:
            self.model.load_state_dict(snapshot)
            self.model.eval()
            logits.append(self.model(data))
        logits = logits[-10:]
        return sum(logits) / len(logits)

class PlateauTrainer:
    def __init__(self, args, model, eval_steps=1):
        self.epochs = args.epochs
        self.lr = args.lr
        self.eval_steps = eval_steps
        if args.optim == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError('Not supported optimizer')
   
        self.model = model
    
    def train(self, data):
        writer = SummaryWriter('plateau')
        evaluator = Evaluator(metric='acc')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5)
        best_val_acc = 0
        best_model_parameters = None
        y_all = data.y.squeeze()
        y_train = y_all[data.train_mask]
        y_valid = y_all[data.val_mask]
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(data)[data.train_mask]
            loss = F.nll_loss(out, y_train)
            writer.add_scalar('train_loss', loss.item(), global_step=epoch)
            writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
            loss.backward()
            self.optimizer.step()

            logits = self.inference(data)
            val_pred = logits[data.val_mask].max(1)[1]
            acc = evaluator(y_valid, val_pred)
            val_loss = F.nll_loss(logits[data.val_mask], y_valid)
            writer.add_scalar('val_acc', acc, global_step=epoch)

            scheduler.step(val_loss.item())
    
    @torch.no_grad()
    def inference(self, data):
        self.model.eval()
        return self.model(data)




