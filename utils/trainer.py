import torch
import ogb
import torch.nn.functional as F
from copy import deepcopy
from utils import Evaluator
import torch.optim.swa_utils
import numpy as np

class TransductiveTrainer: # transductive node classification trainer

    def init_trainer(self, model, epochs, lr, eval_steps=1, metric='acc'):
        self.epochs = epochs
        self.lr = lr
        self.eval_steps = eval_steps
        self.metric = metric
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)
        self.evaluator = Evaluator(metric=self.metric)
        self.best_model_parameters = None
        
    def train(self, data, verbose=False): # choose the epoch of best validation accuracy
        
        best_val_score = 0
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)[data.train_mask]
            loss = F.nll_loss(out, data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()

            if epoch % self.eval_steps == 0:
                with torch.no_grad():
                    self.model.eval()
                    val_pred = self.model(data.x, data.edge_index)[data.val_mask].max(1)[1]
                    val_score = self.evaluator(data.y[data.val_mask], val_pred)
                    if val_score > best_val_score:
                        best_val_score = val_score
                        best_model_parameters = deepcopy(self.model.state_dict())
            
            if verbose:
                print(f'Epoch: {epoch:03d} | train loss: {loss.item():.4f} | val acc: {val_score:.4f} | best val acc: {best_val_score:.4f}')

        if self.best_model_parameters:
            self.model.load_state_dict(self.best_model_parameters)
        
        return best_val_score

    @torch.no_grad()
    def test(self, data, evaluator=None):
        self.model.eval()
        test_pred = self.model(data.x, data.edge_index)[data.test_mask].max(1)[1]

        if evaluator:
            assert type(evaluator) == ogb.nodeproppred.Evaluator
            test_score = evaluator.eval({
                'y_true': data.y[data.test_mask],
                'y_pred': test_pred
            })[self.metric]
        else:
            test_score = self.evaluator(data.y[data.test_mask], test_pred)
        return test_score

class InductiveTrainer: # inductive node classification trainer

    def init_trainer(self, model, epochs, lr, eval_steps=1, metric='acc'):
        self.epochs = epochs
        self.lr = lr
        self.eval_steps = eval_steps
        self.metric = metric
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)
        self.evaluator = Evaluator(metric=self.metric)
        self.best_val_acc = 0
        self.best_model_parameters = None
    
    def train(self, data, verbose=False):
        best_val_score = 0
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(data.train_x, data.train_edge_index)
            loss = F.nll_loss(out, data.train_y)
            loss.backward()
            self.optimizer.step()

            if epoch % self.eval_steps == 0:
                with torch.no_grad():
                    self.model.eval()
                    val_pred = self.model(data.val_x, data.val_edge_index).max(1)[1]
                    val_score = self.evaluator(data.val_y, val_pred)
                    if val_score > best_val_score:
                        best_val_score = val_score
                        best_model_parameters = deepcopy(self.model.state_dict())
            
            if verbose:
                print(f'Epoch: {epoch:03d} | train loss: {loss.item():.4f} | val acc: {val_score:.4f} | best val acc: {best_val_score:.4f}')

        if self.best_model_parameters:
            self.model.load_state_dict(self.best_model_parameters)

        return best_val_score
    
    @torch.no_grad()
    def test(self, data, evaluator=None):
        self.model.eval()
        test_pred = self.model(data.test_x, data.test_edge_index).max(1)[1]
        
        if evaluator:
            assert type(evaluator) == ogb.nodeproppred.Evaluator
            test_score = evaluator.eval({
                'y_true': data.test_y,
                'y_pred': test_pred
            })[self.metric]
        else:
            test_score = self.evaluator(data.test_y, test_pred)
        return test_score
