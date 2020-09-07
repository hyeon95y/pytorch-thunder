import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import uuid
import copy
from tqdm.auto import trange
import math
import dill
import pickle
import matplotlib.pyplot as plt

from solar.eval import sum_of_params
from solar.eval import num_of_params
from solar.eval import sum_of_grads

class SklearnInterfaceModel:
    def __init__(
        self,
        nn_module,
        optimizer,
        criterion,
        epochs,
        patience,
        batch_size,
        device,
        num_workers,
        pin_memory,
        result_path,
        progress,
        y_colname,
        exp_id=None,
        show_history=True,
        history_figsize=(20,4),
        result_absolute_path=False
    ):
        self.nn_module = nn_module
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        if result_absolute_path :
            self.result_path = result_path
        else :
            self.result_path = os.path.abspath(os.path.join(os.getcwd(), result_path))
        self.progress=progress
        self.exp_id = str(uuid.uuid4()) if exp_id is None else exp_id
        self.y_colname = y_colname
        self.show_history=show_history
        self.history_figsize=history_figsize


    def fit(self, x:pd.DataFrame, y:pd.DataFrame, eval_set:list=None):
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, pd.DataFrame) or isinstance(y, pd.Series)
        #
        # Create path to save result
        #
        self.path = self._mkdir(self.result_path, self.exp_id)
        
        # 
        # Get dataloader
        #
        train_dataloader, self.num_batches_train = self._get_dataloader(x, y, valid=False)
        valid_dataloader , self.num_batches_valid = (
            self._get_dataloader(eval_set[0], eval_set[1], valid=True)
            if eval_set is not None
            else [None, None]
        )
        
        #
        # Train
        #
        history, state_dicts = self._train(train_dataloader, valid_dataloader)
        
        #
        # Save history, state_dicts
        #
        best_weight_path = os.path.abspath(os.path.join(self.path, 'state_dicts.pkl'))
        with open(best_weight_path, 'wb') as handle:
            dill.dump(state_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        history_path = os.path.abspath(os.path.join(self.path, 'history.pkl'))
        with open(history_path, 'wb') as handle:
            dill.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        #
        # CUDA memory management
        #
        del(train_dataloader)
        del(valid_dataloader)
        del(state_dicts)
        torch.cuda.empty_cache()
        
        #
        # Show trainnig history
        #
        self.show_training_history(history)
        
        return history
    
    def _train(self, train_dataloader, valid_dataloader) :
        # 
        # Logs to save
        #
        history = {}
        state_dicts = {}
        #
        # Early stop & checkpoint
        #
        lowest_valid_idx = None
        lowest_valid_loss = None
        running_patience = 0
        
        self.nn_module.to(self.device)      
        iterator = (trange(self.epochs, position=1) if self.progress else range(self.epochs))
        for idx_epoch in iterator : 
            loss_train_epoch = 0
            loss_valid_epoch = 0
            #
            # Train set
            #
            self.nn_module.train()
            for idx_batch, data in enumerate(train_dataloader) :
                x, y = data[0], data[1]
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                yhat = self.nn_module(x)
                loss = self.criterion(yhat, y)
                loss.backward()
                self.optimizer.step()
                
                loss_train_epoch += loss.item()
            loss_train_epoch /= self.num_batches_train
            #
            # Valid Set
            #
            with torch.no_grad() :
                self.nn_module.eval()
                for idx_batch, data in enumerate(valid_dataloader) :
                    x, y = data[0], data[1]
                    x, y = x.to(self.device), y.to(self.device)
                    
                    yhat = self.nn_module(x)
                    loss = self.criterion(yhat, y)
                    
                    loss_valid_epoch += loss.item()
            loss_valid_epoch /= self.num_batches_valid
            #
            # Log
            #
            result = {'loss_train':loss_train_epoch,
                     'loss_valid':loss_valid_epoch,
                     'abs_sum_params': sum_of_params(self.nn_module, absolute=True),
                     'abs_sum_grads': sum_of_grads(self.nn_module, absolute=True)}
            history[idx_epoch] = result
            state_dicts[idx_epoch] = copy.deepcopy(self.nn_module.state_dict())
            #
            # Early stopping
            #
            lowest_valid_loss = loss_valid_epoch if idx_epoch == 0 else lowest_valid_loss # initial value
            lowest_valid_idx = idx_epoch if idx_epoch == 0 else lowest_valid_idx # initial value
            
            running_patience = 0 if loss_valid_epoch < lowest_valid_loss else running_patience+1 # update when curret loss < lowest loss
            lowest_valid_idx = idx_epoch if loss_valid_epoch < lowest_valid_loss else lowest_valid_idx # update when curret loss < lowest los
            
            lowest_valid_loss = loss_valid_epoch if loss_valid_epoch < lowest_valid_loss else lowest_valid_loss # update lowest loss
            
            #
            # Terminate condition
            #
            if running_patience == self.patience  :
                break

        #
        # Load best weight
        #
        self.nn_module.load_state_dict(copy.deepcopy(state_dicts[lowest_valid_idx]))
            
        #
        # CUDA memory management
        #
        torch.cuda.empty_cache()

        return history, state_dicts
        

    def predict(self, x):
        assert isinstance(x, pd.DataFrame)
        #
        # Prepare data
        #
        indices = x.index
        x = torch.from_numpy(x.values).float().to(self.device)
        #
        # Inference
        #
        with torch.no_grad() :
            self.nn_module.eval() 
            yhat = self.nn_module(x)
            yhat = pd.DataFrame(yhat.detach().cpu().numpy(), columns=[self.y_colname], index=indices)
        #
        # CUDA memory management
        #
        del(x)
        torch.cuda.empty_cache()
            
        return yhat
    
    def show_training_history(self, history) :
        history = pd.DataFrame().from_dict(history).T
        idx_lowest_loss = history["loss_valid"].idxmin()
        lowest_loss = history["loss_valid"].min()

        plt.figure(figsize=self.history_figsize)
        plt.plot(history["loss_train"], label="train")
        plt.plot(history["loss_valid"], label="valid")
        plt.axvline(
            idx_lowest_loss, color="red", label="lowest_valid_epoch : %s" % idx_lowest_loss,
        )
        plt.axhline(
            lowest_loss, color="grey", label="lowest_valid_loss : %s" % lowest_loss,
        )
        plt.title("Training history with loss, exp_id : %s"%self.exp_id)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        fig_path = self.path + '/' + 'training_history_loss.png'
        plt.savefig(fig_path)

        if self.show_history:
            plt.show()

        plt.figure(figsize=self.history_figsize)
        plt.plot(history["loss_train"], label="train")
        plt.plot(history["loss_valid"], label="valid")
        plt.axvline(
            idx_lowest_loss, color="red", label="lowest_valid_epoch : %s" % idx_lowest_loss,
        )
        plt.axhline(
            lowest_loss, color="grey", label="lowest_valid_loss : %s" % lowest_loss,
        )
        plt.title("Training history with log loss, exp_id : %s"%self.exp_id)
        plt.xlabel("Epochs")
        plt.ylabel("Log Loss")
        plt.yscale("log")
        plt.legend()
        fig_path = self.path + '/' + 'training_history_logloss.png'
        plt.savefig(fig_path)
        if self.show_history:
            plt.show()
            
        plt.figure(figsize=self.history_figsize)
        plt.plot(history["abs_sum_params"], label="abs. sum params.")
        plt.axvline(
            idx_lowest_loss, color="red", label="lowest_valid_epoch : %s" % idx_lowest_loss,
        )
        plt.title("Training history with absolute sum of params., exp_id : %s"%self.exp_id)
        plt.xlabel("Epochs")
        plt.ylabel("Abs. sum of params.")
        plt.legend()
        fig_path = self.path + '/' + 'training_history_abs_sum_params.png'
        plt.savefig(fig_path)
        if self.show_history:
            plt.show()
            
        plt.figure(figsize=self.history_figsize)
        plt.plot(history["abs_sum_grads"], label="abs. sum grads.")
        plt.axvline(
            idx_lowest_loss, color="red", label="lowest_valid_epoch : %s" % idx_lowest_loss,
        )
        plt.title("Training history with absolute sum of grads., exp_id : %s"%self.exp_id)
        plt.xlabel("Epochs")
        plt.ylabel("Abs. sum of grads.")
        plt.legend()
        fig_path = self.path + '/' + 'training_history_abs_sum_grads.png'
        plt.savefig(fig_path)
        if self.show_history:
            plt.show()
        
        return

    def _get_dataloader(self, x, y=None, valid=False):
        if y is None:
            dataset = self.get_unsupervied_dataset(x)
        else:
            dataset = self._get_supervised_dataset(x, y)
        num_batches = dataset.num_batches(self.batch_size)
        if valid : 
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        else : 
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True
            )
            
        return dataloader, num_batches

    def _get_supervised_dataset(self, x, y):

        dataset = SupervisedDataset(x, y)
        return dataset

    def _get_unsupervised_dataset(self, x):

        dataset = UnsupervisedDataset(x, y)
        return dataset
    
    def _mkdir(self, result_path, exp_id) :
        if not os.path.exists(result_path) : 
            os.mkdir(result_path)
        path = result_path + '/' + exp_id
        if not os.path.exists(path) : 
            os.mkdir(path)
        return path
    
class SupervisedDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.values).float()
        self.y = torch.from_numpy(y.values).float()

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return [x, y]

    def __len__(self):
        return self.x.shape[0]

    def num_batches(self, batch_size):
        return math.ceil(self.__len__() / batch_size)
    
class UnsupervisedDataset(Dataset):
    def __init__(self, x):
        self.x = torch.from_numpy(x.values).float()

    def __getitem__(self, idx):
        x = self.x[idx]

        return x

    def __len__(self):
        return self.x.shape[0]

    def num_batches(self, batch_size):
        return math.ceil(self.__len__() / batch_size)