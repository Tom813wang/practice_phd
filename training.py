import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split

from typing import Union
import os

class Data_splitter(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    def split_data(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.label, 
                                                            test_size=test_size, 
                                                            random_state=random_state)
        return X_train, X_test, y_train, y_test
    

class Data_loader(Data_splitter):
    def __init__(self, data, label, batch_size=32):
        super(Data_loader, self).__init__(data, label)
        self.batch_size = batch_size
    
    def get_dataset(self):
        X_train, X_test, y_train, y_test = self.split_data()
        train_dataset = Data_splitter(X_train, y_train)
        test_dataset = Data_splitter(X_test, y_test)
        return train_dataset, test_dataset
    
    def get_dataloader(self):
        train_dataset, test_dataset = self.get_dataset()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader
    

class Training:
    def __init__(self, 
                 *,
                 model : nn.Module = None,
                 optimizer : optim = None,
                 loss_function : nn.Module = None,
                 epochs : int = 10,
                 lr_scheduler = None,
                 clip = None,
                 check_point_path : str = None,
                 device = 'cpu'):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.clip = clip
        self.device = device

        if check_point_path:
            self.check_point_path = check_point_path
        else:
            self.check_point_path = os.getcwd() + '/checkpoints'

    def save_checkpoint(self, idx):
        if not os.path.exists(self.check_point_path):
            os.makedirs(self.check_point_path)

        torch.save(self.model.state_dict(), self.check_point_path + f'/checkpoint_{idx}.pth')

        return self.check_point_path + f'/checkpoint_{idx}.pth'

    def _get_index_checkpoint(self, checkpoint_path):
        index_list = []
        for i in os.listdir(checkpoint_path):
            if i.endswith('.pth'):
                index_list.append(int(i.split('_')[-1].split('.')[0]))
        return index_list

    def load_checkpoint(self, 
                        checkpoint: Union[bool, int] = False):
        
        if checkpoint:
            if isinstance(checkpoint, bool):
                # Load the last checkpoint
                index_list = self._get_index_checkpoint(self.check_point_path)
                idx = max(index_list)
                self.model.load_state_dict(torch.load(self.check_point_path + f'/checkpoint_{idx}.pth'))
                checkpoint_path = self.check_point_path + f'/checkpoint_{idx}.pth'
                return checkpoint_path
            
            elif isinstance(checkpoint, int):
                self.model.load_state_dict(torch.load(self.check_point_path + f'/checkpoint_{checkpoint}.pth'))
                checkpoint_path = self.check_point_path + f'/checkpoint_{checkpoint}.pth'
                return checkpoint_path

            else:
                raise ValueError("checkpoint must be a boolean or an integer")
                   

    def train_loop(self, train_loader):
        self.model.train()
        total_loss = 0

        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_function(output, target)
            loss.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            self.lr_scheduler.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)
    
    def test_loop(self, test_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():

            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += self.loss_function(output, target).item()

        return total_loss / len(test_loader)
    
    def fit(self, train_loader, test_loader,
                checkpoint_save: Union[bool, int] = False,
                checkpoint_load: Union[bool, str] = False,
                epochs : int = None):
        """
        checkpoint: bool or int
            If False, no checkpoint will be saved. If True, checkpoint will be saved after each epoch.
            If int, checkpoint will be saved after each epoch modulo the int value.
        
        checkpoint_load: bool or int   
            If False, no checkpoint will be loaded. If True, the last checkpoint will be loaded.
            If int, the checkpoint will be loaded the epoch modulo the int value.
        """
        self.train_losses = []
        self.test_losses = []

        if epochs is None:
            epochs = self.epochs

        if checkpoint_load:
            checkpoint_path = self.load_checkpoint(checkpoint_load)
            state_dict = torch.load(checkpoint_path)
            self.model.load_state_dict(state_dict)
            print(f'Now You Are Training From Checkpoint {checkpoint_path}')

        for epoch in range(epochs):
            train_loss = self.train_loop(train_loader)
            test_loss = self.test_loop(test_loader)
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            if checkpoint_save:
                if isinstance(checkpoint_save, bool):
                    self.save_checkpoint(epoch)
                    print(f"Each epoch checkpoint saved at {self.check_point_path}")
                elif isinstance(checkpoint_save, int) and isinstance(checkpoint_load, bool):
                    print("No checkpoint loaded, so the checkpoint will be saved at each epoch modulo the int value")
                    if epoch % checkpoint_save == 0:
                        self.save_checkpoint(epoch)
                elif isinstance(checkpoint_save, int) and isinstance(checkpoint_load, int):
                    print(f'{checkpoint_path} is loaded, so the checkpoint will be saved at each epoch modulo the int value')
                    if epoch % checkpoint_save == 0:
                        self.save_checkpoint(epoch + checkpoint_load)
        
        return self.train_losses, self.test_losses
    
    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            output = self.model(data)
            return output
        
    
        