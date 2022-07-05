import torch
import time
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import logging as lg

from sklearn.metrics import accuracy_score

from src.models.resnet import SupConResNet
from src.learners.base import BaseLearner
from src.utils.metrics import forgetting_line   


class CELearner(BaseLearner):
    def __init__(self, args):
        """Cross Entropy learner, without memory buffer.

        Args:
            args: argparser arguments
        """
        super().__init__(args)
        self.results = []
        self.results_forgetting = []

    def load_criterion(self):
        return nn.CrossEntropyLoss()

    def train(self, dataloader, **kwargs):
        """Train the model for on epoch over the given dataloader.

        Args:
            dataloader: Pytorch dataloader
        """
        epoch = kwargs.get('epoch', None)
        task_name = kwargs.get('task_name', None)

        self.model = self.model.train()

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0].to(self.device), batch[1].to(self.device)
            self.stream_idx += len(batch_x)
            
            batch_aug = self.transform_train(batch_x)

            # Inference
            _, logits = self.model(batch_aug)  # (batch_size, projection_dim)

            # Loss
            loss = self.criterion(logits, batch_y.long())
            self.loss = loss.item() 
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            # Plot to tensorboard
            self.plot()

            if ((not j % self.params.print_freq) or (j == (len(dataloader) - 1))) and (j > 0):
                if epoch is not None:
                    s = f"Epoch : {epoch}   batch {j+1}/{len(dataloader)}   Loss : {loss.item():.4f}    time : {time.time() - self.start:.4f}s"
                elif task_name is not None:
                    s = f"Task : {task_name}   batch {j+1}/{len(dataloader)}   Loss : {loss.item():.4f}    time : {time.time() - self.start:.4f}s"
                lg.info(s)
                self.save(model_name=f"ckpt_{task_name}.pth")
            break
    
    def evaluate(self, dataloaders, task_id):
        """Evaluate the model after training ona specified task

        Args:
            dataloaders (Dictonnary of dataloaders): Dictionnary containing every train and test loader for each task
            task_id (int): id of the current task 

        Returns:
            average accuracy and average forgetting
        """
        self.model.eval()
        accs = []

        for j in range(task_id + 1):
            test_preds, test_targets = self.encode(dataloaders[f"test{j}"])
            acc = accuracy_score(test_preds, test_targets)
            accs.append(acc)
        for _ in range(self.params.n_tasks - task_id - 1):
            accs.append(np.nan)
        self.results.append(accs)
        
        line = forgetting_line(pd.DataFrame(self.results), task_id=task_id)
        line = line[0].to_numpy().tolist()
        for _ in range(self.params.n_tasks - task_id - 1):
            line.append(np.nan)
        self.results_forgetting.append(line)

        self.print_results(task_id)

        # Coef to correc caculus of forgetting. As current task forgetting is always 0
        # using np.mean() give a lower forgetting than expected.
        correction = len(self.results_forgetting[-1]) / (len(self.results_forgetting[-1]) - 1)
        return np.nanmean(self.results[-1]), np.nanmean(self.results_forgetting[-1]) * correction
    
    def print_results(self, task_id):
        n_dashes = 20
        pad_size = 8
        lg.info('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        lg.info('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
        for line in self.results:
            lg.info('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line) + f"  {np.nanmean(line):.4f}")


    def save_results(self):
        """Save results to csv.
        """
        if self.params.run_id is not None:
            results_dir = os.path.join(self.params.results_root, self.params.tag, f"run{self.params.run_id}")
        else:
            results_dir = os.path.join(self.params.results_root, self.params.tag, f"run{self.params.seed}")
        lg.info(f"Saving accuracy results in : {results_dir}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        
        df_avg = pd.DataFrame()
        cols = [f'task {i}' for i in range(self.params.n_tasks)]
        # Loop over classifiers results
        # Each classifier has a value for every task. NaN if future task
        df_clf = pd.DataFrame(self.results, columns=cols)
        # Average accuracy over all tasks with not NaN value
        df_avg = df_clf.mean(axis=1)
        df_clf.to_csv(os.path.join(results_dir, 'acc.csv'), index=False)
        df_avg.to_csv(os.path.join(results_dir, 'avg.csv'), index=False)
        
        df_avg = pd.DataFrame()
        lg.info(f"Saving forgetting results in : {results_dir}")
        cols = [f'task {i}' for i in range(self.params.n_tasks)]
        df_clf = pd.DataFrame(self.results_forgetting, columns=cols)
        df_avg = df_clf.mean(axis=1)
        df_clf.to_csv(os.path.join(results_dir, 'forgetting.csv'), index=False)
        df_avg.to_csv(os.path.join(results_dir, 'avg_forgetting.csv'), index=False)

        self.save_parameters()

    def evaluate_offline(self, dataloaders, epoch):
        """Evaluate the model on test set at a specified epoch. Train loader is used for training
        a classifier on some random images selected from train set

        Args:
            dataloaders (dict): Dictionnary containing dataloaders 'train' and 'test'
        """
        with torch.no_grad():
            self.model.eval()

            test_preds, test_targets = self.encode(dataloaders['test'])
            acc = accuracy_score(test_preds, test_targets)
            self.results.append(acc)
            
            if not ((epoch + 1) % 5):
                self.save(model_name=f"checkpoint_{epoch}.pth")
        
        lg.info(f"ACCURACY {self.results[-1]}")
        return self.results[-1]

    def encode(self, dataloader, nbatches=-1):
        """Compute representations - labels pairs for a certain number of batches of dataloader.
            Not really optimized.

        Args:
            dataloader (torch dataloader): dataloader to encode
            nbatches (int, optional): Number of batches to encode. Use -1 for all. Defaults to -1.

        Returns:
            representations - labels pairs
        """
        i = 0
        with torch.no_grad():
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(self.device)
                _, logits = self.model(self.transform_test(inputs))
                preds = nn.Softmax(dim=1)(logits).argmax(dim=1)
                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat = preds.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat = np.hstack([all_feat, preds.cpu().numpy()])
        return all_feat, all_labels

    def save_results_offline(self):
        """Save results as csv for offline training
        """
        if self.params.run_id is not None:
            results_dir = os.path.join(self.params.results_root, self.params.tag, f"run{self.params.run_id}")
        else:
            results_dir = os.path.join(self.params.results_root, self.params.tag)

        lg.info(f"Saving accuracy results in : {results_dir}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)

        pd.DataFrame(self.results).to_csv(os.path.join(results_dir, 'acc.csv'), index=False)

        self.save_parameters()

    def plot(self):
        """Plot loss to tensorboard"""
        self.writer.add_scalar("loss", self.loss, self.stream_idx)