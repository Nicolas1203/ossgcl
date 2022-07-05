import torch
import time
import numpy as np
import os
import pandas as pd
import logging as lg

from sklearn.metrics import accuracy_score

from src.learners.ce import CELearner
from src.utils.metrics import forgetting_line   
from src.utils import name_match

class ERLearner(CELearner):
    """Experience replay learner. https://arxiv.org/abs/1811.11682.
    """
    def __init__(self, args):
        super().__init__(args)
        self.buffer = name_match.buffers[self.params.buffer](
                max_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method,
            )

    def train(self, dataloader, task_name):
        """Train the model for one epoch over the given dataloader

        Args:
            dataloader (Pytorch dataloaser): Pytorch dataloader, supposed to contained the data for a defined task
            task_name (str): Name of the current task
        """
        self.model = self.model.train()

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)
            
            for _ in range(self.params.mem_iters):
                mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:
                    # Combined batch
                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)  # (batch_size, nb_channel, img_size, img_size)

                    # Augment
                    combined_x_aug = self.transform_train(combined_x)

                    # Inference
                    _, logits = self.model(combined_x_aug)

                    # Loss
                    loss = self.criterion(logits, combined_y.long())
                    self.loss = loss.item()
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
            
            # Update reservoir buffer
            self.buffer.update(imgs=batch_x, labels=batch_y, model=self.model)

            # Plot to tensorboard
            self.plot()

            if ((not j % self.params.print_freq) or (j == (len(dataloader) - 1))) and (j > 0):
                lg.info(
                    f"Task : {task_name}   batch {j+1}/{len(dataloader)}   Loss : {loss.item():.4f}    time : {time.time() - self.start:.4f}s"
                )
                self.save(model_name=f"ckpt_{task_name}.pth")
                break
            break

    def encode(self, dataloader, nbatches=-1):
        i = 0
        with torch.no_grad():
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(self.device)
                _, logits = self.model(self.transform_test(inputs))
                _, preds = torch.max(logits, 1)
                
                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat = preds.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat = np.hstack([all_feat, preds.cpu().numpy()])
        return all_feat, all_labels

    def save_results_offline(self):
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
        self.writer.add_scalar("loss", self.loss, self.stream_idx)
    
    def combine(self, batch_x, batch_y, mem_x, mem_y):
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        if self.params.memory_only:
            return mem_x, mem_y
        else:
            return combined_x, combined_y
