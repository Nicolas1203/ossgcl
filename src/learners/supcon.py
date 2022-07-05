import torch
import time
import os
import pandas as pd
import logging as lg


from src.learners.base import BaseLearner
from src.utils.losses import SupConLoss


class SupConLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

    def load_criterion(self):
        return SupConLoss(self.params.temperature)

    def train(self, dataloader, **kwargs):
        epoch = kwargs.get('epoch', None)
        task_name = kwargs.get('task_name', None)
        self.model = self.model.train()

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0].to(self.device), batch[1].to(self.device)
            self.stream_idx += len(batch_x)
            
            # Augment
            batch_aug1 = self.transform_train(batch_x)
            batch_aug2 = self.transform_train(batch_x)

            # Inference
            _, projections1 = self.model(batch_aug1)  # (batch_size, projection_dim)
            _, projections2 = self.model(batch_aug2)
            projections = torch.cat([projections1.unsqueeze(1), projections2.unsqueeze(1)], dim=1)

            # Loss
            loss = self.criterion(features=projections, labels=batch_y)
            self.loss = loss.item()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            # Plot to tensorboard
            self.plot()

            if ((not j % self.params.print_freq) or (j == (len(dataloader) - 1))) and (j > 0):
                lg.info(
                    f"Epoch : {epoch}   batch {j+1}/{len(dataloader)}   Loss : {loss.item():.4f}    time : {time.time() - self.start:.4f}s"
                )
                self.save(model_name=f"ckpt_{task_name}.pth")
            break

    def save_results(self):
        if self.params.run_id is not None:
            results_dir = os.path.join(self.params.results_root, self.params.tag, f"run{self.params.run_id}")
        else:
            results_dir = os.path.join(self.params.results_root, self.params.tag)
        lg.info(f"Saving accuracy results in : {results_dir}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        df_all = pd.DataFrame()
        for clf_name in self.results:
            df_all[clf_name] = pd.DataFrame(self.results[clf_name])
        df_all.to_csv(os.path.join(results_dir, 'acc.csv'), index=False)

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
