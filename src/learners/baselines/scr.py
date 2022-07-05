import torch
import time
import logging as lg

from src.learners.base import BaseLearner
from src.utils.losses import SupConLoss
from src.utils import name_match

class SCRLearner(BaseLearner):
    def __init__(self, args):
        """Supervised Contrastive Learning. Fully labeled.
        https://arxiv.org/abs/2004.11362

        Args:
            args: argparser arguments
        """
        super().__init__(args)
        self.buffer = name_match.buffers[self.params.buffer](
                max_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method,
            )

    def load_criterion(self):
        """Load loss

        Returns:
            pytorch criterion: Loss to use for optimization
        """
        return SupConLoss(self.params.temperature) 

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
                    combined_x_aug1 = self.transform_train(combined_x)
                    combined_x_aug2 = self.transform_train(combined_x)

                    # Inference
                    _, projections1 = self.model(combined_x_aug1)  # (batch_size, projection_dim)
                    _, projections2 = self.model(combined_x_aug2)
                    projections = torch.cat([projections1.unsqueeze(1), projections2.unsqueeze(1)], dim=1)
                    
                    # Loss
                    loss = self.criterion(features=projections, labels=combined_y)

                    self.loss = loss.item()
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    
            # Update buffer
            self.buffer.update(imgs=batch_x, labels=batch_y, model=self.model)

            # Plot to tensorboard
            self.plot()

            if ((not j % self.params.print_freq) or (j == (len(dataloader) - 1))) and (j > 0):
                lg.info(
                    f"Task : {task_name}   batch {j+1}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                )
                self.save(model_name=f"ckpt_{task_name}.pth")
    
    def plot(self):
        """Plot to tensorboard loss and labels distribution in memory
        """
        self.writer.add_scalar("loss", self.loss, self.stream_idx)
        percs = self.buffer.get_labels_distribution()
        for i in range(self.params.n_classes):
            self.writer.add_scalar(f"labels_distribution/{i}", percs[i], self.stream_idx)

    def combine(self, batch_x, batch_y, mem_x, mem_y):
        """Combine batch data and memory data.

        Args:
            batch_x (torch.Tensor): stream images
            batch_y (torch.Tensor): stream labels
            mem_x (torch.Tensor): Memory images
            mem_y (torch.Tensor): Memory labels

        Returns:
            combined images and labels
        """
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        # Return only memory data when training on memory only
        if self.params.memory_only:
            return mem_x, mem_y
        else:
            return combined_x, combined_y