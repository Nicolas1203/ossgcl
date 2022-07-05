from copyreg import pickle
import torch
import numpy as np
import random as r
import logging as lg

class Buffer(torch.nn.Module):
    def __init__(self, max_size=200, shape=(3,32,32), n_classes=10, **kwargs):
        """Buffer abstract class

        Args:
            max_size (int, optional): Maximum size of the buffer. Defaults to 200.
            shape (tuple, optional): shape of the stored data. Defaults to (3,32,32).
            n_classes (int, optional): Number of classes that will be present in buffer, for print purposes only. Defaults to 10.
        """
        super().__init__()
        self.n_classes = n_classes  # For print purposes only
        self.max_size = max_size
        self.shape = shape
        self.n_seen_so_far = 0
        self.n_added_so_far = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.shape is not None:
            if len(self.shape) == 3:
                self.register_buffer('buffer_imgs', torch.FloatTensor(self.max_size, self.shape[0], self.shape[1], self.shape[2]).fill_(0))
            elif len(self.shape) == 1:
                self.register_buffer('buffer_imgs', torch.FloatTensor(self.max_size, self.shape[0]).fill_(0))
        self.register_buffer('buffer_labels', torch.LongTensor(self.max_size).fill_(-1))

    def update(self, imgs, labels=None, **kwargs):
        raise NotImplementedError

    def replace_data(self, idx, img, label):
        """Replace buffer_img - buffer_label pair at idx by img - label 

        Args:
            idx (int): index of the data to replace
            img (torch.Tensor): New data to put in memory
            label (torch.Tensor): Data label to put in memory
        """
        self.buffer_imgs[idx] = img
        self.buffer_labels[idx] = label
        self.n_added_so_far += 1
    
    def is_empty(self):
        return self.n_added_so_far == 0
    
    def random_retrieve(self, n_imgs=100):
        """Randomly retrieve a number of images from memory

        Returns:
            tuples of tensors: n_imgs pairs of data - labels
        """
        if self.n_added_so_far < n_imgs:
            lg.debug(f"""Cannot retrieve the number of requested images from memory {self.n_added_so_far}/{n_imgs}""")
            return self.buffer_imgs[:self.n_added_so_far], self.buffer_labels[:self.n_added_so_far]
        
        ret_indexes = r.sample(np.arange(min(self.n_added_so_far, self.max_size)).tolist(), n_imgs)
        ret_imgs = self.buffer_imgs[ret_indexes]
        ret_labels = self.buffer_labels[ret_indexes]
        
        return ret_imgs, ret_labels
    
    def get_all(self):
        """returns all data and labels stored in buffer

        Returns:
            tuples of tensors: buffer images and buffer labels
        """
        return self.buffer_imgs[:min(self.n_added_so_far, self.max_size)],\
             self.buffer_labels[:min(self.n_added_so_far, self.max_size)]

    def n_data(self):
        """Number of data in buffer
        """
        return len(self.buffer_labels[self.buffer_labels >= 0])

    def is_full(self):
        return self.n_data() == self.max_size

    def get_labels_distribution(self):
        """Compute the histogram of present labels in memory. This is used for print purposes.

        Returns:
            np.array: histogram of class representations in buffer
        """
        np_labels = self.buffer_labels.numpy().astype(int)
        counts = np.bincount(np_labels[self.buffer_labels >= 0], minlength=self.n_classes)
        tot_labels = len(self.buffer_labels[self.buffer_labels >= 0])
        if tot_labels > 0:
            return counts / len(self.buffer_labels[self.buffer_labels >= 0])
        else:
            return counts
