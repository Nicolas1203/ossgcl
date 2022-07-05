"""Utils function for data loading and data processing.
"""
import numpy as np
import logging as lg

from torchvision import transforms
from torch.utils.data import DataLoader

from src.datasets import CIFAR10, SplitCIFAR10, CIFAR100, SplitCIFAR100
from src.datasets.tinyImageNet import TinyImageNet
from src.datasets.split_tiny import SplitTiny

def get_loaders(args):
    """Get dictionnary of data loaders

    Args:
        args: argparser arguments

    Returns:
        Dictionnay containing task_name - dataloader
        Example:
            {
                'train0': DataLoader object,  # train set task 0
                'train1': DataLoader object,
                'test0': DataLoader object,  # test set task 0
                'test1': DataLoader object,
                'train': DataLoader object,  # train set entire data
                'test': DataLoader object,  # test set entire data
            }
    """
    tf = transforms.ToTensor()
    dataloaders = {}
    if args.dataset == 'cifar10':
        dataset_train = CIFAR10(args.data_root_dir, train=True, download=True, transform=tf)
        dataset_test = CIFAR10(args.data_root_dir, train=False, download=True, transform=tf)
    elif args.dataset == 'cifar100':
        dataset_train = CIFAR100(args.data_root_dir, train=True, download=True, transform=tf)
        dataset_test = CIFAR100(args.data_root_dir, train=False, download=True, transform=tf)
    elif args.dataset == 'tiny':
        dataset_train = TinyImageNet(args.data_root_dir, train=True, download=True, transform=tf)
        dataset_test = TinyImageNet(args.data_root_dir, train=False, download=True, transform=tf)   

    if args.training_type == 'inc':
        dataloaders = add_incremental_splits(args, dataloaders, tf, tag="train")
        dataloaders = add_incremental_splits(args, dataloaders, tf, tag="test")
    
    dataloaders['train'] = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloaders['test'] = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return dataloaders


def add_incremental_splits(args, dataloaders, tf, tag="train"):
    """Add a dataloader for a specific subset of classes to dataloaders dictionnary

    Args:
        args: argparser arguments
        dataloaders (Dict): Dictionnary of taskname - dataloader pairs 
        tf (torch.Transform): Transformation to apply. Usually ToTensor. Augmentation is done during the training loop. 
        tag (str, optional): tag to name the task. The id of the task will be appended to it. Defaults to "train".

    Returns:
        Updated dataloaders dictionnary
    """
    if args.labels_order is None:
        l = np.arange(args.n_classes)
        np.random.shuffle(l)
        args.labels_order = l.tolist()
    is_train = tag == "train"
    step_size = int(args.n_classes / args.n_tasks)
    lg.info("Loading incremental splits with labels :")
    for i in range(0, args.n_classes, step_size):
        lg.info([args.labels_order[j] for j in range(i, i+step_size)])
    for i in range(0, args.n_classes, step_size):
        if args.dataset == 'cifar10':
            dataset = SplitCIFAR10(
                args.data_root_dir,
                train=is_train,
                transform=tf,
                download=True,
                number_list=[args.labels_order[j] for j in range(i, i+step_size)]
            )
        elif args.dataset == 'cifar100':
            dataset = SplitCIFAR100(
                args.data_root_dir,
                train=is_train,
                transform=tf,
                download=True,
                number_list=[args.labels_order[j] for j in range(i, i+step_size)]
            )
        elif args.dataset == 'tiny':
            dataset = SplitTiny(
                args.data_root_dir,
                train=is_train,
                transform=tf,
                download=True,
                number_list=[args.labels_order[j] for j in range(i, i+step_size)]
            )
        else:
            raise NotImplementedError
        dataloaders[f"{tag}{int(i/step_size)}"] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
 
    return dataloaders
