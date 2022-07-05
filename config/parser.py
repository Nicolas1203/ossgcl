import argparse
import json

class Parser:
    """
    Command line parser based on argparse. This also includes arguments sanity check.
    """

    def __init__(self):
        parser = argparse.ArgumentParser(description="Pytorch Continual learning repo.")

        # Training parameters
        parser.add_argument('--epochs', default=1, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int,
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('-b', '--batch-size', default=10, type=int,
                            help='Batch size for stream data (default: 10)')
        parser.add_argument('--learning-rate', '-lr', default=0.1, type=float, help='Initial learning rate')
        parser.add_argument('--momentum', default=0, type=float, help='momentum')
        parser.add_argument('--weight-decay', '--wd', default=0, type=float, help='weight decay (default: 0)')
        parser.add_argument('--optim', default='SGD', choices=['Adam', 'SGD'])
        parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
        parser.add_argument('--memory-only', '-mo', action='store_true', help='Training using only the memory ?')
        parser.add_argument('--resume', '-r', default='', 
                            help="Resume old training. Load model given as resume parameter")
        # Logs parameters
        parser.add_argument('--tag', '-t', default='', help="Base name for graphs and checkpoints")
        parser.add_argument('--tb-root', default='./tb_runs/', help="Where do you want tensorboards graphs ?")
        parser.add_argument('--print-freq', '-p', default=200, type=int,
                            help='Number of batches between each print (default: 200)')
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logs.')
        parser.add_argument('--logs', '-l', default='output.log', help="Name of the log file")
        parser.add_argument('--logs-root', default='./logs', help="Defalt root folder for writing logs.")
        parser.add_argument('--results-root', default='./results/', help='Where you want to save the results ?')
        # checkpoints params
        parser.add_argument('--ckpt-root', default='./checkpoints/', help='Directory where to save the model.')
        #####################
        # Dataset parameters
        #####################
        parser.add_argument('--data-root-dir', default='/data/',
                            help='Root dir containing the dataset to train on.')
        parser.add_argument('--min-crop', type=float, default=0.2, help="Minimum size for cropping in data augmentation. range (0-1)")
        parser.add_argument('--dataset', '-d', default="cifar10", choices=['cifar10', 'cifar100', 'tiny'],
                            help='Dataset to train on')
        parser.add_argument('--training-type', default='inc', choices=['uni', 'inc'],
                            help='How to feed the data to the network (incremental (continual learning) or uniform (offline))')
        parser.add_argument('--n-classes', type=int, default=10,
                            help="Number of classes in database.")
        parser.add_argument("--img-size", type=int, default=32, help="Size of the square input image")
        parser.add_argument('--num-workers', '-w', type=int, default=0, help='Number of workers to use for dataloader.')
        parser.add_argument("--n-tasks", type=int, default=5, help="How many tasks do you want ?")
        parser.add_argument("--labels-order", type=int, nargs='+', help="In which order to you want to see the labels ? Random if not specified.")
        parser.add_argument('--nb-channels', type=int, default=3, 
                            help="Number of channels for the input image.")
        # Contrastive loss parameters
        parser.add_argument('--temperature', default=0.07, type=float, help='temperature parameter for softmax')
        parser.add_argument('--alpha', type=float, help="Weight for stream in SemiCon loss", default=1.0)
        # Memory parameters
        parser.add_argument('--mem-size', type=int, default=200, help='Memory size for continual learning')
        parser.add_argument('--mem-batch-size', '-mbs', type=int, default=100, help="How many images do you want to retrieve from the memory/ltm")
        parser.add_argument('--buffer', default='reservoir', choices=['reservoir'], help="Type of buffer.")
        parser.add_argument('--drop-method', default='random', choices=['random'], help="How to drop images from memory when adding new ones.")
        # Learner parameter
        parser.add_argument('--learner', choices=['ER', 'CE','SCR', 'SSCL', 'SC'],
                            help='What learner do you want ?')
        parser.add_argument('--eval-mem', action='store_true', dest='eval_mem')
        parser.add_argument('--eval-random', action='store_false', dest='eval_mem')
        # Multi runs arguments
        parser.add_argument('--n-runs', type=int, default=1, help="Number of runs, with different seeds each time.")
        parser.add_argument('--run-id', type=int, help="Id of the current run in multi run.")
        # Inference parameters
        parser.add_argument('--lab-pc', type=int, default=100, 
                            help="Number of labeled images per class when evaluating outside of memory.")
        # Tricks
        parser.add_argument('--mem-iters', type=int, default=1, help="Number of times to make a grad update on memory at each step")
        parser.add_argument('--review', '-rv', action='store_true', help="Review trick on last memory.")
        parser.add_argument('--rv-iters', '-rvi', type=int, default=20, help="Number of iteration to do on last memory after training.")

        parser.set_defaults(eval_mem=True)
        self.parser = parser

    def parse(self, arguments=None):
        if arguments is not None:
            self.args = self.parser.parse_args(arguments)
        else:
            self.args = self.parser.parse_args()
        self.check_args()
        return self.args

    def check_args(self):
        """Modify default arguments values depending on the method and dataset.
        """
        #############################
        # Dataset parameters sanity #
        #############################
        if self.args.dataset == 'cifar10':
            self.args.img_size = 32
            self.args.n_classes = 10
        if self.args.dataset == 'cifar100': 
            self.args.img_size = 32
            self.args.n_classes = 100
        if self.args.dataset == 'tiny':
            self.args.img_size = 64
            self.args.n_classes = 200
