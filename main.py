import os
import torch
import pandas as pd
import numpy as np
import sys
import logging as lg
import datetime as dt
import random as r

from src.utils.data import get_loaders
from src.utils import name_match
from config.parser import Parser


def main():
    runs_accs = []
    runs_fgts = []
    
    parser = Parser()
    args = parser.parse()

    timestamp = int(dt.datetime.now().timestamp())
    logfile = f'{timestamp}_{args.tag}.log'

    if not os.path.exists(args.logs_root): os.mkdir(args.logs_root)

    # Define logger, timestamp and add timestamp to tag
    timestamp = int(dt.datetime.now().timestamp())
    args.tag = f"{timestamp}_{args.tag}" if len(args.tag) > 0 else str(timestamp)
    logfile = f'{args.tag}.log'
    if not os.path.exists(args.logs_root): os.mkdir(args.logs_root)
    cf = lg.Formatter('%(name)s - %(levelname)s - %(message)s')
    ff = lg.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = lg.getLogger()
    fh = lg.FileHandler(os.path.join(args.logs_root, logfile))
    ch = lg.StreamHandler()
    ch.setFormatter(cf)
    fh.setFormatter(ff)
    logger.addHandler(fh)
    logger.addHandler(ch)
    if args.verbose:
        logger.setLevel(lg.DEBUG)
        logger.warning("Running in VERBOSE MODE.")
    else:
        logger.setLevel(lg.INFO)
    
    for run_id in range(args.n_runs):
        # Re-parse tag. Useful when using multiple runs.
        args = parser.parse()
        args.tag = f"{timestamp}_{args.tag}" if len(args.tag) > 0 else str(timestamp)
        
        # Seed initilization
        if args.n_runs > 1: args.seed = run_id
        np.random.seed(args.seed)
        r.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        lg.info("=" * 60)
        lg.info("=" * 20 + f"RUN N°{run_id} SEED {args.seed}" + "=" * 20)
        lg.info("=" * 60)        
        lg.info("Parameters used for this training")
        lg.info("=" * 20)
        lg.info(args)

        # Dataloaders
        dataloaders = get_loaders(args)

        # Learner
        if args.learner is not None:
            learner = name_match.learners[args.learner](args)
        else:
            raise Warning("Please select the desired learner.")
            
        # Training
        # Class incremental training
        if args.training_type == 'inc':
            for task_id in range(args.n_tasks):
                task_name = f"train{task_id}"
                learner.train(dataloader=dataloaders[task_name], task_name=task_name)
                avg_acc, avg_fgt = learner.evaluate(dataloaders, task_id)
            learner.after_train(task_name=task_name)
            learner.save_results()
        # Uniform training (offline)
        elif args.training_type == 'uni':
            for e in range(args.epochs):
                learner.train(dataloaders['train'], epoch=e)
                avg_acc = learner.evaluate_offline(dataloaders, epoch=e)
                avg_fgt = 0
            learner.save_results_offline()
        runs_accs.append(avg_acc)
        runs_fgts.append(avg_fgt)
    
    # Save runs accs and forgettings
    if args.n_runs > 1:
        df_acc = pd.DataFrame(runs_accs)
        df_fgt = pd.DataFrame(runs_fgts)
        results_dir = os.path.join(args.results_root, args.tag)
        lg.info(f"Results for the aggregated runs are save in : {results_dir}")
        df_acc.to_csv(os.path.join(results_dir, 'runs_accs.csv'), index=False)
        df_fgt.to_csv(os.path.join(results_dir, 'runs_fgts.csv'), index=False)

    # Exits the program
    sys.exit(0)


if __name__ == '__main__':
    main()
  