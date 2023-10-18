import argparse
from math import inf
import optuna
from optuna.samplers import TPESampler
from pathlib import Path
import os
import copy
import torch
import torch.cuda
from time import sleep
import mlflow
from optuna.trial._state import TrialState
from pynvml import *
from mlflow.entities import RunStatus
from cv_train import main

def objective(trial, args, user_attr={}, system_attr={}):
    trial_args = copy.deepcopy(args)
    # fixed hyperparams
    if trial_args.output_dir:
        trial_args.output_dir = os.path.join(trial_args.output_dir, str(trial.number))
    trial_args.name = str(trial.number)
    trial_args.pretrained = True
    
    # hyperparams to be optimized
    trial_args.model = trial.suggest_categorical("model", ["resnet18",
                                                     "resnet34",
                                                     "resnet50",
                                                     "resnet101",
                                                     "resnet152",
                                                     "resnext50_32x4d",
                                                     "resnext101_32x8d",
                                                     "wide_resnet50_2",
                                                     "wide_resnet101_2",
                                                     "shufflenet_v2_x0_5",
                                                     "shufflenet_v2_x1_0",
                                                     "squeezenet1_1",
                                                     "densenet121",
                                                     "densenet169",
                                                     "densenet201",
                                                     "densenet161"])    
    trial_args.lr = trial.suggest_loguniform('lr', 0.0001, 0.015)
    trial_args.lr_gamma = trial.suggest_loguniform('lr_gamma', 0.005, 0.5)
    trial_args.lr_step_size = trial.suggest_discrete_uniform('lr_step_size', 3, 24, 1)
    trial_args.momentum = trial.suggest_uniform('momentum', 0, 0.99)
    trial_args.weight_decay = trial.suggest_discrete_uniform('weight_decay', 0.0001, 0.0005, 0.0001)
    trial_args.balance_samples = trial.suggest_categorical('balance_samples', [True])
    
    return start_and_track(trial, trial_args, user_attr, system_attr, trial_args.cv_start)
    
def start_and_track(trial, trial_args, user_attr={}, system_attr={}, cv_start=0, run_uuid=None):
    metric = 0
    for i in range(cv_start, trial_args.nr_cv): 
        print(f"START CV {i}")
        metric += main(trial_args, trial, i, run_uuid, user_attr, system_attr)
        print(f"END CV {i}")
    return metric / (trial_args.nr_cv - cv_start)


def get_mem_info():
    # try:
    #     return torch.cuda.mem_get_info()[0]
    # except:
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    return info.free
        # try:
        #     # Compatibility reasons
        #     r = torch.cuda.memory_reserved(0)
        #     a = torch.cuda.memory_allocated(0)
        #     return r - a 
        # except:
        #     return inf
        
def check_memory_wrapper(min_memory):
    def check_memory(study, finished_trial):
        print(f"Finished trial {finished_trial}")
        # print(finished_trial.state)
        m = get_mem_info()
        print(f"Free memory: {m}")
        # let's require 2 GB free GPU memory before starting trial
        while(m < min_memory):
            sleep(60)
            m = get_mem_info()
            print(f"Free memory: {m}")
    return check_memory

def required_memory_approximation(args):
    if args.min_memory:
        # Can be overwritten
        return args.min_memory
    return 5694 * args.input_size[0] * args.input_size[1] * args.batch_size**0.25
    

if __name__ == '__main__':
    # import faulthandler

    # faulthandler.enable()
    from cv_train import parse_args
    args = parse_args()
    #storage='postgresql://postgres:SvBU81olI4@172.26.62.216:6543/optuna'
    # study = optuna.load_study(study_name=args.experiment_name,
    #                           storage=storage,
    #                           sampler=TPESampler(n_startup_trials=8),
    #                           pruner=optuna.pruners.MedianPruner(n_startup_trials=10,
    #                                                              n_warmup_steps=20,
                                                                    # interval_steps=20))
    study = optuna.create_study(study_name=args.experiment_name,
                                direction="maximize",
                                
                                sampler=TPESampler(n_startup_trials=8),
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10,
                                                                   n_warmup_steps=20,
                                                                   interval_steps=20))                                                    
    # Approximation for required memory
    min_memory = required_memory_approximation(args)
    if args.min_memory:
        # Can be overwritten
        min_memory = args.min_memory
    print(f"Min memory: {min_memory / 1024**3} GB")
    check_memory = check_memory_wrapper(min_memory)
    check_memory(None, None)
    study.optimize(lambda trial: objective(trial, args, user_attr=args.user_attr, system_attr=args.system_attr), n_trials=100,  gc_after_trial=True, callbacks=[check_memory]) # , catch=(RuntimeError,)