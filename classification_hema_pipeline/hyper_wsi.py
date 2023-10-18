import argparse
import optuna
from optuna.samplers import TPESampler
import json
import requests
from pathlib import Path
import datetime
import os
import time
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from natsort import natsorted
import inspect
import sys
import pickle
import shutil
import copy
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
target_dir = os.path.join(current_dir, 'cocoapi', 'PythonAPI')
sys.path.insert(0, target_dir)

import utils

import mlflow
import mlflow.pytorch
#mlflow.set_tracking_uri('http://mlflow.172.26.62.216.nip.io')

#from torch.utils.tensorboard import SummaryWriter
from train import *


def objective(trial, args):
    
    # fixed hyperparams
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, str(trial.number))
    args.name = str(trial.number)
    args.pretrained = True
    args.deterministic = True
    
    # hyperparams to be optimized
    args.model = trial.suggest_categorical("model", ["resnet18",
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
    args.lr = trial.suggest_loguniform('lr', 0.0001, 0.015)
    args.lr_gamma = trial.suggest_loguniform('lr_gamma', 0.005, 0.5)
    args.lr_step_size = trial.suggest_discrete_uniform('lr_step_size', 3, 24, 1)
    args.momentum = trial.suggest_uniform('momentum', 0, 0.99)
    args.weight_decay = trial.suggest_discrete_uniform('weight_decay', 0.0001, 0.0005, 0.0001)
    
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.name):
        metric = main(args, trial)
    
    return metric

if __name__ == '__main__':
    
    args = parse_args()
    storage='postgresql://mlflow_user:mlflow@172.26.62.216:6543/optuna'
    study = optuna.load_study(study_name=args.experiment_name,
                              storage=storage,
                              sampler=TPESampler(n_startup_trials=8),
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=10,
                                                                 n_warmup_steps=20,
                                                                 interval_steps=20))
    study.optimize(lambda trial: objective(trial, args), n_trials=20, catch=(RuntimeError,))