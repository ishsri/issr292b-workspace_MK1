import datetime
import os
import time

import pickle
import copy
import shutil
from pathlib import Path
import tempfile

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from torchsampler import ImbalancedDatasetSampler
import pandas as pd
from PIL import Image

import utils as utils

import mlflow
import mlflow.pytorch
import optuna

import argparse
from mlflow.entities.run_info import RunInfo
from mlflow.tracking.client import MlflowClient
from mlflow.entities import  RunStatus
from torchmetrics import F1Score
import traceback
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
class MlflowTracker(object):
    
    active_run: mlflow.ActiveRun = None
    tracking_uri = "http://taurusi8033.taurus.hrsk.tu-dresden.de:5000" #http://localhost:5000
    experiment_name = ""
    run_uuid = ""
    run_name = ""
    
    
       
    @staticmethod
    def initialize(experiment_name, run_name="", run_uuid=None):
        MlflowTracker.experiment_name = experiment_name
        MlflowTracker.run_uuid = run_uuid
        MlflowTracker.run_name = run_name
        print("Initialized mlflow: ")
        print(f"  {MlflowTracker.tracking_uri}")
        print(f"  {MlflowTracker.run_uuid}")
        print(f"  {MlflowTracker.experiment_name}")
        

    @staticmethod
    def connect():
        mlflow.set_tracking_uri(MlflowTracker.tracking_uri)
        print(f"Set active experiment to {MlflowTracker.experiment_name}")
        mlflow.set_experiment(MlflowTracker.experiment_name)
        if MlflowTracker.run_uuid is not None:
            # Resume run
            print(f"Resuming run {MlflowTracker.run_uuid}")
            MlflowTracker.active_run = mlflow.start_run(run_id=MlflowTracker.run_uuid)
        else:
            # New run
            print(f"Create new run {MlflowTracker.run_name}")
            
            MlflowTracker.active_run = mlflow.start_run(run_name=MlflowTracker.run_name)            
        run_info: RunInfo = MlflowTracker.active_run.info
        MlflowTracker.run_uuid = run_info.run_uuid
        print("  "+run_info.experiment_id)
        
    @staticmethod
    def reconnect():
        print("Reconnecting to mlflow")
        print(f"  {MlflowTracker.tracking_uri}")
        print(f"  {MlflowTracker.run_uuid}")
        print(f"  {MlflowTracker.experiment_name}")
        mlflow.end_run()
        print("  Ended previous run")
        mlflow.set_tracking_uri(MlflowTracker.tracking_uri)
        mlflow.set_experiment(MlflowTracker.experiment_name)
        MlflowTracker.active_run = mlflow.start_run(run_id=MlflowTracker.run_uuid)
        print("  Reconnected")
        
    @staticmethod
    def finish():
        print("Finish mlflow run")
        mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))

    @staticmethod
    def fail():
        mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))

def load_classification_model(checkpoint_path, modelname, class_names=[0,1], device=torch.device('cpu')):
    args = argparse.Namespace()
    args.__dict__ = { 
                    "model": modelname,
                    "class_names": class_names
                    }
    model = torchvision.models.__dict__[args.model]()
    model = reshape_classification_head(model, args, class_names)
    model_without_ddp = model
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model_without_ddp.load_state_dict(checkpoint['model'])
    model_without_ddp.to(device)
    return model_without_ddp

def classification_model_predict(model, img, input_size, device=torch.device('cpu')):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([
                # transforms.RandomResizedCrop(tuple([499,499])),
                transforms.Resize(tuple(input_size)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
    model.eval()

    img_tensor = img_transforms(img).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img_tensor.to(device))

    return prediction

def calc_log_ROC(model, data_loader, device, classes, cv_str):
    #calculate roc curve, create and save matplotlib fig, send to mlflow
    
    # only tested for binary classification, not distributed mode and gpu
    model.eval()
    with torch.no_grad():
        result = torch.cuda.FloatTensor()
        targets = torch.cuda.LongTensor()
        for image, target in data_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            m = nn.Softmax(dim=1)
            output =  m(output)
            result = torch.cat((result,output),dim=0)
            targets = torch.cat((targets,target),dim=0)
            
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    from matplotlib import pyplot as plt

    class_of_interst = len(classes)-1

    y = targets.cpu()
    scores = result[:,class_of_interst].cpu()
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=class_of_interst)
    roc_auc = auc(fpr, tpr)
            
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        savepath = os.path.join(tmpdirname,f'{args.run_name}_{cv_str}_ROC.png')
        plt.savefig(savepath, dpi=1000)
        roc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        roc_data.insert(0, 'RunName', cv_str)
        csv_path = os.path.join(tmpdirname, f'{args.run_name}_{cv_str}_ROC.csv')
        roc_data.to_csv(csv_path, index=False)
        retry(lambda: mlflow.log_artifact(savepath, artifact_path=f"{args.run_name}_{cv_str}_ROC_Curve"), 5, "Could not log roc")
        retry(lambda: mlflow.log_artifact(csv_path, artifact_path=f"{args.run_name}_{cv_str}_ROC_Data"), 5, "Could not log ROC data")

def log_data_split(args, indices, splitpoint, cv_str=""):
    df = pd.read_csv(args.data_path)
    df_train = df.iloc[indices[:splitpoint],:]
    df_test = df.iloc[indices[splitpoint:],:]
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        df_train.to_csv(tmpdirname+"/train_set.csv", index=False)
        df_test.to_csv(tmpdirname+"/test_set.csv", index=False)
        
        mlflow.log_artifacts(tmpdirname, artifact_path=f"{cv_str}data")
        mlflow.log_artifact(args.data_path, artifact_path=f"{cv_str}data")

def reshape_classification_head(model, args, class_names):
    num_classes = len(class_names)
    if args.model == 'resnet18' or args.model == 'resnet34' or args.model == 'resnet50' or args.model == 'resnet101' or args.model == 'resnet152' or args.model == 'resnext50_32x4d' or args.model == 'resnext101_32x8d' or args.model == 'wide_resnet50_2' or args.model == 'wide_resnet101_2' or args.model == 'shufflenet_v2_x0_5' or args.model == 'shufflenet_v2_x1_0':
        num_ftrs = model.fc.in_features
        # (fc): Linear(in_features=512, out_features=1000, bias=True)
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif args.model == 'squeezenet1_1':
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
    elif args.model == 'densenet121' or args.model == 'densenet169' or args.model == 'densenet201' or args.model == 'densenet161':
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f'changing classification head for model "{args.model}" not supported, exiting...')
    return model
        

class CSVDataset(object):
    def __init__(self, root, path_to_csv, transforms, dataset_type='binary', class_names=[]):
        self.root = root
        self.transforms = transforms
        df = pd.read_csv(path_to_csv)
        
        self.imgs = list(df.iloc[:, 0])
        self.labels = list(df[df.columns[1]])
        
        if dataset_type == 'binary':
            self.classes = class_names if class_names else ['non_'+df.columns[1], df.columns[1]]
        elif dataset_type == 'multi_class':
            self.classes = class_names if class_names else df.iloc[:, 1].unique() 
        elif dataset_type == 'multi_label':
            raise NotImplementedError(f'dataset_type "{dataset_type}" not yet implemented')
        else:
            raise ValueError(f'invalid value for dataset_type "{dataset_type}". Valid values are "binary", "multi_class" and "multi_label"')

    def __getitem__(self, idx):
        # load images and labels
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path)
        target = self.labels[idx]

        if self.transforms is not None:
            try:
                img = self.transforms(img)
            except Exception as e:
                raise Exception(f"Error trying to transform {img_path}: {e}")

        return img, target

    def __len__(self):
        return len(self.imgs)
    
def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('labelBalance', utils.SmoothedValue(window_size=1, fmt='{global_avg}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    print(f'#### {datetime.datetime.now().strftime("%d.%m.%y %H:%M")} - Start Epoch {epoch} train') 
    header = f'Epoch: [{epoch}] Batch:'
    f1 = F1Score(num_classes=2).to(device)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc2 = utils.accuracy(output, target, topk=(1, 2))
        f1_score = f1(output, target)
        # print(f"Target {target}, output: {output}, acc1: {acc1.item()}")
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['f1_score'].update(f1_score.item(), n=batch_size)
        metric_logger.meters['labelBalance'].update(torch.sum(target), n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
        
    print(f'#### {datetime.datetime.now().strftime("%d.%m.%y %H:%M")} - End Epoch {epoch} train') 
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg / 100



def evaluate(model, criterion, data_loader, device, class_names, print_freq=100, cv_str=""):
    num_classes = len(class_names)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        f1 = F1Score(num_classes=num_classes).to(device)
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            
            #calc acc per class
            _, preds = torch.max(output, 1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            acc1, acc2 = utils.accuracy(output, target, topk=(1, 2))
            f1_score = f1(output, target)
        
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['f1_score'].update(f1_score.item(), n=batch_size)
            #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size) 
        # check for distributed mode, if yes, synchronize confusion matrix across all nodes
        if utils.is_dist_avail_and_initialized():
            confusion_matrix = confusion_matrix.to(device)
            torch.distributed.barrier()
            torch.distributed.all_reduce(confusion_matrix)
        print(confusion_matrix)
        recall_per_class = confusion_matrix.diag()/confusion_matrix.sum(1)
        recall_per_class[recall_per_class != recall_per_class] = 0 # set all nans to 0
        precision_per_class = confusion_matrix.diag()/confusion_matrix.sum(0)
        precision_per_class[precision_per_class != precision_per_class] = 0 # set all nans to 0
        
        recall_per_class_list = []
        precision_per_class_list = []
        f1_per_class_list = []
        for i in range(len(recall_per_class)):
            metric_logger.meters[f'{cv_str}recall_per_class_{i}'].update(recall_per_class[i].item(), n=1)
            metric_logger.meters[f'{cv_str}precision_per_class_{i}'].update(precision_per_class[i].item(), n=1)
            f1 = 2*(precision_per_class[i].item()*recall_per_class[i].item()) / (precision_per_class[i].item()+recall_per_class[i].item())
            metric_logger.meters[f'{cv_str}f1_per_class_{i}'].update(f1, n=1)
            recall_per_class_list.append(recall_per_class[i].item())
            precision_per_class_list.append(precision_per_class[i].item())
            f1_per_class_list.append(f1)
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(' * Acc@1 {top1.global_avg:.3f}'
          .format(top1=metric_logger.acc1))
    print(' * F1@1 {top1.global_avg:.3f}'
          .format(top1=metric_logger.f1_score))
    
    recall_dict = {}
    precision_dict = {}
    f1_dict = {}
    for i in range(len(recall_per_class_list)):
        recall_dict[f'{cv_str}Recall '+class_names[i]] = recall_per_class_list[i]
        precision_dict[f'{cv_str}Precision '+class_names[i]] = precision_per_class_list[i]
        f1_dict['F1 '+class_names[i]] = f1_per_class_list[i]
    
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg / 100, metric_logger.f1_score.global_avg, recall_dict, precision_dict, f1_dict

def retry(cb, max_retries=5, err_msg=None):
    i = 0
    exception = None
    while(i<max_retries):
        i += 1
        try:
            return cb()
        except Exception as e:
            exception = e
            print(traceback.format_exc())
            if err_msg is None:
                print(e)
            else:
                print(err_msg.format(e))
            print("Restart mlflow")
            try:
                MlflowTracker.reconnect()
            except Exception as me:
                print(f"Mlflw restart failed: {me}")
            time.sleep(10)
    raise exception
        

def load_data(args, cv):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()

    dataset_train = CSVDataset(
        args.imgs_path,
        data_paths(args, cv)[0],
        transforms.Compose([
            transforms.RandomResizedCrop(tuple(args.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
        args.dataset_type,
        args.class_names)
    print("Took", time.time() - st)

    print("Loading validation data")
    dataset_test = CSVDataset(
        args.imgs_path,
        data_paths(args, cv)[1],
        transforms.Compose([
            transforms.Resize(tuple(args.input_size)),
            transforms.ToTensor(),
            normalize,]),
        args.dataset_type,
        args.class_names)

    class_names = dataset_train.classes
    
    if args.data_path:
        splitpoint=int(len(dataset_train)*(1-args.test_split))
        indices = torch.randperm(len(dataset_train)).tolist()

        dataset_train = torch.utils.data.Subset(dataset_train, indices[:splitpoint])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[splitpoint:])
    
    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        if args.balance_samples:
            # oversample minority class, so that model sees roughly same num_samples from each class
            train_sampler = ImbalancedDatasetSampler(dataset_train, labels=dataset_train.labels)
            print(f"Using Train imbalancedDatasetSampler with weights: {train_sampler.weights}")
            # torch.utils.data.WeightedRandomSampler(weights[train_ys], (max(class_sample_count)*2))
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)
        if args.balance_val_set:
            # oversample minority class, so that model sees roughly same num_samples from each class
            test_sampler = ImbalancedDatasetSampler(dataset_test, labels=dataset_test.labels)
            print(f"Using TEST imbalancedDatasetSampler with weights: {test_sampler.weights}")
        else:
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        
    if args.data_path:
        log_data_split(args, indices, splitpoint)
    else:
        retry(lambda: mlflow.log_artifact(data_paths(args, cv)[0], artifact_path=f"{cv_str(cv)}data"), 5)
        retry(lambda: mlflow.log_artifact(data_paths(args, cv)[1], artifact_path=f"{cv_str(cv)}data"), 5)
    
    return dataset_train, dataset_test, train_sampler, test_sampler, class_names

def output_dir(args, cv):
    #return args.output_dir.format(cv=cv)
    return args.output_dir+'/'+args.run_name+'/'+str(cv)
    # return args.cv_output_dirs[cv]

def data_paths(args, cv):
    # return args.cv_data_paths[cv]
    return [p.format(cv=cv) for p in args.data_paths]

def cv_str(cv):
    return f"cv{cv}_"

def inner_main(args, trial=None, cv=0, resume=False):
    if utils.is_main_process():
        if args.output_dir:
            if Path(output_dir(args, cv)).exists():
                shutil.rmtree(output_dir(args, cv))
            utils.mkdir(output_dir(args, cv))

#    utils.init_distributed_mode(args)
    print(args)

    dataset, dataset_test, train_sampler, test_sampler, class_names = load_data(args, cv)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)

    model = reshape_classification_head(model, args, class_names)
    if args.device == "cuda":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device_arg}")
    device = torch.device(device_arg)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device, cv_str=cv_str(cv))
        return
    
    print("Start training")
    start_time = time.time()
    best_model_performance = 0.0
    best_model_epoch = 0
    
    if utils.is_main_process() and not resume:
        # Log parameters on first execution 
        for key, value in vars(args).items():
            mlflow.log_param(key, value)
            
    early_stopping = EarlyStopping(patience=10, verbose=True)
    print(f"Total epochs {args.epochs}")
    for epoch in range(args.start_epoch, args.epochs):
        print(f"Epoch {epoch}")
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        loss, acc, f1_score, recall_per_class, precision_per_class, f1_per_class = evaluate(model, criterion, data_loader_test, device=device, class_names=class_names, cv_str=cv_str(cv))
                
        f1_macro = sum([v for v in f1_per_class.values()]) / len(f1_per_class.values())
        if utils.is_main_process():
            # log metrics
            retry(lambda: mlflow.log_metric(f'{cv_str(cv)}Accuracy_Train', train_acc, step=epoch), 5)
            retry(lambda: mlflow.log_metric(f'{cv_str(cv)}Accuracy_Val', acc, step=epoch), 5)
            retry(lambda: mlflow.log_metric(f'{cv_str(cv)}Loss_Train', train_loss, step=epoch), 5)
            retry(lambda: mlflow.log_metric(f'{cv_str(cv)}Loss_Val', loss, step=epoch), 5)
            retry(lambda: mlflow.log_metric(f'{cv_str(cv)}F1_Micro_Val', f1_score, step=epoch), 5)
            retry(lambda: mlflow.log_metric(f'{cv_str(cv)}F1_Macro_Val', f1_macro, step=epoch), 5)
            retry(lambda: mlflow.log_metrics(recall_per_class, step=epoch), 5)
            retry(lambda: mlflow.log_metrics(precision_per_class, step=epoch), 5)
                
            # prune run (trial) if its not going well, but only on cv0, otherwise runs will be incomplete
            if trial is not None and not resume:
                intermediate_value = f1_score
                trial.report(intermediate_value, epoch)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            #track best model
            if f1_score > best_model_performance:
                best_model = copy.deepcopy(model_without_ddp)
                best_model_optimizer = copy.deepcopy(optimizer)
                best_model_lr_scheduler = copy.deepcopy(lr_scheduler)
                best_model_performance = f1_score
                best_model_epoch = epoch

                #save best model locally
                # Log every best model to prevent wasting progress
                if args.output_dir:
                    checkpoint = {
                        'model': best_model.state_dict(),
                        'optimizer': best_model_optimizer.state_dict(),
                        'lr_scheduler': best_model_lr_scheduler.state_dict(),
                        'last_epoch': best_model_epoch,
                        'args': args}
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(output_dir(args, cv), 'model.pth'))
                    with open(os.path.join(output_dir(args, cv), 'best_model_epoch.log'), "a") as epoch_log:
                        epoch_log.write(f"Logged model at epoch {best_model_epoch}\n")
                        
            early_stopping(loss)
            #if early_stopping.early_stop:
                #print("Early stopping")
                #break
            
    # log model
    if utils.is_main_process():
        # also save best model performance as last step performance
        retry(lambda: mlflow.log_metric(f'{cv_str(cv)}F1_Best', best_model_performance, best_model_epoch), 5) # "Could not log final accuracy {e}"
            
        if args.log_roc:
            print("\nCalc and log ROC")
            calc_log_ROC(best_model, data_loader_test, device=args.device, classes=class_names, cv_str=cv_str(cv))
        
        if args.log_model:
            print("\nLogging the trained model as a run artifact...")
            retry(lambda: mlflow.log_artifact(os.path.join(output_dir(args, cv), 'model.pth'), artifact_path=f"{cv_str(cv)}pytorch-model"), 5) # "Could not log final model {e}"
            # retry(lambda: mlflow.pytorch.log_model(best_model, artifact_path=f"{args.cv_str}pytorch-model", pickle_module=pickle), 5) # "Could not log final model {e}"
            print("\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(), cv_str(cv), "pytorch-model"))
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return best_model_performance

def main(args, trial=None, cv=0, run_uuid=None, user_attr={}, system_attr={}):
    torch.manual_seed(1)
    MlflowTracker.initialize(args.experiment_name, args.run_name, args.run_uuid)
    MlflowTracker.connect()
    resume = False
    if args.run_uuid is not None:
        resume = True

    with MlflowTracker.active_run:
        if not resume:
            for key, value in user_attr.items():
                if trial is not None:
                    trial.set_user_attr(key, value)
                mlflow.set_tag(key, value)
            for key, value in system_attr.items():
                if trial is not None:
                    trial.set_system_attr(key, value)
                mlflow.set_tag(key, value)
        try:
            return inner_main(args, trial, cv, resume)
        except Exception as e:
            print(traceback.format_exc())
            print(f"{e}")
            return 0

class StoreDictKeyPair(argparse.Action):
     def __call__(self, parser, namespace, values, option_string=None):
         my_dict = {}
         for kv in values.split(","):
             k,v = kv.split("=")
             my_dict[k] = v
         setattr(namespace, self.dest, my_dict)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data-path', help='path to csv containing dataset')
    group.add_argument('--data-paths', nargs=2, help='path to train- and test-csv containing labels and paths to images')
    parser.add_argument('--imgs-path', default='/', help='Root folder containing all the images with relative paths in data-path(s).')
    parser.add_argument('--run_name', default='default', help='name of the training run')
    parser.add_argument('--experiment-name', help='name of the experiment')
    parser.add_argument('--input-size', type=int, nargs=2, default=[150,150], help='shape to which images are resized for training: (h, w)')
    parser.add_argument('--test-split', type=float, default=0.2, help='portion of data to use for testing')
    parser.add_argument('--model', default='resnet50', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.003, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=10, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--run-uuid', dest="run_uuid", default =None, help='')
    
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        #action="store_true",
        default=True
    )
    # distributed training parameters
    parser.add_argument('--distributed', default=False)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--log-model', action='store_true', dest="log_model", help='tore final model in mlflow')
    parser.add_argument('--no-log-model', action='store_false', dest="log_model", help='do not store final model in mlflow')
    parser.add_argument('--log-roc', action='store_true', dest="log_roc", help='store final roc in mlflow')
    parser.add_argument('--no-log-roc', action='store_false', dest="log_roc", help='do not store final roc in mlflow')
    parser.add_argument('--balance_samples', action='store_true', dest="balance_val_set", help='')
    parser.set_defaults(balance_samples=False)
    parser.set_defaults(log_model=True)
    parser.set_defaults(log_roc=True)
    parser.add_argument('--dataset-type', dest="dataset_type", default="binary", help='Dataset classification type. Can be "binary", "multi_class" or "multi_label"')
    parser.add_argument('--class-names', dest="class_names", type=str, nargs="+", default=[], help='Class names in order. I.e. --class_names label0 label1 label2')
    parser.add_argument('--min-memory', type=int, dest="min_memory", help='GPU Memory required for execution (Trying to prevent RuntimeError). Batch 2 of 1920x2560 Images ~ 32849788928, Batch 2 of 1920x2560 Images ~ 27293384704')
    parser.add_argument('-cv','--cross-validation', dest="nr_cv", default=1, type=int, help='Number of cross validation runs')
    parser.add_argument('--cv-start', dest="cv_start", default=0, type=int, help='Number of cross validation runs')
    parser.add_argument("--user_attr", dest="user_attr", action=StoreDictKeyPair, default={}, metavar="KEY1=VAL1,KEY2=VAL2...")
    parser.add_argument("--system_attr", dest="system_attr", action=StoreDictKeyPair, default={}, metavar="KEY1=VAL1,KEY2=VAL2...")

    args = parser.parse_args()
    print(f"Log Model: {args.log_model}")
    print(f"Log Roc: {args.log_roc}")
    return args


if __name__ == "__main__":
    args = parse_args()
    run_uuid = args.run_uuid
       
    metric = 0
    user_attr = {}
    system_attr = {}
    for i in range(args.cv_start, args.nr_cv): 
        print(f"START CV {i}")
        metric += main(args, None, i, args.run_uuid, user_attr, system_attr)
        print(f"END CV {i}")
         
    print(metric / (args.nr_cv - args.cv_start))
    
    
