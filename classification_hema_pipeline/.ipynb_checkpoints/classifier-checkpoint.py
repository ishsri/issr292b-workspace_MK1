import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from pathlib import Path 
from sklearn.model_selection import KFold 
import argparse
import os
from PIL import Image

#setting random seeds for reproducibility

torch.manual_seed(42)
np.random.seed(42)

ap = argparse.ArgumentParser(description= 'Parameters input for classifier')

ap.add_argument('--data_paths', nargs=2)
ap.add_argument('--imgs-path', default='data_small', help='folder containing all the images')
ap.add_argument('--input-size', type=int, nargs=2, default=[150,150], help='shape to which images are resized for training: (h, w)')
ap.add_argument('--dataset-type', default='multi_class', help='binary/multi_class', dest='dataset_type')
ap.add_argument('--kfold', default = 5, help='division of training set into k subset')
ap.add_argument('-b', '--batch-size', default=32, type=int)
ap.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16), rule of thumb: 4-8 workers per GPU')
ap.add_argument('--lr', default=0.003, type=float, help='initial learning rate')
ap.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
ap.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
ap.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
ap.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')

args = ap.parse_args()


#data loading and preprocessing
class CSVDataset(object):
    def __init__(self, root, path_to_csv, transforms, dataset_type='binary'):
        self.root = root
        self.transforms = transforms
        df = pd.read_csv(path_to_csv)
        
        self.imgs = list(df.iloc[:, 0])
        
        if dataset_type == 'binary':
            self.labels = list(df[df.columns[1]])
            self.classes = ['non_'+df.columns[1], df.columns[1]]
        elif dataset_type == 'multi_class':
            class_id_dict = {value:count for (count, value) in enumerate(sorted(df['label'].unique()))}
            labels = list(df[df.columns[1]])
            self.classes = [key for key in class_id_dict]
            self.labels = [class_id_dict.get(item,item) for item in labels]
        elif dataset_type == 'multi_label':
            raise NotImplementedError(f'dataset_type "{dataset_type}" not yet implemented')
        else:
            raise ValueError(f'invalid value for dataset_type "{dataset_type}". Valid values are "binary", "multi_class" and "multi_label"')

    def __getitem__(self, idx):
        # load images and labels
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        target = self.labels[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    
#CSV dataset to dataloader

print("Loading Data")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
print("Loading Training Data")

dataset_train = CSVDataset(args.imgs_path,
                               args.data_paths[0],
                               transforms.Compose([transforms.Resize(tuple(args.input_size)),
                                                   transforms.RandomResizedCrop(tuple(args.input_size)),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   normalize]),
                               dataset_type=args.dataset_type)

print("Training data loaded")
print("---------------------------------------")
print("Loading Testing Data")

dataset_test = CSVDataset(args.imgs_path,
                              args.data_paths[1],
                              transforms.Compose([transforms.Resize(tuple(args.input_size)),
                                                  transforms.ToTensor(),
                                                  normalize]),
                              dataset_type=args.dataset_type)

classes = dataset_train.classes

print("Testing data loaded")
print("---------------------------------------")

#model creation

model = torchvision.models.resnet50(pretrained=True)
num_classes = len(classes)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

print("Model created.")
print("---------------------------------------")


k = args.kfold
kf = KFold(n_splits=k, shuffle=True, random_state=42)

accuracy_list = []
precision_list = []
recall_list = []
auroc_list = []


print("Initiating main loop")

for fold, (train_indices, val_indices) in enumerate (kf.split(dataset_train)):
    
    if len(train_indices) == 0 or len(val_indices) == 0:
        print(f"Fold {fold + 1} has insufficient data for training or validation.")
        continue
            
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)
    
    data_loader_val = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=val_sampler,
                                                 num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False)
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Fold {fold + 1} Training:")
    
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        
        train_loss = 0.0
        correct = 0
        total = 0

        
        for images, labels in data_loader_train:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
                        
        print("training completed for", epoch, "/", args.epochs)
        train_accuracy = 100 * correct / total
        #print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print("Train accuracy = ", train_accuracy)
            
            
    #evaluation on validation set
    
    
    print("---------------------------------------")
    print("Running Evaluation for ", epoch)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader_val:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='macro')
    auroc = roc_auc_score(all_labels, all_preds)

    accuracy_list.append(val_accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    auroc_list.append(auroc)
    
    print(f"Fold {fold + 1} Metrics - "
          f"Accuracy_val: {val_accuracy:.4f}, "
          f"Precision_val: {precision:.4f}, "
          f"Recall_val: {recall:.4f}, "
          f"AUROC_val: {auroc:.4f}")

# Calculate and print average metrics across all folds
average_val_accuracy = np.mean(accuracy_list)
average_precision = np.mean(precision_list)
average_recall = np.mean(recall_list)
average_auroc = np.mean(auroc_list)

print(f"Average Metrics testing set - "
      f"Validation Accuracy: {average_val_accuracy:.4f}, "
      f"Average Precision: {average_precision:.4f}, "
      f"Average Recall: {average_recall:.4f}, "
      f"Average AUROC: {average_auroc:.4f}")