import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
import mlflow
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
import pandas as pd
import os
from PIL import Image
from pathlib import Path

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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
    
    print("Loading Data")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
print("Loading Training Data")



# Set MLflow tracking URI (you can customize this)
mlflow.set_tracking_uri("http://localhost:5000")

def calculate_occlusion_map(model, image, label, window_size=5, stride=1):
    image = image.unsqueeze(0)  # Add batch dimension and move to GPU if available
    orig_prediction = model(image).argmax().item()

    # Generate occlusion map
    height, width = image.shape[-2:]
    patches = view_as_windows(image, (3, window_size, window_size), (1, stride, stride))
    occlusion_map = np.zeros((height, width))

    for i in range(0, height - window_size + 1, stride):
        for j in range(0, width - window_size + 1, stride):
            occluded_image = image.clone()
            occluded_image[:, :, i:i+window_size, j:j+window_size] = 0  # Occlude the region
            occluded_prediction = model(occluded_image).argmax().item()
            occlusion_map[i:i+window_size, j:j+window_size] = orig_prediction - occluded_prediction

    return occlusion_map


def train_model(args, fold, train_indices, val_indices):
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with mlflow.start_run(run_name=f"{args.name}_Fold {fold + 1}_{args.experiment_name}"):
        # Log hyperparameters
        mlflow.log_params(vars(args))
        mlflow.log_param("fold", fold + 1)

        # Data Loaders
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                                   num_workers=args.num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=val_sampler,
                                                 num_workers=args.num_workers, pin_memory=True)
        

        print(f"Fold {fold + 1} Training:")
        print(f"Batch Size: {args.batch_size}")
        print(f"Number of Epochs: {args.num_epochs}")
        print(f"Fold: {fold + 1}")

        for epoch in range(args.num_epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            print("Epoch: ", epoch)

            for images, labels in train_loader:
                #images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_accuracy = 100 * correct / total
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
        

        # ... Validation and metrics computation ...
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            predicted_list = []
            val_labels_list = []
            val_labels_list = torch.tensor(val_labels_list)
            predicted_list = torch.tensor(predicted_list)
            
            for val_images, val_labels in val_loader:
                
                print("val_label",val_labels.size())
                val_preds = model(val_images)
                
                print("val_preds", val_preds.size())
                loss = criterion(val_preds, val_labels)
                val_loss += loss.item()
                _, predicted = val_preds.max(1)
                predicted_list = torch.cat((predicted_list, predicted), 0)
                val_labels_list = torch.cat((val_labels_list, val_labels), 0)
                print("predicted: ", predicted)
               

        val_accuracy = accuracy_score(val_labels_list, predicted_list)
        
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        #mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        
        val_precision = precision_score(val_labels_list, predicted_list, average='macro', zero_division=1)
        mlflow.log_metric("val_precision", val_precision)
        val_recall = recall_score(val_labels_list, predicted_list, average='macro', zero_division=1)
        mlflow.log_metric("val_recall", val_recall)
        #val_auroc = roc_auc_score(val_labels, val_preds)
        #mlflow.log_metric("val_auroc", val_auroc)
        #val_confusion_matrix = confusion_matrix(val_labels, val_preds)
        
        
        
    return model, val_accuracy, val_precision, val_recall
        
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Validation with ResNet-50 and MLflow Integration")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument('--data_paths', nargs=2)
    parser.add_argument('--experiment_name', default='', help='name of the experiment')
    parser.add_argument('--name', default='default', help='name of the training run')
    parser.add_argument('--imgs-path', default='data_small', help='folder containing all the images')
    parser.add_argument('--input-size', type=int, nargs=2, default=[150,150], help='shape to which images are resized for training: (h, w)')
    parser.add_argument('--dataset-type', default='multi_class', help='binary/multi_class', dest='dataset_type')
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)
    
    train_dataset = CSVDataset(args.imgs_path,
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

    test_dataset = CSVDataset(args.imgs_path,
                              args.data_paths[1],
                              transforms.Compose([transforms.Resize(tuple(args.input_size)),
                                                  transforms.ToTensor(),
                                                  normalize]),
                              dataset_type=args.dataset_type)



    print("Testing data loaded")
    print("---------------------------------------")
    
    p = Path('/beegfs/ws/1/issr292b-workspace_MK1/classification_hema_pipeline/')
    
    # Device setup
    model = torchvision.models.resnet50(pretrained=True)
    num_classes = len(train_dataset.classes)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    print("Training Overview:")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Epochs: {args.num_epochs}")
    print(f"Number of Folds: {args.k_folds}")

    # Initialize k-fold cross-validation
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    # Lists to store metrics for each fold
    accuracy_list = []
    precision_list = []
    recall_list = []
    auroc_list = []
    occlusion_maps_list = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
        if len(train_indices) == 0 or len(val_indices) == 0:
            print(f"Fold {fold + 1} has insufficient data for training or validation.")
            continue
        
        print("train_indices: ", len(train_indices))
        print("val_indices: ", len(val_indices))
        
        model_e, val_accu, val_prec, val_rec = train_model(args, fold, train_indices, val_indices)
        
        torch.save(model_e.state_dict(), Path(p,'resnet50_fold_{}.pth'.format(fold)))       
        
        accuracy_list.append(val_accu)
        precision_list.append(val_prec)
        recall_list.append(val_rec)
                

    # ... Average metrics and final test evaluation ...
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    # Calculate and log average metrics across all folds
    average_val_accuracy = np.mean(accuracy_list)
    average_precision = np.mean(precision_list)
    average_recall = np.mean(recall_list)
    #average_auroc = np.mean(auroc_list)

    mlflow.log_metric("avg_val_accuracy", average_val_accuracy)
    mlflow.log_metric("avg_val_precision", average_precision)
    mlflow.log_metric("avg_val_recall", average_recall)
    #mlflow.log_metric("avg_val_auroc", average_auroc)

    print(f"Average Metrics - "
          f"Avg Validation Accuracy: {average_val_accuracy:.4f}, "
          f"Avg Precision: {average_precision:.4f}, "
          f"Avg Recall: {average_recall:.4f}")
          #f"Avg AUROC: {average_auroc:.4f}")

    # Now, evaluate the model on the test set
    model.eval()
    test_preds = []
    test_labels = []
    test_occlusion_maps = []

    with torch.no_grad():
        for images, test_labels in test_loader:
            
            test_preds = model(images)
            _, preds = torch.max(test_preds, 1)
            #test_preds.extend(preds.cpu().numpy())
            #test_labels.extend(labels.cpu().numpy())

            # Calculate occlusion maps for a sample of images
            #if len(test_occlusion_maps) < 5:  # Calculate occlusion maps for the first 5 images
            #    for i in range(len(images)):
            #        occlusion_map = calculate_occlusion_map(model, images[i], labels[i])
            #        test_occlusion_maps.append(occlusion_map)

    test_accuracy = accuracy_score(test_labels, preds)
    test_precision = precision_score(test_labels, preds, average='macro', zero_division=1)
    test_recall = recall_score(test_labels, preds, average='macro', zero_division=1)
    #test_auroc = roc_auc_score(test_labels, test_preds)
    #test_confusion_matrix = confusion_matrix(test_labels, test_preds)

    # Log test metrics and artifacts to MLflow
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    #mlflow.log_metric("test_auroc", test_auroc)
    #mlflow.log_artifact(plot_roc_curve(fpr, tpr))  # Log ROC curve to MLflow
    #mlflow.log_artifact(plot_confusion_matrix(test_confusion_matrix))  # Log confusion matrix to MLflow

    print(f"Test Metrics - "
          f"Test Accuracy: {test_accuracy:.4f}, "
          f"Test Precision: {test_precision:.4f}, "
          f"Test Recall: {test_recall:.4f}")
          #f"Test AUROC: {test_auroc:.4f}")
    
    # Visualize occlusion maps for a sample of images
#    for i, occlusion_map in enumerate(test_occlusion_maps):
#        plt.imshow(occlusion_map, cmap='jet', alpha=0.8)
#        plt.colorbar()
#        plt.title(f"Occlusion Map for Image {i + 1}")
#        plt.savefig(f"occlusion_map_image_{i + 1}.png")
#        mlflow.log_artifact(f"occlusion_map_image_{i + 1}.png")
#        plt.close()


    class ensemblemodel(nn.Module):
        def __init__(self, *models):
            super().__init__
            self.models = nn.ModuleList(models)
            self.num_models = len(models)
            self.classifier = nn.Linear(200*self.num_models, 200)
            #check this again, numbers not fitting properly
            
            
            def forward(self, x):
                model_outputs = [model(x) for model in self.models]
                x = torch.cat(model_outputs, dim=1)
                out = self.classifier(x)
                
                return out
            
            
    ensemble_model = ensemblemodel(m1, m2, m3)
    
    for params in ensemble_model.parameters():
        param.require_grad = False
        
    for params in ensemble_model.classifier.parameters():
        param.require_grad = True
        
    