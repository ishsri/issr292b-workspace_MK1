# load model an evaluate on selected dataset
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms

#from torch.utils.tensorboard import SummaryWriter

import utils

from train import *


def main(args):
    
    if args.pretrained_models_dir:
        os.environ['TORCH_HOME'] = args.pretrained_models_dir
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset_test = torchvision.datasets.ImageFolder(
        args.data_path,
        transforms.Compose([
            transforms.Resize((1500,2000)),
            #transforms.CenterCrop(1900),
            transforms.ToTensor(),
            normalize,
        ]))

    class_names = dataset_test.classes
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=4, pin_memory=True)
    

    model = torchvision.models.__dict__['resnet50'](pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.to(args.device)
    
    #load weights from checkpoint
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    acc, recall_per_class, precision_per_class = evaluate(model, criterion, data_loader_test, args.device, len(class_names))
    
    recall_dict = {}
    precision_dict = {}
    for i in range(len(recall_per_class)):
        recall_dict['Recall '+ class_names[i]] = recall_per_class[i]
        precision_dict['Precision '+ class_names[i]] = precision_per_class[i]
        
    print('Accuracy:')
    print(acc)
    print('Precision:')
    print(precision_dict)
    print('Recall:')
    print(recall_dict)



        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data_path', help='path to data on which to eval on')
    parser.add_argument('-m','--model_path', help='path to model')
    parser.add_argument('-p','--pretrained_models_dir', help='path where pytorch stores its standard models locally')
    parser.add_argument('--device', default='cuda', help='cpu or cuda')
    args = parser.parse_args()
    
    main(args)