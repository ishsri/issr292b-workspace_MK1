import torchvision
import random
import torch
import os
from PIL import Image
from torchvision import transforms
import json
from torch import nn

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, augmentations):
        super(MyDataset, self).__init__()
        
        f = open(json_path)
        locations = json.load(f)
        img_list = locations['cell_slices']
        
        self.img_list = img_list
        self.augmentations = augmentations
        self.master_img = Image.open(locations['src_img'])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.master_img.crop(self.img_list[idx])
        return self.augmentations(img)
    
class MyDatasetThreadsave(torch.utils.data.Dataset):
    def __init__(self, json_path, augmentations):
        super(MyDatasetThreadsave, self).__init__()
        
        f = open(json_path)
        locations = json.load(f)
        img_list = locations['cell_slices']
        master_img = Image.open(locations['src_img'])
        master_img = transforms.functional.to_tensor(master_img)
        img_list = [transforms.functional.resized_crop(master_img, int(x[1]),int(x[0]),int(x[3])-int(x[1]),int(x[2])-int(x[0]),(250,250)) for x in img_list]
        
        self.img_list = img_list
        self.augmentations = augmentations

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        return self.augmentations(img)

def reshape_classification_head(model, model_architecture, num_classes):
    if model_architecture in ['resnet18',
                              'resnet34',
                              'resnet50',
                              'resnet101',
                              'resnet152',
                              'resnext50_32x4d',
                              'resnext101_32x8d',
                              'wide_resnet50_2',
                              'wide_resnet101_2',
                              'shufflenet_v2_x0_5',
                              'shufflenet_v2_x1_0']:
        num_ftrs = model.fc.in_features
        # (fc): Linear(in_features=512, out_features=1000, bias=True)
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_architecture in ['squeezenet1_1']:
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
    elif model_architecture in ['densenet121',
                                'densenet169',
                                'densenet201',
                                'densenet161']:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f'changing classification head for model "{model_architecture}" not supported, exiting...')
    return model

def load_model(args):
    class_names = ['AML',
                  'APL',
                  'Healthy']
    
    device = torch.device('cpu')
    model = torchvision.models.__dict__[args.model]()
    model = reshape_classification_head(model, args.model, len(class_names))
    
    #load weights from checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    model.eval()

    return model

def predict(args):
    
    if args.pretrained_models_dir:
        os.environ['TORCH_HOME'] = args.pretrained_models_dir
        
    model = load_model(args)

    device = torch.device('cpu')
    model.eval()

    f = open(args.json_path)
    locations = json.load(f)

    img = Image.open(locations['src_img'])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = transforms.Compose([transforms.Resize((250,250)),
                                          transforms.ToTensor(),
                                          normalize])
    predictions = []
    with torch.no_grad():
        for entry in locations['cell_slices']:
            print(entry)
            img2 = img.crop(entry)
            img2 = data_transforms(img2)
            img2 = img2.unsqueeze(0)
            prediction = model(img2.to(device))
            predictions.append(int(torch.argmax(prediction)))
#            print(int(torch.argmax(prediction)))
    return predictions
            

def predict_batched(args):
    
    if args.pretrained_models_dir:
        os.environ['TORCH_HOME'] = args.pretrained_models_dir
        
    model = load_model(args)

    device = torch.device('cpu')
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = transforms.Compose([transforms.Resize((250,250)),
                                          transforms.ToTensor(),
                                          normalize])
    
    dataset = MyDataset(args.json_path, data_transforms)
    test_sampler = torch.utils.data.SequentialSampler(dataset)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=1, pin_memory=True)

    predictions = []
    with torch.no_grad():
        for image in data_loader:
            output = model(image)
            preds = torch.argmax(output, dim=1)
            predictions += [int(x) for x in preds]
            # [print(int(x)) for x in preds]
    return predictions
	
def predict_batched_multithread(args):
    
    if args.pretrained_models_dir:
        os.environ['TORCH_HOME'] = args.pretrained_models_dir
        
    model = load_model(args)

    device = torch.device('cpu')
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = transforms.Compose([transforms.Resize((250,250)),
                                          normalize])
    
    dataset = MyDatasetThreadsave(args.json_path, data_transforms)
    test_sampler = torch.utils.data.SequentialSampler(dataset)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)


    predictions = []
    with torch.no_grad():
        for image in data_loader:
            output = model(image)
            preds = torch.argmax(output, dim=1)
            predictions += [int(x) for x in preds]
#            [print(int(x)) for x in preds]
    return predictions
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', help='path to input json')
    parser.add_argument('--model_path', help='path to model')
    parser.add_argument('--pretrained_models_dir', help='path where pytorch stores its standard models locally')
    parser.add_argument('--model', default='resnet50', help='model')
    parser.add_argument('--not_batched', action="store_true", help='turn off batched inference, might speed up inference on bad CPUs')
    parser.add_argument('--single_thread', action="store_true", help='turn off multithreading')
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers')
    
    args = parser.parse_args()
    predictions = []
    if args.not_batched:
        predictions = predict(args)
    elif args.single_thread:
        predictions = predict_batched(args)
    else:
        predictions = predict_batched_multithread(args)
    print(predictions)