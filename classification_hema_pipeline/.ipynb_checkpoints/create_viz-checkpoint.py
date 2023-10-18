import torch
import torch.nn.functional as F

from PIL import Image

import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms
import torch.nn as nn

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from pathlib import Path


def get_pretrained_model(model_path, class_names, device='cuda'):

    model = torchvision.models.__dict__['resnet50'](pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.to(device)
    
    #load weights from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    model.eval()
    return model

def get_classes(dataset_dir):
    dataset = torchvision.datasets.ImageFolder(dataset_dir)
    classes = dataset.classes
    return classes

def process(img_path, model, mode, outdir):

    #load image
    img = Image.open(img_path)

    transformed_img = transform(img)

    input = transform_normalize(transformed_img)
    input = input.unsqueeze(0)
    input = input.to('cuda')

    #inference
    output = model(input)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    prediction_score_float = prediction_score.squeeze().item()

    pred_label_idx.squeeze_()
    predicted_label = classes[pred_label_idx]

    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')], N=256)

    label = img_path.split('/')[-2]
    classification_correct = predicted_label == label

    if mode == 'grad':
        integrated_gradients = IntegratedGradients(model)
        attributions_ig = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=200, internal_batch_size=1)

        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              ["original_image", "heat_map","blended_heat_map", "masked_image"],
                                              ["all", "positive", "positive", "positive"],
                                              [f'{classification_correct} \n groundtruth: {label}\n predicted class: {predicted_label}\n confidence: {prediction_score_float}',"heat_map","blended_heat_map", "masked_image"],
                                              (40, 12),
                                              cmap=default_cmap,
                                              show_colorbar=True)
    elif mode == 'occ':
        occlusion = Occlusion(model)

        attributions_occ = occlusion.attribute(input,
                                               strides = (3, 20, 20),
                                               target=pred_label_idx,
                                               sliding_window_shapes=(3,30, 30),
                                               baselines=0)

        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              ["original_image", "heat_map","blended_heat_map", "masked_image"],
                                              ["all", "positive", "positive", "positive"],
                                              [f'{classification_correct} \n groundtruth: {label}\n predicted class: {predicted_label}\n confidence: {prediction_score_float}',"heat_map","blended_heat_map", "masked_image"],
                                              (40, 12),
                                              show_colorbar=True,
                                              outlier_perc=2,
                                             )

    test = '/'.join(img_path.split('/')[-3:])
    #####
#    test = test[:-3] + 'jpeg'
    ####
    save_location = os.path.join(outdir, mode, test)
    os.makedirs(os.path.dirname(save_location), exist_ok=True)

    _[0].savefig(save_location, dpi=600)
    
if __name__ == "__main__":
    
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location', default='../data_all/npm1')
    parser.add_argument('--mode', default='occ', help='grad or occ')
    parser.add_argument('--model_path', default = 'model_npm1.pth')
    parser.add_argument('--target_location', default='/lustre/scratch2/ws/0/s2558947-hema_pytorch/wholeslide_class/viz')
    parser.add_argument('--img_width', type=int, help='width the input images are scaled to', default=1500)
    parser.add_argument('--img_height', type=int, help='width the input images are scaled to', default=2000)
    args = parser.parse_args()

    classes=get_classes(args.data_location)
    model = get_pretrained_model(args.model_path, classes)

    #transforms
    transform = transforms.Compose([
     transforms.Resize((args.img_width, args.img_height)),
     #transforms.Resize((600,800)),
     transforms.ToTensor()
    ])

    transform_normalize = transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]
     )

    # create list of already existing graphs to skip them in for loop
    p_out = Path(os.path.join(args.target_location, args.mode))
    existing_outs = [str(i).split('/')[-3:] for i in p_out.rglob('*.png')]
    
    # iterate over inputs
    p_in = Path(args.data_location)
    for i in p_in.rglob('*.png'):  
        if str(i).split('/')[-3:] in existing_outs:
            continue
        process(str(i), model, args.mode, args.target_location)
        print(f'{i} processed!')
        
        
#        a = str(i).split('/')[-1]
#        b = a.split('-')[0]
#        if b == 'pat10008' or b == 'pat22294' or b == 'pat23536' or b == 'pat29519':
#            print('yes')
#            process(str(i), model, args.mode, args.target_location)
#            print(f'{i} processed!')
        
    print('done!')