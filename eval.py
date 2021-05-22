import pathlib
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
# from model import Net
import matplotlib.pyplot as plt

from torch import nn
import tqdm
import copy
import timm
import argparse
import natsort
import os
from PIL import Image
import pandas as pd
import torchattacks
import foolbox as fb
from utils.utils import ValidationSet, update_bn_params
from models.other_layers import SparseAttention
from models.DoubleViT import DoubleViT

model_to_arch = {
    "vit" : "vit_large_patch16_224_in21k",
    "inception_resnet_v2": "inception_resnet_v2",
    "pit" : "pit_b_distilled_224",
    "doublevit": "asdf"
}
def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--model', "-m" , type=str, default="inception_resnet_v2")
    add_arg('--checkpoint', '-c', type=str)
    add_arg('--check_robustness', '-cr', type=int, default=0)
    add_arg('--sparse_attn_k', '-sa', type=int, default=0)
    add_arg('--update_bn', '-bn', type=int, default=0)
    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    # Create a pytorch dataset
    data_dir = pathlib.Path('./tiny-imagenet-200/')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} images'.format(image_count))

    # Create the training data generator
    batch_size = 32
    im_height = 64
    im_width = 64
    

    im_height=224
    im_width=224


    transform_test = transforms.Compose([
        transforms.Resize((im_height,im_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    torch.cuda.device("cuda:0")
    device = "cuda:0"

    if args.model in model_to_arch:
         if args.model == "doublevit":
            model = DoubleViT(224)
        else:
            model = timm.create_model(model_to_arch[args.model], pretrained=True)    
    else:
        print("model does not exist")
    
    if args.model == "inception_resnet_v2":
        num_ftrs = model.classif.in_features
        model.classif = nn.Sequential(
                      nn.Dropout(0.4),
                      nn.Linear(num_ftrs, 1024),
                      nn.ReLU(),
                      nn.Linear(1024, 256),
                      nn.ReLU(),
                      nn.Linear(256, 200))
    elif args.model == "pit":
        num_ftrs = model.head.in_features
        if args.sparse_attn_k:
            for transformer in model.transformers:
                for block in transformer.blocks:
                    block.attn = SparseAttention(block.attn, args.sparse_attn_k)
        model.head =  nn.Sequential(
                      nn.Dropout(0.4),
                      nn.Linear(num_ftrs, 1024), 
                      nn.ReLU(),
                      nn.Linear(1024, 256),
                      nn.ReLU(),
                      nn.Linear(256, 200))
        model.head_dist = nn.Sequential(
                      nn.Dropout(0.4),
                      nn.Linear(num_ftrs, 1024), 
                      nn.ReLU(),
                      nn.Linear(1024, 256),
                      nn.ReLU(),
                      nn.Linear(256, 200))
    elif args.model == "vit":
        num_ftrs = model.head.in_features
        if args.sparse_attn_k:
            for block in model.blocks:
                block.attn = SparseAttention(block.attn, args.sparse_attn_k)
        model.head =  nn.Sequential(
                      nn.Dropout(0.4),
                      nn.Linear(num_ftrs, 1024), 
                      nn.ReLU(),
                      nn.Linear(1024, 256),
                      nn.ReLU(),
                      nn.Linear(256, 200))
        
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['net'])
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    
    validation_set = ValidationSet(data_dir / 'val', transform_test)
#     robusta.batchnorm.adapt(model, adapt_type="batch_wise")
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)
    if args.update_bn:
        model = update_bn_params(model, val_loader, args.update_bn, device)  
    model.eval()
    if args.check_robustness:
        fmodel = fb.PyTorchModel(model, bounds=(0,1))
        attack = fb.attacks.LinfPGD(steps=10)
        epsilons = [0.1]
        robust_accuracies = []

    all_preds = []
    all_labels = []
    all_losses = []
    index = 0
    for batch in tqdm.tqdm(val_loader):
        inputs = batch[0]
        targets = batch[1]
        targets = targets.cuda()
        inputs = inputs.cuda()
        if args.check_robustness:
            _, advs, success = attack(fmodel, inputs, targets, epsilons=epsilons) 
            robust_accuracy = 1 - success.cpu().numpy().astype(np.float).mean(axis=-1)
            robust_accuracies.append(robust_accuracy)
    #         print("robust accuracy for perturbations with")
    #         for eps, acc in zip(epsilons, robust_accuracy):
    #             print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
        with torch.no_grad():
            preds = model(inputs)
        loss = nn.CrossEntropyLoss()(preds, targets)
        all_losses.append(loss.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(targets.cpu())
    if args.check_robustness:
        print("Top 1 Validation Accuracy (robust): {}".format(np.mean(np.array(robust_accuracies))))

    
    top_preds = [x.argsort(dim=-1)[:,-1:].squeeze() for x in all_preds]
    correct = 0
    for idx, batch_preds in enumerate(top_preds):
        correct += torch.eq(all_labels[idx], batch_preds).sum()
    print(correct.item())
    print(len(all_labels))
    accuracy = correct.item() / (32 * len(all_labels))
    print(f"Top 1 Validation Accuracy (clean): {accuracy}")

    top_preds = [x.argsort(dim=-1)[:,-3:] for x in all_preds]
    correct = 0
    for idx, batch_preds in enumerate(top_preds):
        correct += torch.eq(all_labels[idx], batch_preds[:,0:1].squeeze()).sum()
        correct += torch.eq(all_labels[idx], batch_preds[:,1:2].squeeze()).sum()

        correct += torch.eq(all_labels[idx], batch_preds[:,2:3].squeeze()).sum()

    accuracy = correct.item() / (32 * len(all_labels))
    print(f"top 3 Validation Accuracy (clean): {accuracy}")
    

if __name__ == '__main__':
    main()
