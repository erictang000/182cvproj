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

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--model', "-m" , type=str, default="inception_resnet_v2")
    add_arg('--checkpoint', '-c', type=str)
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



    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    torch.cuda.device("cuda:0")
    device = "cuda:0"


    model = timm.create_model(args.model, pretrained=True)
        
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.classif.in_features
    model.fc =  nn.Sequential(
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
    
    validation_set = torchvision.datasets.ImageFolder(data_dir / 'test', transform_test)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

    model.eval()
    all_preds = []
    all_labels = []
    all_losses = []
    with torch.no_grad():
        index = 0
        for batch in tqdm.tqdm(val_loader):
            inputs = batch[0]
            targets = batch[1]
            if index == 0:
                print(inputs, targets)
                index += 1
            targets = targets.cuda()
            inputs = inputs.cuda()
            preds = model(inputs)
            loss = nn.CrossEntropyLoss()(preds, targets)
            all_losses.append(loss.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(targets.cpu())
    top_preds = [x.argsort(dim=-1)[:,-1:].squeeze() for x in all_preds]
    correct = 0
    for idx, batch_preds in enumerate(top_preds):
        correct += torch.eq(all_labels[idx], batch_preds).sum()
    print(correct.item())
    print(len(all_labels))
    accuracy = correct.item() / (32 * len(all_labels))
    print(f"Top 1 Validation Accuracy: {accuracy}")

    top_preds = [x.argsort(dim=-1)[:,-3:] for x in all_preds]
    correct = 0
    for idx, batch_preds in enumerate(top_preds):
        correct += torch.eq(all_labels[idx], batch_preds[:,0:1].squeeze()).sum()
        correct += torch.eq(all_labels[idx], batch_preds[:,1:2].squeeze()).sum()

        correct += torch.eq(all_labels[idx], batch_preds[:,2:3].squeeze()).sum()

    accuracy = correct.item() / (32 * len(all_labels))
    print(f"top 3 Validation Accuracy: {accuracy}")
    

if __name__ == '__main__':
    main()
