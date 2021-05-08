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

class ValidationSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir / 'images'
        labels_csv = pd.read_csv(main_dir / 'val_labels.csv')
        self.img_to_class = {}
        image_names = labels_csv["image_name"]
        classes = labels_csv["class"]
        for i in range(len(image_names)):
            self.img_to_class[image_names[i]] = classes[i]
        self.class_to_idx = class_to_idx = {'n01443537': 0, 'n01629819': 1, 'n01641577': 2, 'n01644900': 3, 'n01698640': 4, 'n01742172': 5, 'n01768244': 6, 'n01770393': 7, 'n01774384': 8, 'n01774750': 9, 'n01784675': 10, 'n01855672': 11, 'n01882714': 12, 'n01910747': 13, 'n01917289': 14, 'n01944390': 15, 'n01945685': 16, 'n01950731': 17, 'n01983481': 18, 'n01984695': 19, 'n02002724': 20, 'n02056570': 21, 'n02058221': 22, 'n02074367': 23, 'n02085620': 24, 'n02094433': 25, 'n02099601': 26, 'n02099712': 27, 'n02106662': 28, 'n02113799': 29, 'n02123045': 30, 'n02123394': 31, 'n02124075': 32, 'n02125311': 33, 'n02129165': 34, 'n02132136': 35, 'n02165456': 36, 'n02190166': 37, 'n02206856': 38, 'n02226429': 39, 'n02231487': 40, 'n02233338': 41, 'n02236044': 42, 'n02268443': 43, 'n02279972': 44, 'n02281406': 45, 'n02321529': 46, 'n02364673': 47, 'n02395406': 48, 'n02403003': 49, 'n02410509': 50, 'n02415577': 51, 'n02423022': 52, 'n02437312': 53, 'n02480495': 54, 'n02481823': 55, 'n02486410': 56, 'n02504458': 57, 'n02509815': 58, 'n02666196': 59, 'n02669723': 60, 'n02699494': 61, 'n02730930': 62, 'n02769748': 63, 'n02788148': 64, 'n02791270': 65, 'n02793495': 66, 'n02795169': 67, 'n02802426': 68, 'n02808440': 69, 'n02814533': 70, 'n02814860': 71, 'n02815834': 72, 'n02823428': 73, 'n02837789': 74, 'n02841315': 75, 'n02843684': 76, 'n02883205': 77, 'n02892201': 78, 'n02906734': 79, 'n02909870': 80, 'n02917067': 81, 'n02927161': 82, 'n02948072': 83, 'n02950826': 84, 'n02963159': 85, 'n02977058': 86, 'n02988304': 87, 'n02999410': 88, 'n03014705': 89, 'n03026506': 90, 'n03042490': 91, 'n03085013': 92, 'n03089624': 93, 'n03100240': 94, 'n03126707': 95, 'n03160309': 96, 'n03179701': 97, 'n03201208': 98, 'n03250847': 99, 'n03255030': 100, 'n03355925': 101, 'n03388043': 102, 'n03393912': 103, 'n03400231': 104, 'n03404251': 105, 'n03424325': 106, 'n03444034': 107, 'n03447447': 108, 'n03544143': 109, 'n03584254': 110, 'n03599486': 111, 'n03617480': 112, 'n03637318': 113, 'n03649909': 114, 'n03662601': 115, 'n03670208': 116, 'n03706229': 117, 'n03733131': 118, 'n03763968': 119, 'n03770439': 120, 'n03796401': 121, 'n03804744': 122, 'n03814639': 123, 'n03837869': 124, 'n03838899': 125, 'n03854065': 126, 'n03891332': 127, 'n03902125': 128, 'n03930313': 129, 'n03937543': 130, 'n03970156': 131, 'n03976657': 132, 'n03977966': 133, 'n03980874': 134, 'n03983396': 135, 'n03992509': 136, 'n04008634': 137, 'n04023962': 138, 'n04067472': 139, 'n04070727': 140, 'n04074963': 141, 'n04099969': 142, 'n04118538': 143, 'n04133789': 144, 'n04146614': 145, 'n04149813': 146, 'n04179913': 147, 'n04251144': 148, 'n04254777': 149, 'n04259630': 150, 'n04265275': 151, 'n04275548': 152, 'n04285008': 153, 'n04311004': 154, 'n04328186': 155, 'n04356056': 156, 'n04366367': 157, 'n04371430': 158, 'n04376876': 159, 'n04398044': 160, 'n04399382': 161, 'n04417672': 162, 'n04456115': 163, 'n04465501': 164, 'n04486054': 165, 'n04487081': 166, 'n04501370': 167, 'n04507155': 168, 'n04532106': 169, 'n04532670': 170, 'n04540053': 171, 'n04560804': 172, 'n04562935': 173, 'n04596742': 174, 'n04597913': 175, 'n06596364': 176, 'n07579787': 177, 'n07583066': 178, 'n07614500': 179, 'n07615774': 180, 'n07695742': 181, 'n07711569': 182, 'n07715103': 183, 'n07720875': 184, 'n07734744': 185, 'n07747607': 186, 'n07749582': 187, 'n07753592': 188, 'n07768694': 189, 'n07871810': 190, 'n07873807': 191, 'n07875152': 192, 'n07920052': 193, 'n09193705': 194, 'n09246464': 195, 'n09256479': 196, 'n09332890': 197, 'n09428293': 198, 'n12267677': 199}

        self.transform = transform
        all_imgs = os.listdir(self.main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        label = self.class_to_idx[self.img_to_class[self.total_imgs[idx]]]
        tensor_image = self.transform(image)
        return tensor_image, label

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
    # Parameters of newly constructed modules have requires_grad=True by default
    if args.model == "inception_resnet_v2":
        num_ftrs = model.classif.in_features
        model.classif = nn.Sequential(
                      nn.Dropout(0.4),
                      nn.Linear(num_ftrs, 1024),
                      nn.ReLU(),
                      nn.Linear(1024, 256),
                      nn.ReLU(),
                      nn.Linear(256, 200))
    else:
        num_ftrs = model.head.in_features
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
    #validation_set = torchvision.datasets.ImageFolder(data_dir / 'test', transform_test)
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
                print("input")
                print(inputs)
                print("targets")
                print(targets)
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
