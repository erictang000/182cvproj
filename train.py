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

# Create a pytorch dataset
data_dir = pathlib.Path('/data/tiny-imagenet/')
image_count = len(list(data_dir.glob('**/*.JPEG')))
CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
print('Discovered {} images'.format(image_count))

# Create the training data generator
batch_size = 32
im_height = 64
im_width = 64

# data_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
# ])

data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
#     transforms.RandomCrop(64, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
train_set = torchvision.datasets.ImageFolder(data_dir / 'train', data_transforms)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=4, pin_memory=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# model_path = 'deepaugment_and_augmix.pth.tar'
# output_path = 'deepaug_and_mix.pt'
num_epochs = 30


model = timm.create_model('inception_resnet_v2', pretrained=True)
# checkpoint = torch.load(model_path)
# model = nn.DataParallel(model)
# model.load_state_dict(checkpoint['state_dict'])
# model = model.module
# Create a simple model
for param in list(model.parameters())[:-30]:
    param.requires_grad = False
    
# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.classif.in_features

model.fc =  nn.Sequential(
                  nn.Dropout(0.4),
                  nn.Linear(num_ftrs, 1024), 
                  nn.ReLU(),
                  nn.Linear(1024, 256),
                  nn.ReLU(),
                  nn.Linear(256, 200))
model = model.to(device)
# model = Net(len(CLASS_NAMES), im_height, im_width)
optim = torch.optim.Adam(
    [
        {"params": list(model.parameters())[-30:-6], "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3}
    ])
criterion = nn.CrossEntropyLoss()

for i in range(num_epochs):
    train_total, train_correct = 0,0
    
    ##evaluate
    validation_set = torchvision.datasets.ImageFolder(data_dir / 'val', transform_test)
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
    accuracy = correct.item() / (32 * len(all_labels))
    print(f"Top 1 Validation Accuracy: {accuracy}") 
    
    top_preds = [x.argsort(dim=-1)[:,-3:] for x in all_preds]
    correct = 0
    for idx, batch_preds in enumerate(top_preds):
        correct += torch.eq(all_labels[idx], batch_preds[:,0:1].squeeze()).sum()
        correct += torch.eq(all_labels[idx], batch_preds[:,1:2].squeeze()).sum()

        correct += torch.eq(all_labels[idx], batch_preds[:,2:3].squeeze()).sum()

    accuracy = correct.item() / (32 * len(all_labels))
    print(f"Top 3 Validation Accuracy: {accuracy}")

    model.train()
    for idx, (inputs, targets) in enumerate(train_loader):
        # print(inputs[1])
        # plt.imshow(inputs[1].permute(1,2,0))
        # plt.show()
        inputs, targets = inputs.to(device), targets.to(device)
        optim.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optim.step()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()
        # if idx % 100 == 0:
        print("\r", end='')
        print(f'training {100 * idx / len(train_loader):.2f}%: {train_correct / train_total:.3f}', end='')
    torch.save({
        'net': model.state_dict(),
    }, "results/inceptionresnetsv2{}.pt".format(i))
