import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
from focal_loss import FocalLoss
from hair_removal import hair_remove
import numpy as np
from PIL import Image

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import argparse
import cv2
import numpy as np
from PIL import Image

def hair_remove(image):
    """
    Remove hair from the image using inpainting (Source: https://www.kaggle.com/code/youssefhatem1/melanoma-hair-remove)
    """
    # convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1,(17,17))
    
    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    
    # apply thresholding to blackhat
    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    
    # inpaint with original image and threshold image
    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)
    
    return final_image

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Lambda(lambda img: Image.fromarray(hair_remove(np.array(img)))),
        transforms.Resize((224, 224)),
        # transforms.Lambda(lambda img: Image.fromarray(hair_remove(np.array(img)))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Lambda(lambda img: Image.fromarray(hair_remove(np.array(img)))),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
data_dir = 'ml_takehome_dataset'
image_datasets = {x: datasets.ImageFolder(root=f'{data_dir}/{x}', transform=data_transforms[x]) for x in ['train', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'test']}

def initialize_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

#EfficientNet
def initialize_efficientnet(v=0):
    model = getattr(models, f'efficientnet_b{v}')(pretrained=True)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=1)
    return model

def validate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    preds = []
    targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs) > 0.5).long()
            preds.extend(predicted.numpy())
            targets.extend(labels.numpy())
    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    print(f'F1 Score: {f1}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Accuracy: {accuracy}')
    return accuracy

# Train the model
def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(dataloaders['train']):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        print(f'Loss: {epoch_loss:.4f}')
        validate_model(model, dataloaders['test'])
    print('Training complete')

# Argument parser
parser = argparse.ArgumentParser(description='Train a ResNet model for cancer detection')
parser.add_argument('--model_name', type=str, default='model.pth', help='Name of the model file to save')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model')
parser.add_argument('--loss', type=str, default='bce', help='Loss function to use (focal or bce)', choices=['focal', 'bce'])
parser.add_argument('--efficientnet', type=int, default=-1, help='Use EfficientNet model with specified version')
args = parser.parse_args()

# Initialize the model
if args.efficientnet != -1: # Use EfficientNet model
    print(f'Using EfficientNet B{args.efficientnet}')
    model = initialize_efficientnet(args.efficientnet)
else: # Use ResNet model
    print('Using ResNet18')
    model = initialize_model()

# Initialize the EfficientNet model


# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Define the loss function
if args.loss == 'focal':
    criterion = FocalLoss()
else:
    criterion = nn.BCEWithLogitsLoss()

# Train the model
train_model(model, dataloaders, criterion, optimizer, num_epochs=args.epochs)



# Save the model with the specified name
model_path = f'models/{args.model_name}'
if '.pth' not in model_path:
    model_path += '.pth'
print(f'Saving model to {model_path}')
torch.save(model.state_dict(), model_path)

# Validate the model
validate_model(model, dataloaders['test'])

