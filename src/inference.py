import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import re 

import argparse

from model_configs import model_configs

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv

# Define data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load test dataset
data_dir = 'ml_takehome_dataset/test'
test_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load the model
def load_model(model_path):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

#EfficientNet
def initialize_efficientnet(v=0,model_path = 'models/model_v3.pth'):
    model = getattr(models, f'efficientnet_b{v}')(pretrained=True)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_model(model_params):
    if model_params['model'] == 'ResNet-18 (ImageNet Pretrained)':
        return load_model(model_params['model_path'])
    else:
        version = re.findall(r'\d+', model_params['model'])[0]
        print(version)
        return initialize_efficientnet(v=version, model_path=model_params['model_path'])
    
def load_ensemble_model(args):
    models = []

    for model_name in model_configs['ensemble']['models']:
        models.append(get_model(model_configs[model_name]))

    return models

# Run inference
def run_inference(models, dataloader, threshold=0.5,csv_path='inference_results.csv'):
    if not isinstance(models, list):
        models = [models]

    preds = []
    targets = []
    file_paths = []
    logits = []
    losses = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = [model(inputs) for model in models]
            avg_outputs = torch.mean(torch.stack(outputs), dim=0)
            predicted = (torch.sigmoid(avg_outputs) > threshold).long()
            preds.extend(predicted.numpy())
            targets.extend(labels.numpy())
            # file_paths.extend([path[0] for path in dataloader.dataset.samples])
            logits.extend(torch.sigmoid(avg_outputs).numpy())
            loss = nn.BCEWithLogitsLoss()(avg_outputs, labels.float().view(-1, 1)).item()
            for i in range(len(labels)):
                loss = nn.BCEWithLogitsLoss()(avg_outputs[i].view(-1,1), labels[i].float().view(-1, 1)).item()
                losses.append(loss)
    file_paths = [path[0] for path in dataloader.dataset.samples]

    
    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    print(f'F1 Score: {f1}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Accuracy: {accuracy}')

    # Save results to CSV
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File Path', 'Target', 'Prediction', 'Logits', 'Loss'])
        for file_path, target, pred, logit, loss in zip(file_paths, targets, preds, logits, losses):
            writer.writerow([file_path, target, pred, logit, loss])
    
    print(len(targets), len(preds), len(file_paths), len(logits), len(losses))


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('--model', type=str, required=True, help='Model version to use for inference', choices=['v1', 'v2', 'v3','ensemble'])
argument_parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for classification')
argument_parser.add_argument('--csv', type=str, default='inference_results.csv', help='Path to save inference results')
argument_parser.add_argument('--override', type=bool, default=False, help='Override model threshold with argument')
args = argument_parser.parse_args()

model_params = model_configs[args.model]

if args.model in ['v1', 'v2', 'v3']:
    model = get_model(model_configs[args.model])
else:
    model = load_ensemble_model(args)
# Check if threshold is provided in the model config
if 'threshold' in model_params and not args.override:
    threshold = model_params['threshold']
    print(f'Using threshold from config: {threshold}')
else:
    threshold = args.threshold

# Run inference
run_inference(model, test_dataloader, threshold=threshold, csv_path=args.csv)
