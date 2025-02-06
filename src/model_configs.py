# model_configs.py

# Configuration for model building
model_configs = {
    'v1': {
        'input_shape': (224, 224, 3),
        'num_classes': 1,
        'batch_size': 32,
        'epochs': 5,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss_function': 'cross_entropy',
        'model': 'ResNet-18 (ImageNet Pretrained)',
        'model_path': 'models/model_v1.pth',
        'threshold': 0.55,
        'hair_removal': False
    },
    'v2': {
        'input_shape': (224, 224, 3),
        'num_classes': 1,
        'batch_size': 32,
        'epochs': 5,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss_function': 'focal_loss',
        'model': 'ResNet-18 (ImageNet Pretrained)',
        'model_path': 'models/model_v2.pth',
        'threshold': 0.45,
        'hair_removal': False
    }

    ,
    'v3': {
        'input_shape': (224, 224, 3),
        'num_classes': 1,
        'batch_size': 32,
        'epochs': 5,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss_function': 'cross_entropy',
        'metrics': ['accuracy', 'f1_score', 'precision', 'recall'],
        'model': 'EfficientNet-B0 (ImageNet Pretrained)',
        'model_path': 'models/model_v3.pth',
        'threshold': 0.28,
        'hair_removal': False
    },

    'v4': {
        'input_shape': (224, 224, 3),
        'num_classes': 1,
        'batch_size': 32,
        'epochs': 3,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss_function': 'cross_entropy',
        'model': 'ResNet-18 (ImageNet Pretrained)',
        'model_path': 'models/model_v4.pth',
        'threshold': 0.45,
        'hair_removal': True
    },

    'v5': {
        'input_shape': (224, 224, 3),
        'num_classes': 1,
        'batch_size': 32,
        'epochs': 5,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss_function': 'cross_entropy',
        'model': 'ResNet-18 (ImageNet Pretrained)',
        'model_path': 'models/model_v4.pth',
        'threshold': 0.45,
        'hair_removal': False   
    },

    'v4': {
        'input_shape': (224, 224, 3),
        'num_classes': 1,
        'batch_size': 32,
        'epochs': 5,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss_function': 'bce',
        'model': 'ResNet-18 (ImageNet Pretrained)',
        'model_path': 'models/model_v4.pth',
        'threshold': 0.45,
    },

    'ensemble': {
        'models': ['v1', 'v2', 'v3'],
        'threshold': 0.4,
        'hair_removal': False
    }
}

# Function to print the configuration
def print_config(config):
    for key, value in config.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    for model_name, config in model_configs.items():
        print(f"\nModel Training Summary for {model_name}")
        print(f"Model: {config['model']}")
        print(f"Loss Function: {config['loss_function']}")
        print(f"Optimizer: {config['optimizer']}")
        print(f"Learning Rate: {config['learning_rate']}")
        if 'training_summary' in config:
            print("Training Summary:")
            for metric, value in config['training_summary'].items():
                print(f"  {metric}: {value}")