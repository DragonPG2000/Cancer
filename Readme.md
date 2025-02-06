# Cancer Detection Project

This project focuses on detecting cancer using various deep learning models. The models and data used in this project can be found at the following link:

[Google Drive - Cancer Detection Project](https://drive.google.com/drive/folders/1pmu139n2qBkxdw46nkIdDuYNMXN3pEe9?usp=sharing)

## Directory Structure

Ensure that the necessary files and folders are placed in the root directory of the repository. The expected directory structure is as follows:

```
Cancer-Detection-Project/
├── __pycache__/
├── ml_takehome_dataset/
│   ├── test/
│   ├── train/
├── models/
│   ├── model_v1.pth
│   ├── model_v2.pth
│   ├── model_v3.pth
├── results/
│   ├── inference_ensemble.csv
│   ├── inference_v1.csv
│   ├── inference_v2.csv
│   ├── inference_v3.csv
├── src/
│   ├── __pycache__/
│   ├── focal_loss.py
│   ├── inference.py
│   ├── model_configs.py
│   ├── train.py
├── cancer_detection.yml
├── Readme.md
├── results.ipynb
├── results.md
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/DragonPG2000/Cancer-Detection-Project.git
   ```

2. Navigate into the project directory:
   ```bash
   cd Cancer-Detection-Project
   ```

3. Set up the environment using the `cancer_detection.yml` file:
   ```bash
   conda env create -f cancer_detection.yml
   conda activate cancer_detection
   ```

## Usage

### Training the Model
To train the model, run (Example command):
```bash
python src/train.py --model_name model_v3 --efficientnet 0
```
Replace `model_v3` with the desired filename you want to store the weights in. The `--efficientnet` argument specifies the version of EfficientNet to use. Set it to `-1` for ResNet-18, or specify the version of EfficientNet you want to use.

### Command Line Arguments

The training script `train.py` accepts several command line arguments to customize the training process. Below are the available arguments:

- `--model_name`: Name of the model file to save (default: `model.pth`)
- `--lr`: Learning rate for the optimizer (default: `0.001`)
- `--epochs`: Number of epochs to train the model (default: `5`)
- `--loss`: Loss function to use, either `focal` or `bce` (default: `bce`)
- `--efficientnet`: Use EfficientNet model with specified version (default: `-1` for ResNet-18)

These arguments allow you to specify the model name, learning rate, number of epochs, loss function, and EfficientNet version for training.

### Running Inference
To perform inference using a trained model, run:
```bash
python src/inference.py --model [v1|v2|v3|ensemble]
```
Replace `[v1|v2|v3|ensemble]` with the desi (best performing model is the ensemble)

### Model Configuration
The file `model_configs.py` contains the information on what hyperparameters were used for training each model

## Results
Inference results are stored in the `results/` directory as CSV files:
- `inference_ensemble.csv`
- `inference_v1.csv`
- `inference_v2.csv`
- `inference_v3.csv`

### Visualizing Results
To analyze results and visualize data, use:
```bash
jupyter notebook results.ipynb
```

Other results and logs are documented in `results.md`.
