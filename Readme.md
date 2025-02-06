# Cancer Detection Project

This project focuses on detecting cancer using various deep learning models. The models and data used in this project can be found at the following link:

[Google Drive - Cancer Detection Project](https://drive.google.com/drive/folders/1pmu139n2qBkxdw46nkIdDuYNMXN3pEe9?usp=sharing)

## Directory Structure

Ensure that the necessary files and folders are placed in the root directory of the repository. The expected directory structure is as follows:

```
Cancer-Detection-Project/
├── __pycache__/
├── ml_takehome_dataset/
├── models/
├── cancer_detection.yml
├── focal_loss.py
├── inference_results.md
├── inference.py
├── model_configs.py
├── Readme.md
├── results.md
├── train.py
└── voxel_viz.py
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Cancer-Detection-Project.git
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
To train the model, run:
```bash
python train.py
```

### Running Inference
To perform inference using a trained model, run:
```bash
python inference.py
```

### Visualizing Results
To visualize voxel data, use:
```bash
python voxel_viz.py
```

### Model Configuration
Modify `model_configs.py` to change model hyperparameters and settings.

## Results
Inference results are stored in `inference_results.md`, and other results are logged in `results.md`.

## Contributing
If you'd like to contribute, please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License.
