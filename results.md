# Cancer Detection Project

This project focuses on detecting cancer using various deep learning models. The models and data used in this project can be found at the following link:
[Google Drive - Cancer Detection Project](https://drive.google.com/drive/folders/1pmu139n2qBkxdw46nkIdDuYNMXN3pEe9?usp=sharing)

## Preprocessing
- **Data Transformations**:
  - **Train**:
    - Hair Removal (Optional)
    - Resize to (224, 224)
    - Random Horizontal Flip
    - Convert to Tensor
    - Normalize with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]
  - **Test**:
    - Hair Removal (Optional)
    - Resize to (224, 224)
    - Convert to Tensor
    - Normalize with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]

## Post-processing
- **Thresholding**: Apply a threshold to the sigmoid output to determine the final class label.
- **Metrics Calculation**: Calculate accuracy, precision, recall, and F1 score based on the predicted and true labels.

## Model Training Summary

### Model_v1
- **Model**: ResNet-18 (ImageNet Pretrained)
- **Loss Function**: Cross Entropy Loss
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Epochs**: 5
- **Threshold**: 0.55
- **Hair Removal**: False
- **Metrics**:
  - **F1 Score**: 0.9258713794796269
  - **Precision**: 0.9093539054966249
  - **Recall**: 0.943
  - **Accuracy**: 0.9245

### Model_v2
- **Model**: ResNet-18 (ImageNet Pretrained)
- **Loss Function**: Focal Loss
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Epochs**: 5
- **Threshold**: 0.45
- **Hair Removal**: False
- **Metrics**:
  - **F1 Score**: 0.9211045364891519
  - **Precision**: 0.9085603112840467
  - **Recall**: 0.934
  - **Accuracy**: 0.92

### Model_v3
- **Model**: EfficientNet-B0 (ImageNet Pretrained)
- **Loss Function**: Cross Entropy Loss
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Epochs**: 5
- **Threshold**: 0.28
- **Hair Removal**: False
- **Metrics**:
  - **F1 Score**: 0.942495126705653
  - **Precision**: 0.9192015209125475
  - **Recall**: 0.967
  - **Accuracy**: 0.941

### Model_v4
- **Model**: ResNet-18 (ImageNet Pretrained)
- **Loss Function**: Cross Entropy Loss
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Epochs**: 5
- **Threshold**: 0.45
- **Hair Removal**: True
- **Metrics**:
  - **F1 Score**: 0.8519537699504678
  - **Precision**: 0.9473684210526315
  - **Recall**: 0.774
  - **Accuracy**: 0.8655

  ### Model_v5
  - **Model**: ResNet-18 (ImageNet Pretrained)
  - **Loss Function**: Cross Entropy Loss
  - **Optimizer**: Adam
  - **Learning Rate**: 0.001
  - **Epochs**: 5
  - **Threshold**: 0.45
  - **Hair Removal**: False
  - **Metrics**:
    - **F1 Score**: 0.897
    - **Precision**: 0.897
    - **Recall**: 0.897
    - **Accuracy**: 0.897

### Ensemble Model
- **Models**: v1, v2, v3
- **Threshold**: 0.5
- **Metrics**:
  - **F1 Score**: 0.9472111553784861
  - **Precision**: 0.9434523809523809
  - **Recall**: 0.951
  - **Accuracy**: 0.947

## Summary Table

| Model           | F1 Score          | Precision         | Recall           | Accuracy         | Threshold | Training Time Hair removal | Testing Time Hair Removal |
|-----------------|-------------------|-------------------|------------------|------------------|-----------|-------------------|------------------|
| Model_v1        | 0.9258713794796269| 0.9093539054966249| 0.943            | 0.9245           | 0.55      | No               | No              |
| Model_v2        | 0.9211045364891519| 0.9085603112840467| 0.934            | 0.92             | 0.45      | No               | No              |
| Model_v3        | 0.942495126705653 | 0.9192015209125475| 0.967            | 0.941            | 0.28      | No               | No             |
| Model_v4       | 0.8519537699504678 | 0.9473684210526315 | 0.774 | 0.8655 | 0.45 | Yes | Yes |
| Model_v5 (No Hair removal at test time) | 0.897 | 0.897 | 0.897 | 0.897 | 0.45 | Yes | No |   |
| Ensemble Model  | 0.9472111553784861| 0.9434523809523809| 0.951            | 0.947            | 0.5       | No               | No              |

For more details about the hair removal process, please refer to the `hair_removal.ipynb` notebook.
