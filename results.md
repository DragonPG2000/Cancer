# Model Training Summary

## Model_v1
- **Model**: ResNet-18 (ImageNet Pretrained)
- **Loss Function**: Cross Entropy Loss
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Epochs**: 5
- **Threshold**: 0.55
- **Metrics**:
  - **F1 Score**: 0.9258713794796269
  - **Precision**: 0.9093539054966249
  - **Recall**: 0.943
  - **Accuracy**: 0.9245

## Model_v2
- **Model**: ResNet-18 (ImageNet Pretrained)
- **Loss Function**: Focal Loss
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Epochs**: 5
- **Threshold**: 0.45
- **Metrics**:
  - **F1 Score**: 0.9211045364891519
  - **Precision**: 0.9085603112840467
  - **Recall**: 0.934
  - **Accuracy**: 0.92