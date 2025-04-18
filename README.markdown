# Eye Disease Classification: Comparing VGG and MobileNet with Knowledge Distillation

## Project Overview
This project implements and compares deep learning models for diabetic retinopathy (DR) classification using the APTOS 2019 Blindness Detection dataset. The study focuses on two primary architectures—modified VGG-16 and MobileNet—and explores knowledge distillation techniques to optimize model efficiency. A customized VGG-16 serves as the teacher model, transferring knowledge to a MobileNet student model, achieving high accuracy with reduced computational complexity.

## Dataset
- **Source**: APTOS 2019 Blindness Detection dataset
- **Size**: 3,662 high-resolution fundus photographs
- **Labels**: 5 severity levels of diabetic retinopathy (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative)
- **Challenges**: Class imbalance (50% Class 0), variability in lighting, camera devices, and image quality
- **Preprocessing**:
  - Black border removal using grayscale thresholding
  - Contrast enhancement with CLAHE (LAB color space)
  - Normalization using ImageNet statistics
- **Augmentation**:
  - Random flips, rotations (±20°), color jitter
- **Split**: 80-20 stratified train-validation split

## Models
### VGG-16 (Modified)
- **Base**: Pretrained VGG16 with batch normalization (ImageNet weights)
- **Custom Head**: Fully-connected layers (25,088 → 4096 → 1024 → 5) with ReLU and dropout
- **Parameters**: ~134M (25M+ trainable)
- **Performance**: 
  - Validation Accuracy: ~84.84%
  - Quadratic Weighted Kappa: ~0.9171

### MobileNet
- **Base**: MobileNetV2 pretrained on ImageNet
- **Custom Head**: Fully-connected layers (1280 → 512 → 128 → 5) with ReLU and dropout
- **Parameters**: ~2.2M
- **Performance (with distillation)**:
  - Validation Accuracy: ~83.74% (α=0.1), ~83.61% (α=0.01)
  - Quadratic Weighted Kappa: ~0.9049 (α=0.1), ~0.8996 (α=0.01)

### EfficientNet-B0
- **Base**: Pretrained EfficientNet-B0 (ImageNet weights)
- **Custom Head**: Fully-connected layers (1280 → 256 → 5) with ReLU and dropout
- **Parameters**: ~4.0M
- **Performance**:
  - Validation Accuracy: ~84.43%
  - Quadratic Weighted Kappa: ~0.9104

### Knowledge Distillation
- **Teacher**: EfficientNet-B0
- **Student**: MobileNetV2
- **Loss**: Combined Cross-Entropy (hard targets) + KL Divergence (soft targets)
- **Configurations**:
  - α=0.5, T=4
  - α=0.1, T=3
  - α=0.01, T=3
- **Results**: Distillation maintains high performance with fewer parameters, especially at α=0.1 (Student Accuracy: ~83.74%, Kappa: ~0.9049).

## Training Setup
- **Optimizer**: AdamW (lr=1e-4 or 5e-5, weight decay=1e-4 or 1e-5)
- **Scheduler**: OneCycleLR or CosineAnnealingLR
- **Batch Size**: 32
- **Epochs**: 15 (early stopping, patience=3)
- **Techniques**: Mixed-precision training, label smoothing (ε=0.1)

## Evaluation Metrics
- **Primary**: Quadratic Weighted Kappa (Cohen’s Kappa) for ordinal classification
- **Secondary**: Classification Accuracy

## Repository Structure
```
├── data/                   # Dataset (not included, download from APTOS 2019)
├── figures/                # Training curves and visualizations
│   ├── distribution.png
│   ├── efficientnet.png
│   ├── vgg.png
│   ├── 1mobilenet.png
│   ├── mobilenet0.1.png
│   ├── 0.1bar.png
│   ├── mobilenet_0.01.png
│   ├── 0.01bar.png
├── report/                 # LaTeX report source
│   ├── report.tex
│   ├── report.pdf
├── README.md               # This file
```

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- NumPy
- OpenCV
- scikit-learn
- matplotlib
- pandas

Install dependencies:
```bash
pip install torch torchvision numpy opencv-python scikit-learn matplotlib pandas
```

## Usage
1. **Download Dataset**: Obtain the APTOS 2019 dataset from [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection).
2. **Preprocessing**: Run preprocessing scripts (not provided) to apply border removal, CLAHE, and normalization.
3. **Training**:
   - Configure model scripts for VGG-16, MobileNetV2, or EfficientNet-B0.
   - Run training with specified hyperparameters (see report for details).
4. **Knowledge Distillation**:
   - Train EfficientNet-B0 as the teacher.
   - Use pretrained teacher to distill knowledge to MobileNetV2 (adjust α and T as needed).
5. **Evaluation**: Compute accuracy and Quadratic Weighted Kappa on validation set.

## Results
- **VGG-16**: High accuracy (84.84%) but computationally expensive (138M parameters).
- **EfficientNet-B0**: Balanced performance (84.43%, Kappa: 0.9104) with fewer parameters (~4M).
- **MobileNetV2 (Distilled)**: Efficient (2.2M parameters) with strong performance (83.74%, Kappa: 0.9049 at α=0.1).
- **Key Insight**: Knowledge distillation enables MobileNetV2 to approach teacher performance while being significantly lighter.

## Figures
- Class distribution: `figures/distribution.png`
- Training curves: `figures/efficientnet.png`, `figures/vgg.png`, `figures/1mobilenet.png`, `figures/mobilenet0.1.png`, `figures/mobilenet_0.01.png`
- Model comparisons: `figures/0.1bar.png`, `figures/0.01bar.png`

## Report
The full report (`report/report.pdf`) provides detailed methodology, model architectures, training configurations, and results. LaTeX source is available in `report/report.tex`.

## Future Work
- Explore additional lightweight architectures (e.g., EfficientNet-B1, ShuffleNet).
- Test distillation with varied temperatures and α values.
- Address class imbalance with advanced sampling or loss weighting.
- Deploy models for real-time DR screening on edge devices.

## License
This project is licensed under the MIT License.

## Contact
For questions, please contact [Your Name] at [Your Email].