# Digit Recognition using ResNet18

A deep learning project that implements digit recognition using a ResNet18 architecture trained on the MNIST dataset. This project demonstrates modern CNN techniques for handwritten digit classification with PyTorch and Apple Silicon GPU acceleration.

## 🎯 Project Overview

This project solves the classic MNIST digit recognition problem using a state-of-the-art ResNet18 architecture. The model is trained to classify handwritten digits (0-9) from 28x28 grayscale images with high accuracy.

### Key Features
- **ResNet18 Architecture**: Adapted for single-channel MNIST input
- **Apple Silicon Optimization**: Uses MPS (Metal Performance Shaders) for GPU acceleration
- **Data Augmentation**: Random rotation for improved generalization
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Scheduling**: StepLR scheduler for optimal convergence
- **Comprehensive Evaluation**: Training/validation monitoring with visualizations

## 📊 Dataset

- **Training Set**: 42,000 labeled digit images (90% train, 10% validation)
- **Test Set**: 28,000 unlabeled images for prediction
- **Image Format**: 28x28 grayscale pixels
- **Classes**: 10 digits (0-9)

## 🏗️ Architecture

The model uses a ResNet18 backbone with the following modifications:
- **Input Layer**: Modified to accept single-channel (grayscale) input
- **Output Layer**: 10-class classification head for digits 0-9
- **Residual Connections**: Skip connections for better gradient flow
- **Batch Normalization**: Stabilizes training and improves convergence

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch with MPS support (for Apple Silicon)
- Required packages: `torch`, `torchvision`, `pandas`, `matplotlib`, `seaborn`, `tqdm`

### Installation
```bash
pip install torch torchvision pandas matplotlib seaborn tqdm
```

### Usage
1. Place your MNIST data files in the appropriate directory:
   - `./Downloads/digit-recognizer/train.csv`
   - `./Downloads/digit-recognizer/test.csv`

2. Run the Jupyter notebook:
   ```bash
   jupyter notebook DigitRecognition.ipynb
   ```

3. Execute cells sequentially to:
   - Load and preprocess data
   - Train the ResNet18 model
   - Generate predictions on test set
   - Save results and model weights

## 📈 Training Process

The training pipeline includes:

1. **Data Preprocessing**: Normalization and augmentation
2. **Model Initialization**: ResNet18 with MNIST-specific modifications
3. **Training Loop**: Adam optimizer with StepLR scheduling
4. **Validation**: Real-time monitoring of loss and accuracy
5. **Early Stopping**: Prevents overfitting with patience mechanism
6. **Model Checkpointing**: Saves best performing weights

### Training Configuration
- **Optimizer**: Adam (lr=1e-2)
- **Scheduler**: StepLR (step_size=5, gamma=0.05)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32 (training), 64 (inference)
- **Max Epochs**: 15 with early stopping
- **Patience**: 10 epochs

## 📁 Project Structure

```
DigitRecongitionResNet/
├── DigitRecognition.ipynb    # Main training and inference notebook
├── README.md                 # This file
├── cus_data.py              # Custom dataset implementation
└── DigitRecognitionResNet18.pt  # Saved model weights
```

## 🔧 Custom Dataset

The project includes a custom dataset class (`cus_data.py`) that:
- Handles data loading and preprocessing
- Applies transforms for training and validation
- Supports data augmentation techniques

## 📊 Results

The model achieves competitive performance on the MNIST dataset with:
- Efficient training on Apple Silicon GPUs
- Robust generalization through data augmentation
- Stable convergence with early stopping
- High accuracy on digit classification

## 🎯 Output Files

After training and inference, the following files are generated:
- `solution.csv`: Predictions for test set (competition format)
- `DigitRecognitionResNet18.pt`: Trained model weights

## 🔍 Key Implementation Details

### Data Augmentation
- Random rotation (±10 degrees) for training data
- Normalization using dataset statistics (mean=0.1310, std=0.3085)

### Model Architecture Adaptations
- Modified first convolutional layer for single-channel input
- Adjusted final fully connected layer for 10-class output
- Preserved ResNet18's residual structure for optimal performance

### Training Optimizations
- GPU acceleration with MPS on Apple Silicon
- Multi-worker data loading for efficiency
- Memory pinning for faster data transfer
- Gradient accumulation and proper weight initialization

## 🤝 Contributing

Feel free to contribute to this project by:
- Improving data augmentation strategies
- Experimenting with different architectures
- Optimizing training hyperparameters
- Adding evaluation metrics and visualizations

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- MNIST dataset creators
- PyTorch team for the excellent framework
- Apple for MPS GPU acceleration support
- ResNet paper authors for the architecture design
- GPT for helping with commenting code and adding markdowns. 
