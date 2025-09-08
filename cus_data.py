"""
Custom Dataset Implementation for MNIST Digit Recognition

This module provides a custom PyTorch dataset class that handles
MNIST data loading with support for data augmentation and transforms.
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Callable


class CustomDataset(Dataset):
    """
    Custom dataset class for MNIST digit recognition.
    
    This class wraps data subsets and applies transforms for training and validation.
    It supports both labeled data (for training/validation) and unlabeled data (for testing).
    
    Args:
        data: Dataset containing tuples of (image, label) or just images for test data
        transforms: Optional transform to be applied to the data
    """
    
    def __init__(self, data, transforms: Optional[Callable] = None):
        """
        Initialize the custom dataset.
        
        Args:
            data: Input data - can be:
                - List of tuples (image_tensor, label_tensor) for training/validation
                - Tensor of images for test data
            transforms: Optional transform function to apply to images
        """
        self.data = data
        self.transforms = transforms
        
        # Check if data contains labels (training/validation) or is just images (test)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], tuple):
            self.has_labels = True
        else:
            self.has_labels = False
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            For training/validation data: (transformed_image, label)
            For test data: (transformed_image,)
        """
        if self.has_labels:
            # Training/validation data: (image, label) tuple
            image, label = self.data[idx]
        else:
            # Test data: just image tensor
            image = self.data[idx]
            label = None
        
        # Apply transforms if provided
        if self.transforms is not None:
            image = self.transforms(image)
        
        # Return appropriate format based on data type
        if self.has_labels:
            return image, label
        else:
            return image,
    
    def get_sample_info(self, idx: int) -> dict:
        """
        Get information about a specific sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing sample information
        """
        info = {
            'index': idx,
            'has_labels': self.has_labels,
            'transforms_applied': self.transforms is not None
        }
        
        if self.has_labels:
            _, label = self.data[idx]
            info['label'] = label.item() if isinstance(label, torch.Tensor) else label
        
        return info
    
    def get_data_shape(self) -> tuple:
        """
        Get the shape of the data in the dataset.
        
        Returns:
            Tuple containing the shape information
        """
        if len(self.data) == 0:
            return (0,)
        
        if self.has_labels:
            image, _ = self.data[0]
            return image.shape
        else:
            return self.data[0].shape


    def create_data_info(dataset: CustomDataset) -> dict:
        """
        Create a summary of dataset information.
        
        Args:
            dataset: CustomDataset instance
            
        Returns:
            Dictionary containing dataset summary
        """
        info = {
            'total_samples': len(dataset),
            'has_labels': dataset.has_labels,
            'data_shape': dataset.get_data_shape(),
            'transforms_enabled': dataset.transforms is not None
        }
        
        if dataset.has_labels and len(dataset) > 0:
            # Get label distribution for labeled data
            labels = []
            for i in range(min(1000, len(dataset))):  # Sample first 1000 for efficiency
                _, label = dataset.data[i]
                labels.append(label.item() if isinstance(label, torch.Tensor) else label)
            
            info['label_distribution'] = {
                'unique_labels': len(set(labels)),
                'sample_labels': labels[:10]  # First 10 labels as example
            }
        
        return info


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the CustomDataset
    print("CustomDataset Implementation for MNIST")
    print("=" * 40)
    
    # Create dummy data for demonstration
    dummy_images = torch.randn(100, 1, 28, 28)
    dummy_labels = torch.randint(0, 10, (100,))
    dummy_data = [(dummy_images[i], dummy_labels[i]) for i in range(100)]
    
    # Create dataset without transforms
    dataset = CustomDataset(dummy_data)
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Has labels: {dataset.has_labels}")
    print(f"Data shape: {dataset.get_data_shape()}")
    
    # Test getting a sample
    sample_image, sample_label = dataset[0]
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample label: {sample_label}")
    
    # Get dataset info
    info = create_data_info(dataset)
    print(f"Dataset info: {info}")
