# Data Directory

This directory contains the datasets used for training and testing the face mask detection models.

## Directory Structure

```
data/
├── raw/                    # Original, immutable data
│   ├── training_set/       # Training images
│   │   ├── with_mask/      # Images of people wearing masks
│   │   └── without_mask/   # Images of people not wearing masks
│   ├── test_set/           # Test/validation images
│   │   ├── with_mask/      
│   │   └── without_mask/   
│   └── single_prediction/  # Individual test images
└── processed/              # Cleaned and processed data
    ├── training_set/
    ├── test_set/
    └── annotations/        # Labels, metadata, etc.
```

## Dataset Requirements

For training the mask classification model, organize your images as follows:

### Training Data
- **Location**: `data/raw/training_set/`
- **Structure**: 
  - `with_mask/`: Contains images of people wearing face masks
  - `without_mask/`: Contains images of people not wearing face masks
- **Format**: JPG, PNG, JPEG
- **Recommended size**: At least 1000 images per class
- **Image size**: Any size (will be resized to 64x64 during preprocessing)

### Test Data
- **Location**: `data/raw/test_set/`
- **Structure**: Same as training data
- **Purpose**: Model evaluation and validation
- **Recommended size**: 20-30% of total dataset

## Obtaining Datasets

### Public Datasets
1. **Face Mask Detection Dataset** (Kaggle)
   - URL: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
   - Size: ~12,000 images
   - License: Open source

2. **Medical Mask Dataset**
   - URL: https://github.com/prajnasb/observations/tree/master/experiements/data
   - Size: ~1,376 images
   - License: MIT

### Creating Your Own Dataset
1. Use `examples/data_collection.py` to capture images from webcam
2. Manually sort images into appropriate directories
3. Ensure balanced classes (similar number of mask/no-mask images)

## Data Preprocessing

The training script automatically applies the following preprocessing:
- Resize to 64x64 pixels
- Normalize pixel values to [0,1] range
- Data augmentation (rotation, zoom, horizontal flip)

## Usage

```bash
# Download sample dataset
python scripts/download_sample_data.py

# Train model with your data
python -m face_mask_detection.cli.train --data-dir data/raw

# Evaluate model
python -m face_mask_detection.cli.evaluate --test-dir data/raw/test_set
```

## Data Privacy

- Ensure you have proper permissions for all images in your dataset
- Consider privacy implications when collecting facial images
- Follow applicable data protection regulations (GDPR, etc.)
- Do not commit personal/private images to version control

**Note**: Data files are excluded from git due to size and privacy concerns. Download or create datasets locally.