# Models Directory

This directory contains the pre-trained model files used for face mask detection.

## Required Files

### Face Detection Models (OpenCV DNN)
- **File**: `deploy.prototxt`
- **Description**: Network architecture definition for face detection
- **Download**: Available in OpenCV repository or face-mask-detection releases

- **File**: `res10_300x300_ssd_iter_140000.caffemodel`
- **Description**: Pre-trained weights for SSD face detection model
- **Download**: Available from OpenCV repository or face-mask-detection releases

### Mask Classification Model
- **File**: `face_mask_classifier.keras` (or `my_model.keras`)
- **Description**: Trained CNN model for mask/no-mask classification
- **Input Size**: 64x64 RGB images
- **Architecture**: CNN with Conv2D, MaxPooling, Dense layers
- **Training**: Custom trained on mask/no-mask dataset

## Automatic Download

When you first run the application, it will automatically attempt to download missing model files to this directory.

You can also manually download the models:

```bash
# Download face detection models
wget -O models/deploy.prototxt https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget -O models/res10_300x300_ssd_iter_140000.caffemodel https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

# The mask classification model should be trained using src/face_mask_detection/cli/train.py
```

## Training Your Own Model

To train a custom mask classification model:

1. Organize your dataset in the `data/raw/` directory
2. Run the training script:
   ```bash
   python -m face_mask_detection.cli.train --config config/default.yaml
   ```
3. The trained model will be saved to this directory

## Model Information

| Model | Size | Accuracy | Speed |
|-------|------|----------|-------|
| Face Detection (DNN) | ~10MB | >95% | ~30ms |
| Mask Classifier | ~5MB | ~95% | ~5ms |

**Note**: Model files are excluded from git due to their size. Please download them separately or train your own models.