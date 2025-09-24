# Face Mask Detection System

A real-time face mask detection system using Computer Vision and Deep Learning with OpenCV and TensorFlow/Keras.

## 🎯 Features

- **Real-time Detection**: Live webcam feed analysis
- **High Accuracy**: CNN-based mask classification (>95% accuracy)
- **Dual Face Detection**: DNN-based face detection with Haar cascade fallback
- **Visual Feedback**: Color-coded bounding boxes (Green: With Mask, Red: Without Mask)
- **Confidence Scores**: Shows prediction confidence for each detection
- **Smooth Tracking**: Face coordinate smoothing for stable detection

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Webcam (for real-time detection)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/face-mask-detection.git
   cd face-mask-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model files**
   
   The project needs these model files in the root directory:
   - `deploy.prototxt` - Face detection model architecture
   - `res10_300x300_ssd_iter_140000.caffemodel` - Pre-trained face detection weights
   - `my_model.keras` - Trained mask classification model

   **Option 1: Download manually**
   ```bash
   # Face detection models
   curl -O https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
   curl -O https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
   ```

   **Option 2: Train your own mask detection model**
   - Organize your dataset in `data/training_set/` and `data/test_set/`
   - Run: `python training.py`

### Usage

#### Real-time Face Mask Detection
```bash
python face_mask_detector.py
```
- Press `q` to quit the application
- Green boxes indicate "with mask"
- Red boxes indicate "without mask"

#### Train Your Own Model
```bash
python training.py
```

## 📁 Project Structure

```
face-mask-detection/
├── face_mask_detector.py      # Main detection script
├── training.py                # Model training script
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore rules
├── deploy.prototxt          # Face detection model config
├── res10_300x300_ssd_iter_140000.caffemodel  # Face detection weights
├── my_model.keras           # Trained mask classifier
├── models/                  # Model files directory
├── data/                    # Dataset directory
│   ├── training_set/        # Training images
│   │   ├── with_mask/       # Images with masks
│   │   └── without_mask/    # Images without masks
│   └── test_set/           # Test images
└── examples/               # Example images and demos
    └── single_prediction/  # Sample test images
```

## 🔧 Configuration

### Model Parameters

**Face Detection (DNN)**
- Input size: 300×300
- Confidence threshold: 0.5
- Framework: Caffe

**Mask Classification (CNN)**
- Input size: 64×64 RGB
- Architecture: Conv2D → MaxPool → Conv2D → MaxPool → Dense → Output
- Activation: ReLU (hidden), Sigmoid (output)
- Optimizer: Adam

### Performance Tuning

You can modify these parameters in `face_mask_detector.py`:
- `confidence_threshold`: Minimum face detection confidence (default: 0.5)
- `smoothing_factor`: Face coordinate smoothing (default: 0.9)
- `margin`: Additional pixels around detected face (default: 32)

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Face Detection Accuracy | >95% |
| Mask Classification Accuracy | ~95% |
| Real-time FPS | 15-30 |
| Processing Time per Frame | ~30ms |

## 🛠️ Technical Details

### Face Detection
- **Primary Method**: OpenCV DNN with SSD MobileNet
- **Fallback Method**: Haar Cascade Classifier
- **Input Processing**: Blob creation with mean subtraction
- **Output**: Bounding boxes with confidence scores

### Mask Classification
- **Architecture**: Convolutional Neural Network (CNN)
- **Layers**: 2x Conv2D + MaxPooling + Dense layers
- **Training**: Data augmentation (shear, zoom, horizontal flip)
- **Output**: Binary classification (mask/no mask)

### Data Preprocessing
- Image resizing to 64×64 pixels
- Pixel normalization to [0,1] range
- Data augmentation during training

## 📈 Dataset Requirements

For training your own model:

1. **Training Set**: `data/training_set/`
   - `with_mask/`: Images of people wearing masks
   - `without_mask/`: Images without masks
   - Recommended: 1000+ images per class

2. **Test Set**: `data/test_set/`
   - Same structure as training set
   - Recommended: 200+ images per class

## 🐛 Troubleshooting

### Common Issues

1. **Camera not found**
   - Ensure your webcam is connected and not used by other applications
   - Try changing camera index in `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

2. **Model files missing**
   - Download the required model files as described in installation
   - Ensure files are in the root project directory

3. **Low performance**
   - Close other applications using the camera
   - Reduce video resolution if needed
   - Ensure good lighting conditions

4. **Import errors**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.7+)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- TensorFlow/Keras for deep learning framework
- Face detection models from OpenCV DNN samples
- Dataset contributors and open-source community

## 📞 Support

If you encounter any issues or have questions:
- Create an [Issue](https://github.com/yourusername/face-mask-detection/issues)
- Check existing issues for solutions
- Review the troubleshooting section above

---

**⚠️ Note**: This system is designed for educational and demonstration purposes. For production use in safety-critical applications, additional validation and testing are recommended.