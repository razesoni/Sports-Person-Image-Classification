# Sports Person Image Classification

A machine learning project for classifying and identifying sports personalities from images using computer vision and machine learning techniques.

## Project Overview

This project implements a sports person classification system that detects and classifies individuals in images using face detection and machine learning algorithms. The system processes sports-related images to identify specific athletes or categorize them by their sport.

## Methodology

### Image Processing Pipeline
- **Face Detection**: Uses OpenCV Haar Cascade Classifiers for frontal face detection
  - Haar Cascade: `haarcascade_frontalface_default.xml`
  - Eye Detection: `haarcascade_eye.xml`
- **Image Preprocessing**: Grayscale conversion for feature extraction
- **Feature Extraction**: Raw pixel features and Wavelet-based features using PyWavelets

### Machine Learning Models Evaluated
The project compares multiple classification algorithms with grid search optimization:
- **Support Vector Machines (SVM)** - For non-linear classification
- **Random Forest Classifier** - Ensemble learning approach
- **Logistic Regression** - Baseline linear classifier

All models undergo training-set grid search for hyperparameter optimization to identify the best performing configuration.

## Technical Stack

### Languages & Frameworks
- **Python**: Core programming language
- **Jupyter Notebook**: Interactive development and experimentation

### Dependencies
```
numpy          - Numerical computing
pandas         - Data manipulation and analysis
matplotlib     - Data visualization
seaborn        - Statistical data visualization
scikit-learn   - Machine learning algorithms and grid search
opencv-python  - Computer vision and face detection
PyWavelets     - Wavelet feature extraction
joblib         - Model serialization and persistence
```

## Project Structure

```
Sports-Person-Image-Classification/
├── sports_person_classifier_model.ipynb  # Main notebook with model implementation
├── model/
│   ├── saved_model.pkl                  # Trained classifier model
│   └── class_dictionary.json            # Class label mappings
├── dataset/                             # Training images (not included)
├── requirements.txt                     # Project dependencies
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

## Key Features

✓ Face detection and localization using Haar Cascades  
✓ Multiple feature extraction methods (raw pixels and wavelets)  
✓ Comparison of SVM, Random Forest, and Logistic Regression  
✓ Hyperparameter tuning via grid search  
✓ Model performance evaluation and comparison  
✓ Trained model persistence for inference  

## Installation

1. Clone the repository:
```bash
git clone https://github.com/razesoni/Sports-Person-Image-Classification.git
cd Sports-Person-Image-Classification
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebook:
```bash
python -m jupyterlab sports_person_classifier_model.ipynb
```

### Required Assets

The training images are **not included** in the repository. To run the notebook, you need:
- `sharapova1.jpg` - Sample image for initial demonstration
- `dataset/<person-name>/` - Directories containing training images for each person
- `dataset/cropped/` - Directory for face crop outputs
- OpenCV Haar Cascade XML files referenced in the notebook

You can obtain OpenCV cascade files through `cv2.data.haarcascades` or download them from the [OpenCV repository](https://github.com/opencv/opencv/tree/master/data/haarcascades).

## Model Performance

The notebook evaluates and compares model performance across:
- **Accuracy** - Overall classification accuracy
- **Precision** - Correct positive predictions among all positive predictions
- **Recall** - Correct positive predictions among all actual positives
- **F1-Score** - Harmonic mean of precision and recall

Performance metrics are computed during the model evaluation phase in the notebook and compared across all three algorithms.

## Limitations

- **Closed-set classification**: Designed for known identities; performance on unseen identities is not established
- **Data contamination**: Random train-test splits may inflate scores due to near-duplicate images
- **Uncontrolled conditions**: Not tested on diverse image conditions or angles
- **Face detection dependency**: Performance limited by Haar Cascade face detection accuracy

For robust evaluation, use stratified cross-validation and test on independent image sets.

## Future Enhancements

- Integration of deep learning models (CNN-based architectures like ResNet, VGG)
- Expanded dataset with more sports personalities and diverse conditions
- Real-time video classification capabilities
- Web-based inference interface
- Transfer learning for improved feature extraction
- Multi-face detection and classification in group images

## Author

[razesoni](https://github.com/razesoni)

## License

This project is open source and available under the repository's license.

---

**Note**: This is an active machine learning project focused on computer vision applications in sports analytics. The trained models are persisted in the `/model` directory for reuse.
