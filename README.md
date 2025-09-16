# Facial Recognition System

A comprehensive facial recognition system that allows you to train a custom model to recognize members of your group using transfer learning with Xception architecture.

## Project Structure

```
facial-recognition-system/
├── app.py                 # Main Streamlit application
├── face_capture.py        # Script for capturing face images
├── face_model.h5          # Trained model (generated after training)
└── dataset/               # Directory containing training images
    ├── Person1/
    ├── Person2/
    ├── ...
    └── Unknown/
```

## Features

1. **Custom Model Training**: Train a facial recognition model using transfer learning with Xception architecture
2. **Real-time Inference**: Upload images and get predictions with confidence scores
3. **Unknown Detection**: Automatically detects and classifies unknown faces/objects
4. **Training Visualization**: View accuracy and loss graphs during training
5. **Adjustable Confidence Threshold**: Customize the sensitivity of recognition
6. **Face Capture Utility**: Script to easily capture training images

## Installation

1. Clone or download this project
2. Install required dependencies:

```bash
pip install streamlit tensorflow opencv-python scikit-learn matplotlib numpy
```

## Usage

### 1. Preparing Your Dataset

Create a folder structure for your training images:

```
dataset/
├── Person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Person2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Unknown/
    ├── object1.jpg
    ├── non-face1.jpg
    └── ...
```

### 2. Capturing Training Images

Use the face capture utility to easily create training images:

```bash
python face_capture.py
```

Before running, edit `face_capture.py` to set:
- `PERSON_NAME = "NameOfPerson"` for each team member
- `CAM_INDEX = 0` (usually 0 for default webcam)

Press SPACE to capture images, Q to quit.

### 3. Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

### 4. Training the Model

1. Click "Start Training" to begin training your custom model
2. The app will:
   - Load and preprocess your dataset
   - Build a model using Xception base with custom classification layers
   - Train the model with data augmentation
   - Fine-tune the model for better accuracy
   - Save the model to `face_model.h5`

### 5. Making Predictions

1. Adjust the confidence threshold using the slider
2. Upload an image to test the model
3. View the prediction results with confidence scores for all classes

## Technical Details

### Model Architecture

- **Base Model**: Xception (pretrained on ImageNet)
- **Input Size**: 229×229 pixels (Xception's optimal input size)
- **Custom Layers**: GlobalAveragePooling2D → Dropout(0.5) → Dense(128, ReLU) → Dropout(0.3) → Output Layer

### Training Process

1. **Initial Training**: Freeze base model, train only custom layers
2. **Fine-tuning**: Unfreeze top layers of base model, train with lower learning rate
3. **Data Augmentation**: Rotation, shifting, shearing, zooming, flipping
4. **Class Weighting**: Automatic handling of imbalanced datasets

### Confidence Threshold

The system uses an adjustable confidence threshold (default: 0.7) to:
- Recognize known faces when confidence is above threshold
- Classify as "Unknown" when confidence is below threshold

## Tips for Better Accuracy

1. **Capture diverse images**: Different angles, lighting conditions, expressions
2. **Include enough samples**: At least 30-50 images per person
3. **Balance your dataset**: Similar number of images for each person
4. **Quality over quantity**: Clear, well-lit images work best
5. **Include variety in Unknown class**: Various objects, non-face images, other people

## Troubleshooting

### Common Issues

1. **"dataset folder not found"**: Create a dataset folder with subfolders for each person
2. **Webcam not working**: Check CAM_INDEX in face_capture.py (try 0, 1, 2)
3. **Low accuracy**: 
   - Add more training images
   - Adjust confidence threshold
   - Ensure good image quality
4. **Memory errors**: Reduce batch size or image dimensions

### Performance Notes

- Xception provides better accuracy than MobileNetV2 but requires more resources
- Training may take several minutes depending on dataset size and hardware
- Inference is relatively fast once the model is loaded

## File Descriptions

- **app.py**: Main application with training and inference capabilities
- **face_capture.py**: Utility for capturing training images via webcam
- **face_model.h5**: Saved model file (created after training)

## Future Enhancements

Potential improvements for this system:
- Real-time webcam recognition
- Support for additional model architectures
- Export/import functionality for models
- Advanced data augmentation techniques
- Model performance metrics and analysis

## License

This project is for educational and research purposes. Please ensure you have proper permissions when capturing and using facial images.
