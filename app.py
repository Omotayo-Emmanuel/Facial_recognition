# app.py


import streamlit as st  # Importing Streamlit as the UI framework and aliasing it as st
import tensorflow as tf  # Importing TensorFlow for model building and training
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Importing ImageDataGenerator for on-the-fly data augmentation
from tensorflow.keras.applications import Xception  # Importing Xception architecture for transfer learning (unused import kept as original)
from tensorflow.keras.applications.xception import preprocess_input  # Importing preprocessing function matching Xception inputs
from tensorflow.keras.layers import Flatten, Dense, Dropout  # Importing common Keras layers used in the model head
from tensorflow.keras.models import Model, load_model  # Importing Model for functional API and load_model to reload saved models
from tensorflow.keras.optimizers import Adam  # Importing Adam optimizer to compile the model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Importing training callbacks for early stop and LR scheduling
import math  # Importing math for mathematical operations like ceiling
import numpy as np  # Importing NumPy for numerical array operations
import os  # Importing os for filesystem checks and file operations
import cv2  # Importing OpenCV for image I/O and preprocessing
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting training history
from sklearn.utils.class_weight import compute_class_weight  # Importing utility to compute class weights for imbalanced data
import warnings  # Importing warnings to control warning output
warnings.filterwarnings('ignore')  # Suppressing non-critical warnings to keep logs cleaner

# App Settings
st.set_page_config(page_title="Custom Facial Recognition", layout="wide")  # Configuring Streamlit page title and layout
st.title("Custom Facial Recognition System")  # Displaying main title in the Streamlit app
st.markdown("Train a model to recognize members of your group using transfer learning.")  # Displaying a brief description under the title

# Add confidence threshold to session state if not exists
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.7  # Setting a default confidence threshold in session state when not present

# Helper Functions
@st.cache_resource
def get_model(num_classes):
    """
    Loads a pre-trained MobileNetV2 model and adds new, trainable layers for classification.
    Uses dropout for regularization to prevent overfitting.
    """
    st.write("Loading and configuring the base model...")  # Notifying user that model construction is starting
    # Use average pooling instead of flattening for better feature extraction
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(229, 229, 3)
    )  # Loading Xception base model with pretrained ImageNet weights and excluding the top classification layers

    
    # Freeze the base model initially
    base_model.trainable = False  # Freezing all layers of the base model to train only the new head first

    # Add new, trainable layers on top with regularization
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)  # Applying global average pooling to reduce spatial dims
    x = Dropout(0.5)(x)  # Applying dropout to reduce overfitting in the new dense block
    x = Dense(128, activation='relu')(x)  # Adding a dense layer with ReLU activation as the feature head
    x = Dropout(0.3)(x)  # Adding another dropout layer for additional regularization
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # Adding final softmax layer matching number of classes

    # Create the custom model
    model = Model(inputs=base_model.input, outputs=predictions)  # Creating the Keras Model combining base and new head
    return model, base_model  # Returning both the full model and the base model reference for later fine-tuning

@st.cache_data
def train_model(epochs=15):
    """
    Prepares the dataset and fine-tunes the model.
    Returns the trained model and class labels.
    Implements class weighting to handle imbalanced datasets.
    """
    # Create data generators with consistent preprocessing
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # Using Xception preprocessing function for consistent input scaling
        rotation_range=20,  # Enabling random rotations up to 20 degrees for augmentation
        width_shift_range=0.2,  # Enabling horizontal shifts as augmentation
        height_shift_range=0.2,  # Enabling vertical shifts as augmentation
        shear_range=0.2,  # Enabling shear transformations as augmentation
        zoom_range=0.2,  # Enabling zoom augmentation
        horizontal_flip=True,  # Allowing horizontal flips as augmentation
        fill_mode='nearest',  # Setting fill mode for augmented pixels
        validation_split=0.2  # Reserving 20% of data for validation via subset selection
    )
    
    try:
        train_generator = train_datagen.flow_from_directory(
            'dataset',
            target_size=(229, 229),
            batch_size=32,
            class_mode='categorical',
            subset='training'  # Selecting the training subset from the directory flow
        )
        
        validation_generator = train_datagen.flow_from_directory(
            'dataset',
            target_size=(229, 229),
            batch_size=32,
            class_mode='categorical',
            subset='validation'  # Selecting the validation subset from the directory flow
        )
    except Exception as e:
        st.error(f"Error loading dataset: {e}")  # Reporting dataset loading errors to the user
        st.warning("Please make sure you have a 'dataset' folder with subfolders for each person.")  # Guiding user to correct dataset structure
        return None, None, None  # Returning None tuple to indicate failure in loading data

    num_classes = len(train_generator.class_indices)  # Computing number of classes based on discovered subfolders
    class_labels = list(train_generator.class_indices.keys())  # Capturing the class label names in a list
    st.session_state.class_labels = class_labels  # Storing class labels in session state for inference later
    
    # Compute class weights to handle imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )  # Calculating balanced class weights to mitigate class imbalance during training
    class_weights_dict = dict(enumerate(class_weights))  # Converting class weights array into a dictionary keyed by class index
    
    model, base_model = get_model(num_classes)  # Building the model and obtaining the base model reference

    
    # Initial training of new layers
    st.write("Starting initial training of new layers...")  # Informing user that initial head training is starting
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])  # Compiling model with Adam optimizer and categorical crossentropy suitable for multi-class problems
    
    # Callbacks for better training
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Setting up early stopping to prevent overfitting
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)  # Setting up LR reduction on plateau for stable fine-tuning
    
    # Train the model with progress bar
    with st.spinner('Initial training in progress...'):
        steps_per_epoch = math.ceil(train_generator.samples / train_generator.batch_size)
        validation_steps = math.ceil(validation_generator.samples / validation_generator.batch_size)
        history = model.fit(
            train_generator,
            steps_per_epoch= steps_per_epoch,
            validation_data= validation_generator,
            validation_steps= validation_steps,
            epochs=epochs // 2,
            class_weight=class_weights_dict,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )  # Fitting the model on training data with validation, class weights, and callbacks

    # Fine-tuning: Unfreeze some layers of the base model
    st.write("Starting fine-tuning of base model...")  # Notifying user that fine-tuning is starting
    # base_model = model.layers[0]
    
    # Unfreeze the top layers of the base model (more strategic than unfreezing all)
    base_model.trainable = True  # Setting base_model.trainable True to allow selective unfreezing of top layers
    for layer in base_model.layers[:100]:
        layer.trainable = False  # Freezing the bottom-most layers while keeping top layers trainable for fine-tuning
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])  # Recompiling with a much smaller learning rate for stable fine-tuning
    
    # Continue training with fine-tuning
    with st.spinner('Fine-tuning in progress...'):
        history_fine = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            epochs=epochs // 2,
            class_weight=class_weights_dict,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )  # Continuing training while updating unfrozen layers and monitoring validation performance
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # Creating matplotlib figure with two subplots for accuracy and loss
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'] + history_fine.history['accuracy'])  # Plotting combined training accuracy across phases
    ax1.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'])  # Plotting combined validation accuracy across phases
    ax1.set_title('Model Accuracy')  # Setting title for accuracy subplot
    ax1.set_ylabel('Accuracy')  # Labeling y-axis for accuracy plot
    ax1.set_xlabel('Epoch')  # Labeling x-axis for epoch count
    ax1.legend(['Train', 'Validation'], loc='upper left')  # Adding legend to the accuracy plot
    
    # Plot loss
    ax2.plot(history.history['loss'] + history_fine.history['loss'])  # Plotting combined training loss across phases
    ax2.plot(history.history['val_loss'] + history_fine.history['val_loss'])  # Plotting combined validation loss across phases
    ax2.set_title('Model Loss')  # Setting title for loss subplot
    ax2.set_ylabel('Loss')  # Labeling y-axis for loss plot
    ax2.set_xlabel('Epoch')  # Labeling x-axis for epoch count
    ax2.legend(['Train', 'Validation'], loc='upper left')  # Adding legend to the loss plot
    
    st.pyplot(fig)  # Rendering the training plots in the Streamlit app
    
    model.save('face_model.h5')  # Saving the trained model to disk for later inference
    st.success("Training and fine-tuning complete! Model saved.")  # Informing user of successful training and save
    return model, class_labels, history  # Returning the trained model, class labels, and training history

def load_saved_model():
    """Load a previously saved model if available"""
    if os.path.exists('face_model.h5'):
        try:
            model = load_model('face_model.h5')  # Loading saved Keras model from file
            # Try to load class labels from session state or create placeholder
            class_labels = getattr(st.session_state, 'class_labels', [f"Class_{i}" for i in range(model.output_shape[1])])  # Retrieving stored labels or creating generic placeholders
            return model, class_labels  # Returning loaded model and labels
        except:
            return None, None  # Returning None tuple if loading fails
    return None, None  # Returning None tuple if model file does not exist

def predict_image(model, image, class_labels, threshold = 0.7):
    """
    Processes the uploaded image and makes a prediction.
    Applies the same preprocessing as during training.
    """
    # Convert BGR to RGB (OpenCV loads as BGR, but model expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converting OpenCV BGR image to RGB color order
    
    # Resize and preprocess the image exactly like training data
    img = cv2.resize(image_rgb, (229, 229))  # Resizing image to model input dimensions
    img = np.expand_dims(img, axis=0)  # Adding batch dimension to image array
    img = preprocess_input(img)  # Applying MobileNetV2 preprocessing to match training pipeline

    # Predict the class
    prediction = model.predict(img, verbose=0)  # Running model inference to obtain class probabilities
    predicted_class_index = np.argmax(prediction)  # Selecting the index with highest predicted probability
    confidence = prediction[0][predicted_class_index]  # Extracting confidence score for the top prediction

     # Apply confidence threshold
    if confidence < threshold:
        return "Unknown", confidence, prediction[0]  # Returning Unknown when confidence falls below threshold
    
    # Get the name and format the output
    predicted_name = class_labels[predicted_class_index]  # Mapping predicted index back to human-readable class name

    return predicted_name, confidence, prediction[0]  # Returning predicted label, confidence, and full probability vector

# --- Main GUI ---
st.header("1. Train Your Custom Model")  # Displaying header for the training section
st.info("Ensure you have a 'dataset' folder with subfolders for each person before training.")  # Showing an info message about dataset requirements

# Try to load existing model first
if 'model' not in st.session_state:
    model, class_labels = load_saved_model()  # Attempting to load a previously saved model into session
    if model is not None:
        st.session_state.model = model  # Storing loaded model in session state for reuse
        st.session_state.class_labels = class_labels  # Storing loaded class labels in session state
        st.success("Loaded previously trained model!")  # Notifying user that a saved model was loaded

col1, col2 = st.columns(2)  # Creating two columns in the Streamlit layout for training and model management controls

with col1:
    if st.button("Start Training"):
        if os.path.isdir('dataset'):
            with st.spinner("Training in progress..."):
                trained_model, class_labels, history = train_model(epochs=20)  # Kicking off training process when button is pressed
            if trained_model and class_labels:
                st.session_state.model = trained_model  # Saving trained model to session state after training completes
                st.session_state.class_labels = class_labels  # Saving class labels to session state after training completes
                st.session_state.training_history = history  # Saving training history to session state for potential later use
        else:
            st.warning("The 'dataset' folder was not found.")  # Warning user when dataset folder is missing

with col2:
    if st.button("Clear Model"):
        if 'model' in st.session_state:
            del st.session_state.model  # Removing model from session state when clearing
        if os.path.exists('face_model.h5'):
            os.remove('face_model.h5')  # Deleting saved model file from disk if present
        st.success("Model cleared! You can train a new one.")  # Confirming model removal to the user

st.header("2. Test Your Model")  # Displaying header for the testing/inference section
if 'model' in st.session_state and 'class_labels' in st.session_state:
    # Added confidence threshold slider
    st.session_state.confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.confidence_threshold,
        help="Set the minimum confidence level required to recognize a face. Lower values may increase false positives."
    )  # Presenting a slider to adjust the confidence threshold used for classifying as known vs unknown
    st.info("Model is ready for inference. Upload an image to identify a person.")  # Informing user that model is available for inference
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])  # Providing a file uploader for image input

    if uploaded_file is not None:
        # Read the uploaded file and convert to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)  # Reading uploaded bytes and converting to NumPy array
        image = cv2.imdecode(file_bytes, 1)  # Decoding image bytes into an OpenCV BGR image
        st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)  # Displaying the uploaded image in the app

        with st.spinner("Analyzing..."):
            predicted_name, confidence, all_predictions = predict_image(
                st.session_state.model, 
                image, 
                st.session_state.class_labels,
                st.session_state.confidence_threshold
            )  # Running inference on the uploaded image using the stored model and threshold

        # Display result with appropriate color
        if predicted_name == "Unknown":
            st.warning(f"Prediction: {predicted_name}")  # Showing a warning when the model returns Unknown
        else:
            st.success(f"Prediction: {predicted_name}")  # Showing a success message when a person is recognized
            
        st.write(f"Confidence: {confidence:.2%}")  # Presenting the confidence as a percentage
        
        # Show confidence for all classes
        st.subheader("Confidence for all classes:")
        for i, (class_name, conf) in enumerate(zip(st.session_state.class_labels, all_predictions)):
              # Highlight the predicted class
            if class_name == predicted_name:
                st.markdown(f"**{class_name}: {conf:.2%}**")  # Emphasizing the predicted class in the list
            else:
                st.write(f"{class_name}: {conf:.2%}")  # Listing other classes with their confidences

else:
    st.warning("Please train the model first by clicking 'Start Training'.")  # Warning the user to train model before testing
    st.caption("You must have a 'dataset' folder with images of the people you want to recognize.")  # Providing a caption reminding dataset requirements
