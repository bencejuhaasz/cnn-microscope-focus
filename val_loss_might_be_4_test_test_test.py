import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam


from tensorflow.keras.callbacks import TensorBoard

tensorboard_cb = TensorBoard(log_dir='./logs', histogram_freq=1)


# Paths
IMAGE_DIR = "data/train_data"  # Directory containing all the image files
TRAIN_CSV = "data/data_labels_train.csv"


def load_image(filepath):
    """Load an image and preprocess it."""
    image = Image.open(filepath).convert("L")  # Convert to grayscale
    image = image.resize((128, 128))  # Resize to 128x128 for consistency
    return np.array(image) / 255.0  # Normalize to [0, 1]


def prepare_data(df, image_dir):
    """Prepare data by matching filename_ids with their images."""
    images = []
    labels = []
    for _, row in df.iterrows():
        base_name = row['filename_id']
        label = row['defocus_label']

        # Load all three images
        amp_img = load_image(os.path.join(image_dir, base_name + "_amp.png"))
        phase_img = load_image(os.path.join(image_dir, base_name + "_phase.png"))
        mask_img = load_image(os.path.join(image_dir, base_name + "_mask.png"))

        # Stack them into a single input tensor
        stacked_img = np.stack([amp_img, phase_img, mask_img], axis=-1)

        images.append(stacked_img)
        labels.append(label)

    return np.array(images), np.array(labels)


# Load the training data
train_df = pd.read_csv(TRAIN_CSV)
X, y = prepare_data(train_df, IMAGE_DIR)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_model(input_shape):
    """Build an improved CNN model for regression."""
    inputs = Input(shape=input_shape)

    # First Convolutional Block
    x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)  # Dropout layer

    # Second Convolutional Block
    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)  # Dropout layer

    # Third Convolutional Block
    x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)  # Dropout layer

    # Fully Connected Layers
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)  # Larger Dense layer
    x = Dropout(0.5)(x)  # Stronger regularization here

    # Output Layer
    outputs = Dense(1, activation='linear')(x)  # Regression output

    # Compile Model
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.00085),
        loss='mse',
        metrics=['mae']
    )
    return model


# Build the model
input_shape = (128, 128, 3)  # Three channels (amp, phase, mask)
model = build_model(input_shape)
model.summary()

from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.8,  # Less aggressive reduction
    patience=30,  # Reduce LR after 10 epochs of stagnation
    min_lr=0.00001  # Minimum learning rate
)


# Train the model
try:
    with tf.device('/GPU:0'):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30000,
            batch_size=16,
            verbose=1,
            callbacks=[lr_scheduler,tensorboard_cb]
        )
finally:
    import time

    # Get a unique timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")


    model.save("cnn_"+timestamp+"_defocus_model.h5")
    # Predict on the test set (load test data similarly as train data)
    # Assuming test_df is available with filename_ids for the test set

    test_df = pd.read_csv("data/example_solutions.csv")  # Placeholder path for test data
    print(test_df.columns)
    print(test_df.head())
    IMAGE_DIR = "data/test_data"
    X_test, _ = prepare_data(test_df, IMAGE_DIR)

    # Predict distances
    predictions = model.predict(X_test)
    rounded_predictions = np.abs(np.round(predictions)).astype(int)

    # Create submission file
    submission = pd.DataFrame({
        "Id": test_df['filename_id'],
        "Expected": rounded_predictions.flatten()
    })
    submission.to_csv("submission"+timestamp+".csv", index=False)
