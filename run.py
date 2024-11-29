import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam

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
    x = Dropout(0.4)(x)  # Dropout layer

    # Second Convolutional Block
    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)  # Dropout layer

    # Third Convolutional Block
    x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)  # Dropout layer

    # Fully Connected Layers
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)  # Larger Dense layer
    x = Dropout(0.7)(x)  # Stronger regularization here

    # Output Layer
    outputs = Dense(1, activation='linear')(x)  # Regression output

    # Compile Model
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model


# Build the model
input_shape = (128, 128, 3)  # Three channels (amp, phase, mask)
model = build_model(input_shape)
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=16,
    verbose=1
)

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
submission.to_csv("submission.csv", index=False)
