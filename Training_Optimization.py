import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical, set_random_seed
from keras.preprocessing import image
from keras.layers import Resizing, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam


set_random_seed(0)

# Label and Image paths
csv_file_path = r"C:\Users\utkar\OneDrive\Desktop\dataset and csv for postive and negative labels\CLIP\csv for bates_CLIP.csv"
images_dir = r"C:\Users\utkar\Downloads\Utkarsh Gupta\NUIG\Semester 2\DA Project\Extracted Dataset updated"

# Loading CSV having filenames and labels
df = pd.read_csv(csv_file_path)
print(df.head())

# Converting string labels to numerical labels
df['label'] = pd.Categorical(df['label']).codes
num_classes = len(df['label'].unique())

# Resizing and defining batch size
target_size = (96, 96)
batch_size = 32

# Loading templates and preprocessing them
def load_and_preprocess_images(df, images_dir, target_size):
    images = []
    labels = []
    for i, row in df.iterrows():
        img_path = os.path.join(images_dir, row['filename'])
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array /= 255.0
        images.append(img_array)
        labels.append(row['label'])
    return np.array(images), np.array(labels)

x_train, y_train = load_and_preprocess_images(df, images_dir, target_size)
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Converting labels to categorical labels (one-hot encoding)
y_train = to_categorical(y_train, num_classes=num_classes)

# Resampling in case of large dataset
def subsample(X, y=None, p=0.1):
    n_samples = X.shape[0]
    subsample_size = int(n_samples * p)
    idxs = np.random.choice(n_samples, subsample_size, replace=False)
    if y is None:
        return X[idxs]
    else:
        return X[idxs], y[idxs]

x_train, y_train = subsample(x_train, y_train, 0.2)

# Creating CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Model parameters
input_shape = (target_size[0], target_size[1], 3)

# Model compilation
model = create_cnn_model(input_shape, num_classes)

#  Model Training
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=batch_size,
    validation_split=0.2
)

# Saving the model for future use
model.save('cnn_model_batesdata.keras')

# Plotting training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
