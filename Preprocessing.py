import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical, set_random_seed
from keras.preprocessing import image
from keras.layers import Resizing
import umap

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

# Plotting some images
plt.figure(figsize=(5, 5))
plt.imshow(x_train[0])
plt.title(f"Label: {np.argmax(y_train[0])}")
plt.show()

resizing_layer = Resizing(target_size[0], target_size[1])
x_train_resized = resizing_layer(x_train).numpy()

# Function to visualize embeddings
def visualise_embedding(X, y=None, p_subsample=None):

    n_samples, n_features = X.shape

    if y is not None:
        if n_samples != y.shape[0]:
            raise ValueError(f"X and y have incompatible shapes {X.shape} and {y.shape}")

        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

    if p_subsample is not None:
        if y is None:
            X = subsample(X, p_subsample)
        else:
            X, y = subsample(X, y, p_subsample)

    if n_features > 2:
        reducer = umap.UMAP()
        X = StandardScaler().fit_transform(X)
        X = reducer.fit_transform(X)

    colors = plt.get_cmap('tab10')

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        indices = np.where(y == i)
        plt.scatter(X[indices, 0], X[indices, 1], label=f'Class {i}', color=colors(i))

    axis0_lab = "UMAP0" if n_features > 2 else "z0"
    axis1_lab = "UMAP1" if n_features > 2 else "z1"

    plt.legend()
    plt.xlabel(axis0_lab)
    plt.ylabel(axis1_lab)
    plt.show()

# Plotting the embedding
visualise_embedding(x_train_resized.reshape(len(x_train_resized), -1), y_train)
