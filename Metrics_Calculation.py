import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical, set_random_seed
from keras.preprocessing import image
from keras.models import Sequential, load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

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

# Mapping numeric label values to string labels
label_map = {0: 'Negative', 1: 'Positive'}

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

# Loading the test dataset, which is also the Bates dataset, to predict the labels. Variables have been named as "test" so as to make it easier for the views to understand
x_test, y_test = load_and_preprocess_images(df, images_dir, target_size)
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Converting labels to categorical labels (one-hot encoding)
y_test = to_categorical(y_test, num_classes=num_classes)

# Loading the saved CNN model
model = load_model('cnn_model_batesdata.keras')

# Label Prediction
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Model Evaluation
print("Model Evaluation:")
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=[label_map[i] for i in range(num_classes)]))

# Printing Confusion Matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[label_map[i] for i in range(num_classes)], yticklabels=[label_map[i] for i in range(num_classes)])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
