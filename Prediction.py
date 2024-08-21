import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical, set_random_seed
from keras.preprocessing import image
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch
import clip
from PIL import Image

set_random_seed(0)

# Label and Image paths
csv_file_path = r"C:\Users\utkar\OneDrive\Desktop\dataset and csv for postive and negative labels\CLIP\csv for bates_CLIP.csv"
images_dir = r"C:\Users\utkar\Downloads\Utkarsh Gupta\NUIG\Semester 2\DA Project\Extracted Dataset updated"

# Loading CSV having filenames and labels
df = pd.read_csv(csv_file_path)
print(df.head())

# Printing the mapping of labels
label_mapping = dict(enumerate(pd.Categorical(df['label']).categories))
print("Label Mapping:", label_mapping)

# Defining class names based on their label mapping
class_names = list(label_mapping.values())

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

# Loading the test dataset, which is also the Bates dataset, to predict the labels. Variables have been named as "test" so as to make it easier for the views to understand
x_test, y_test = load_and_preprocess_images(df, images_dir, target_size)
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Converting labels to categorical labels (one-hot encoding)
y_test = to_categorical(y_test, num_classes=num_classes)

# Loading the saved CNN model
model = load_model('cnn_model_newdataset_new.keras')

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
print(classification_report(y_test_classes, y_pred_classes, target_names=class_names))

# Printing Confusion Matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
print(conf_matrix)

# Plotting confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Loading fine-tuned CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
model_clip.load_state_dict(torch.load('clip_model_pos_neg_csv.pth'))  # Load the fine-tuned model

# Preprocessing all images in Bates dataset for CLIP
def preprocess_for_clip(img_paths, preprocess):
    images = []
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        img_preprocessed = preprocess(img).unsqueeze(0).to(device)
        images.append(img_preprocessed)
    return images

# Converting the file paths for the images in the Bates dataset
image_paths = [os.path.join(images_dir, filename) for filename in df['filename']]
x_test_clip = preprocess_for_clip(image_paths, preprocess_clip)

# Image Encoding using CLIP on Bates dataset
with torch.no_grad():
    image_features = torch.cat([model_clip.encode_image(img) for img in x_test_clip])

# Loading and preprocessing the new image
new_image_path = r"C:\Users\utkar\OneDrive\Desktop\dataset\spongebob-squarepants-tv-series-1999-usa-season-8-2010-created-by-FX7KKA.jpg"
new_image_clip = preprocess_clip(Image.open(new_image_path).convert("RGB")).unsqueeze(0).to(device)

# Image Encoding using CLIP on the new image
with torch.no_grad():
    new_image_features = model_clip.encode_image(new_image_clip)

# Calculating cosine similarity
similarities = torch.nn.functional.cosine_similarity(new_image_features, image_features)

# Identifying the most similar image
most_similar_idx = torch.argmax(similarities).item()
most_similar_image_path = image_paths[most_similar_idx]

# Extracting the filename of the most similar image
most_similar_image_name = os.path.basename(most_similar_image_path)

# Calculating the cosine similarity value
cosine_sim_score = similarities[most_similar_idx].item()

# Extracting the true label of the most similar image
most_similar_image_true_label = class_names[df.iloc[most_similar_idx]['label']]

# Printing details of the most similar Image
print(f"\nMost Similar Image DataFrame Row:\n{df.iloc[most_similar_idx]}")

# Preprocessing the new image for CNN
new_image_cnn = image.load_img(new_image_path, target_size=target_size)
new_image_cnn = image.img_to_array(new_image_cnn) / 255.0
new_image_cnn = np.expand_dims(new_image_cnn, axis=0)

# Predicting label of the new image
new_image_pred = model.predict(new_image_cnn)
new_image_pred_class = np.argmax(new_image_pred, axis=1)[0]
new_image_pred_label = class_names[new_image_pred_class]

# Printing details of the most similar Image
print(f"Most Similar Image Name: {most_similar_image_name}")
print(f"Most Similar Image Path: {most_similar_image_path}")
print(f"Cosine Similarity Score: {cosine_sim_score:.4f}")
print(f"True Label of Most Similar Image: {most_similar_image_true_label}")

# Plotting the new image and its most similar image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(Image.open(new_image_path))
plt.title(f"New Image\nPredicted Label: {new_image_pred_label}")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(Image.open(most_similar_image_path))
plt.title(f"Most Similar Image in the dataset\nCosine Similarity: {cosine_sim_score:.4f}\nTrue Label: {most_similar_image_true_label}")
plt.axis('off')

plt.show()
