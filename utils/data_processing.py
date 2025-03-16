import pandas as pd
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from tqdm import tqdm
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
import os


# Method to load dataset from a directory path
def load_dataset(directory):
    """
    Loads images and their corresponding labels from a given directory.

    Args:
        directory (str): Path to the root directory containing subdirectories for each label.

    Returns:
        tuple: A tuple containing two lists:
            - image_paths: List of file paths to images.
            - labels: List of corresponding labels for the images.
    """
    image_paths = []
    labels = []

    # Iterate through each subdirectory (label) in the root directory
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):  # Ensure it's a directory
            for filename in os.listdir(label_dir):
                image_path = os.path.join(label_dir, filename)
                image_paths.append(image_path)
                labels.append(label)
        else:
            print(f"Skipping non-directory: {label}")

    return image_paths, labels


# Convert all images to a fixed shape (48x48) and grayscale
def extract_features(images):
    """
    Converts a list of image paths into a NumPy array of grayscale images with a fixed size (48x48).

    Args:
        images (list): List of file paths to images.

    Returns:
        np.ndarray: A 4D NumPy array of shape (num_images, 48, 48, 1) representing the images.
    """
    features = []
    for image in tqdm(images, desc="Extracting Features"):
        try:
            # Load the image in grayscale and convert it to a NumPy array
            img = load_img(image, color_mode='grayscale', target_size=(48, 48))
            img = np.array(img)
            features.append(img)
        except Exception as e:
            print(f"Error processing image {image}: {e}")

    # Convert the list of images to a NumPy array and reshape it
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features


def load_and_preprocess_data(train_dir, test_dir, num_classes):
    """
    Loads and preprocesses the training and testing datasets for use in a machine learning model.

    Args:
        train_dir (str): Path to the training dataset directory.
        test_dir (str): Path to the testing dataset directory.
        num_classes (int): Number of classes (categories) in the dataset.

    Returns:
        tuple: A tuple containing four elements:
            - x_train: Training images as a NumPy array (normalized).
            - x_test: Testing images as a NumPy array (normalized).
            - y_train: Training labels as one-hot encoded vectors.
            - y_test: Testing labels as one-hot encoded vectors.
    """
    # Load and shuffle the training dataset
    train = pd.DataFrame()
    train['image'], train['label'] = load_dataset(train_dir)
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle with a fixed seed
    print("Training dataset loaded and shuffled.")

    # Load the testing dataset
    test = pd.DataFrame()
    test['image'], test['label'] = load_dataset(test_dir)
    print("Testing dataset loaded.")

    # Extract features (convert images to arrays of shape 48x48x1)
    train_features = extract_features(train['image'])
    test_features = extract_features(test['image'])

    # Normalize the pixel values to the range [0, 1]
    x_train = train_features / 255.0
    x_test = test_features / 255.0
    print("Images normalized.")

    # Encode labels as integers
    le = LabelEncoder()
    le.fit(train['label'])
    y_train = le.transform(train['label'])
    y_test = le.transform(test['label'])

    # Convert labels to one-hot encoded vectors
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    print("Labels encoded and converted to one-hot vectors.")

    return x_train, x_test, y_train, y_test