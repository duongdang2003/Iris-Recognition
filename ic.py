import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Function to extract HOG features from an image
def extract_features(image, resize_dim=(128, 128)):
    resized_img = cv2.resize(image, resize_dim)
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                   cells_per_block=(1, 1), visualize=False)
    return features

# Load dataset
data = []
labels = []

# Assume you have a dataset directory with subdirectories for each person
dataset_dir = 'E:/IC/gpt/dataset'
for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    if os.path.isdir(person_dir):
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                features = extract_features(img)
                data.append(features)
                labels.append(person)

data = np.array(data)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train an SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Function to recognize iris from a new image
def recognize_iris(image):
    features = extract_features(image)
    label = clf.predict([features])
    return label[0]

# Test the recognition on a new image
test_image = cv2.imread('E:/IC/gpt/test/305.jpg')
person = recognize_iris(test_image)
print(f'This iris belongs to: {person}')
