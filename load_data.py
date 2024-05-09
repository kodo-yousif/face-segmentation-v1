import os
import cv2
import numpy as np

base_dir = 'dataset/V2'  
train_rgb_dir = os.path.join(base_dir, 'Train_RGB')
train_labels_dir = os.path.join(base_dir, 'Train_Labels')
test_rgb_dir = os.path.join(base_dir, 'Test_RGB')
test_labels_dir = os.path.join(base_dir, 'Test_Labels')

def load_images(directory):
    images = []
    
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.bmp'):
            img_path = os.path.join(directory, filename)
            img = img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            images.append(np.array(img))
    return images


def load_data():
    
    train_images = load_images(train_rgb_dir)
    train_labels = load_images(train_labels_dir)
    test_images = load_images(test_rgb_dir)
    test_labels = load_images(test_labels_dir)

    return [train_images, train_labels, test_images, test_labels]

