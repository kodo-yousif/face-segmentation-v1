import cv2
import numpy as np

def normalize_and_resize(images):

    for i in range(len(images)):
        images[i] = images[i].astype('float32') / 255.0
        images[i] = cv2.resize(images[i], (300 , 500), interpolation=cv2.INTER_AREA)

    
    return np.array(images)

def pre_processing(train_images, train_labels, test_images, test_labels):

    train_images = normalize_and_resize(train_images)
    train_labels = normalize_and_resize(train_labels)
    test_images = normalize_and_resize(test_images)
    test_labels = normalize_and_resize(test_labels)


    return [train_images, train_labels, test_images, test_labels]