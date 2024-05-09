import numpy as np
from load_data import load_data
from pre_processing import pre_processing
from unet_model import unet_model

[train_images, train_labels, test_images, test_labels] = load_data()

[train_images, train_labels, test_images, test_labels] = pre_processing(train_images, train_labels, test_images, test_labels)

print("Train Images Shape:", train_images.shape)
print("Train Labels Shape:", train_labels.shape)
print("Test Images Shape:", test_images.shape)
print("Test Labels Shape:", test_labels.shape)


model = unet_model(input_size=(500, 300, 3), num_classes=3)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
