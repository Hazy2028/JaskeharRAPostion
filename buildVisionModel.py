import numpy as np
import pandas as pd
import h5py
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the ResNet152 model.
model = ResNet152(weights='imagenet')

# Prepare the training data.
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    '/uufs/chpc.utah.edu/common/home/u1313462/VQA-Med-2019/VQAMed2019Test/ParentImagesClass',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Compile the model.
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model.
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10)

# Extract the fc7 features.
fc7_features = model.predict_generator(train_generator, steps=len(train_generator))

# Save the fc7 features and the image file list using h5py.
with h5py.File('path/to/save/fc7_features.h5', 'w') as f:
    f.create_dataset('fc7_features', data=fc7_features)
    f.create_dataset('image_file_list', data=train_generator.filenames)