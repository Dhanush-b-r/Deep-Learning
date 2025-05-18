import os
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import kagglehub
dataset_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print("Path to dataset files:", dataset_path)

DATASET_BASE_DIR = "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
TRAIN_PATH = DATASET_BASE_DIR + "/train"
VALIDATION_PATH = DATASET_BASE_DIR + "/valid"
TEST_PATH = "/kaggle/input/new-plant-diseases-dataset/test"
IMAGE_SIZE = (256, 256)
BATCH = 64

disease_classes = os.listdir(TRAIN_PATH)

class_names = []
image_counts = []
for disease in disease_classes:
    class_names.append(disease)
    disease_images = os.listdir(TRAIN_PATH + "/" + disease)
    image_counts.append(len(disease_images))
image_counts.sort(reverse=True)

sns.set(style="whitegrid", color_codes=True)
plt.figure(figsize=(20,20), dpi=200)
ax = sns.barplot(x=image_counts, y=class_names, palette="Reds", hue=class_names)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_generator = train_image_generator.flow_from_directory(
    TRAIN_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH,
    class_mode='categorical'
)

validation_data_generator = validation_image_generator.flow_from_directory(
    VALIDATION_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH,
    class_mode='categorical'
)

test_data_generator = validation_image_generator.flow_from_directory(
    TEST_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=False
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization,
                                     Dropout, GlobalAveragePooling2D, Dense, LeakyReLU)

model_cnn = Sequential([
    # Block 1
    Conv2D(64, (3, 3), padding='same', input_shape=(256, 256, 3)),
    BatchNormalization(),
    LeakyReLU(),
    MaxPooling2D(pool_size=(2, 2)),

    # Block 2
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(256, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(),
    Conv2D(256, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    # Block 4
    Conv2D(512, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(),
    Conv2D(512, (3, 3), padding='same'),
    BatchNormalization(),
    LeakyReLU(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    # Final Layers
    GlobalAveragePooling2D(),
    Dense(256),
    LeakyReLU(),
    Dropout(0.5),
    Dense(38, activation='softmax')  # 38 classes for PlantVillage dataset
])

model_cnn.summary()

model_cnn.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

training_callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True)
]

history_cnn = model_cnn.fit(
    train_data_generator,
    steps_per_epoch=train_data_generator.samples // BATCH,
    validation_data=validation_data_generator,
    validation_steps=validation_data_generator.samples // BATCH,
    epochs=15,
    callbacks=training_callbacks
)

test_loss, test_acc = model_cnn.evaluate(test_data_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

plt.figure(figsize=(12, 4))
plt.plot(history_cnn.history['accuracy'], label='Train Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(history_cnn.history['loss'], label='Train Loss')
plt.plot(history_cnn.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
