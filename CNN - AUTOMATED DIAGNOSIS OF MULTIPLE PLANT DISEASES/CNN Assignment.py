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
print("Dataset directory:", dataset_path)

DATASET_BASE = "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
TRAIN_PATH = os.path.join(DATASET_BASE, "train")
VALID_PATH = os.path.join(DATASET_BASE, "valid")
TEST_PATH = "/kaggle/input/new-plant-diseases-dataset/test"
IMAGE_SIZE = (256, 256)
BATCH = 64

class_names = os.listdir(TRAIN_PATH)

plant_categories = []
image_counts = []
for cls in class_names:
    plant_categories.append(cls)
    images_in_class = os.listdir(os.path.join(TRAIN_PATH, cls))
    image_counts.append(len(images_in_class))

# Sort counts descending for visualization, but keep categories aligned
sorted_pairs = sorted(zip(image_counts, plant_categories), reverse=True)
image_counts_sorted, plant_categories_sorted = zip(*sorted_pairs)

sns.set(style="whitegrid", color_codes=True)
plt.figure(figsize=(20,20), dpi=200)
sns.barplot(x=image_counts_sorted, y=plant_categories_sorted, palette="Reds")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

# Data augmentation for training
train_augmenter = ImageDataGenerator(
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

valid_augmenter = ImageDataGenerator(rescale=1./255)

train_data_gen = train_augmenter.flow_from_directory(
    TRAIN_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH,
    class_mode='categorical'
)

valid_data_gen = valid_augmenter.flow_from_directory(
    VALID_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH,
    class_mode='categorical'
)

test_data_gen = valid_augmenter.flow_from_directory(
    TEST_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=False
)

model_net = Sequential([
    # Conv Block 1
    Conv2D(64, (3,3), padding='same', input_shape=(256,256,3)),
    BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    MaxPooling2D((2,2)),

    # Conv Block 2
    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Conv Block 3
    Conv2D(256, (3,3), padding='same'),
    BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    Conv2D(256, (3,3), padding='same'),
    BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    MaxPooling2D((2,2)),
    Dropout(0.4),

    # Conv Block 4
    Conv2D(512, (3,3), padding='same'),
    BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    Conv2D(512, (3,3), padding='same'),
    BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    MaxPooling2D((2,2)),
    Dropout(0.4),

    # Output Layers
    GlobalAveragePooling2D(),
    Dense(256),
    tf.keras.layers.LeakyReLU(),
    Dropout(0.5),
    Dense(38, activation='softmax')
])

model_net.summary()

model_net.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callback_list = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint("best_plant_model.h5", monitor='val_loss', save_best_only=True)
]

dbr_fit = model_net.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // BATCH,
    validation_data=valid_data_gen,
    validation_steps=valid_data_gen.samples // BATCH,
    epochs=15,
    callbacks=callback_list
)

test_loss_val, test_acc_val = model_net.evaluate(test_data_gen)
print(f"Test accuracy: {test_acc_val * 100:.2f}%")

plt.figure(figsize=(12,4))
plt.plot(dbr_fit.history['accuracy'], label='Training Accuracy')
plt.plot(dbr_fit.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(dbr_fit.history['loss'], label='Training Loss')
plt.plot(dbr_fit.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()