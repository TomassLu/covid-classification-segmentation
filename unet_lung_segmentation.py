import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# All image and mask paths
train_image_dir1 = "kagglehub/datasets/anasmohammedtahir/covidqu/versions/7/Infection Segmentation Data/Infection Segmentation Data/Train/COVID-19/images"
train_image_dir2 = "kagglehub/datasets/anasmohammedtahir/covidqu/versions/7/Infection Segmentation Data/Infection Segmentation Data/Train/Non-COVID/images"
train_image_dir3 = "kagglehub/datasets/anasmohammedtahir/covidqu/versions/7/Infection Segmentation Data/Infection Segmentation Data/Train/Normal/images"

train_mask_dir1 = "kagglehub/datasets/anasmohammedtahir/covidqu/versions/7/Infection Segmentation Data/Lung Masks/Train/COVID-19/lung masks"
train_mask_dir2 = "kagglehub/datasets/anasmohammedtahir/covidqu/versions/7/Infection Segmentation Data/Lung Masks/Train/Non-COVID/lung masks"
train_mask_dir3 = "kagglehub/datasets/anasmohammedtahir/covidqu/versions/7/Infection Segmentation Data/Lung Masks/Train/Normal/lung masks"

val_image_dir1 = "kagglehub/datasets/anasmohammedtahir/covidqu/versions/7/Infection Segmentation Data/Infection Segmentation Data/Val/COVID-19/images"
val_image_dir2 = "kagglehub/datasets/anasmohammedtahir/covidqu/versions/7/Infection Segmentation Data/Infection Segmentation Data/Val/Non-COVID/images"
val_image_dir3 = "kagglehub/datasets/anasmohammedtahir/covidqu/versions/7/Infection Segmentation Data/Infection Segmentation Data/Val/Normal/images"
val_mask_dir1 = "kagglehub/datasets/anasmohammedtahir/covidqu/versions/7/Infection Segmentation Data/Lung Masks/Val/COVID-19/lung masks"
val_mask_dir2 = "kagglehub/datasets/anasmohammedtahir/covidqu/versions/7/Infection Segmentation Data/Lung Masks/Val/Non-COVID/lung masks"
val_mask_dir3 = "kagglehub/datasets/anasmohammedtahir/covidqu/versions/7/Infection Segmentation Data/Lung Masks/Val/Normal/lung masks"

# Image size and batch size
IMG_WIDTH = 256
IMG_HEIGHT = 256
BATCH_SIZE = 8


# Load, resize and normalize images
def load_and_preprocess_image(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH])
    mask = tf.cast(mask, tf.float32) / 255.0
    return image, mask


# Load training images and masks
train_image_paths = []
train_mask_paths = []
for image_file in os.listdir(train_image_dir1):
    if image_file.endswith(('.png')):
        train_image_paths.append(os.path.join(train_image_dir1, image_file))
        train_mask_paths.append(os.path.join(train_mask_dir1, image_file))
for image_file in os.listdir(train_image_dir2):
    if image_file.endswith(('.png')):
        train_image_paths.append(os.path.join(train_image_dir2, image_file))
        train_mask_paths.append(os.path.join(train_mask_dir2, image_file))
for image_file in os.listdir(train_image_dir3):
    if image_file.endswith(('.png')):
        train_image_paths.append(os.path.join(train_image_dir3, image_file))
        train_mask_paths.append(os.path.join(train_mask_dir3, image_file))

# Load validation images and masks
val_image_paths = []
val_mask_paths = []
for image_file in os.listdir(val_image_dir1):
    if image_file.endswith(('.png')):
        val_image_paths.append(os.path.join(val_image_dir1, image_file))
        val_mask_paths.append(os.path.join(val_mask_dir1, image_file))
for image_file in os.listdir(val_image_dir2):
    if image_file.endswith(('.png')):
        val_image_paths.append(os.path.join(val_image_dir2, image_file))
        val_mask_paths.append(os.path.join(val_mask_dir2, image_file))
for image_file in os.listdir(val_image_dir3):
    if image_file.endswith(('.png')):
        val_image_paths.append(os.path.join(val_image_dir3, image_file))
        val_mask_paths.append(os.path.join(val_mask_dir3, image_file))


# Create train image-mask dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))
train_dataset = train_dataset.shuffle(buffer_size=10)
train_dataset = train_dataset.map(load_and_preprocess_image).batch(BATCH_SIZE)

# Create validation image-mask dataset
val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))
val_dataset = val_dataset.shuffle(buffer_size=10)
val_dataset = val_dataset.map(load_and_preprocess_image).batch(BATCH_SIZE)


# U-Net model
def create_unet_model():
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Encoder Path
    s1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    s1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(s1)
    p1 = layers.MaxPooling2D((2, 2))(s1)

    s2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    s2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(s2)
    p2 = layers.MaxPooling2D((2, 2))(s2)

    s3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    s3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(s3)
    p3 = layers.MaxPooling2D((2, 2))(s3)

    s4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    s4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(s4)
    p4 = layers.MaxPooling2D((2, 2))(s4)

    # Bridge
    b1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    b1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(b1)

    # Decoder Path
    u1 = layers.UpSampling2D((2, 2))(b1)
    c1 = layers.concatenate([u1, s4])
    d1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c1)
    d1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(d1)

    u2 = layers.UpSampling2D((2, 2))(d1)
    c2 = layers.concatenate([u2, s3])
    d2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    d2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(d2)

    u3 = layers.UpSampling2D((2, 2))(d2)
    c3 = layers.concatenate([u3, s2])
    d3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    d3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(d3)

    u4 = layers.UpSampling2D((2, 2))(d3)
    c4 = layers.concatenate([u4, s1])
    d4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c4)
    d4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(d4)

    outputs = layers.Conv2D(1, 1, padding="same", activation='sigmoid')(d4)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# Create and compile model
model = create_unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)  # Adjust epochs as needed


# Save the model
model.save("unet_lung_segmentation_model.keras")
