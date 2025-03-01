import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Define image size
img_height, img_width = 256, 256


#  Define classification dataset
train_ds_classification = tf.keras.utils.image_dataset_from_directory(
    r"kagglehub/datasets/anasmohammedtahir/covidqu/versions/7/Infection Segmentation Data/Infection Segmentation Data/Train",
    labels='inferred',
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=16,  # Adjust batch size as needed
    shuffle=True,
    seed=123,
)

# Define validation dataset
val_ds_classification = tf.keras.utils.image_dataset_from_directory(
    r"kagglehub/datasets/anasmohammedtahir/covidqu/versions/7/Infection Segmentation Data/Infection Segmentation Data/Val",
    labels='inferred',
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=16,
    shuffle=True,
    seed=123,
)


# Define the pre-trained densenet model
densenet201 = tf.keras.applications.DenseNet201(
    include_top=False,
    input_shape=(img_height, img_width, 3),
    weights="imagenet",
    pooling='avg',
    classes=3,
)

# Create the model
def create_model():
    model = keras.Sequential([
        densenet201,
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', # Changed to 'categorical_crossentropy'
                  metrics=['accuracy'])
    return model

# Create and compile the model
dnn_model = create_model()

# Fit the model
model_history = dnn_model.fit(
    train_ds_classification,
    verbose=1,
    validation_data=val_ds_classification,
    use_multiprocessing=True,
    workers=6,
    epochs=5
)

# Generating model loss graph
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.savefig('densenet_metrics.png')
plt.show()

# Save the model
dnn_model.save("densenet_covid_model.keras")