import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Path to image with which to test the model
test_image_path = "covid_1585.png"

# Load Models
classification_model = tf.keras.models.load_model("models/densenet_covid_model.keras")
infection_segmentation_model = tf.keras.models.load_model("models/unet_infection_segmentation_model1.keras")
lung_segmentation_model = tf.keras.models.load_model("models/unet_lung_segmentation_model.keras")

# Image size
IMG_WIDTH = 256
IMG_HEIGHT = 256


# Load and resize images
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    return image

# Show the model predictions
def show_image_prediction(image_path):
    image = load_image(image_path)
    expanded_image = np.expand_dims(image, axis=0)

    # Predict classification
    prediction = classification_model.predict(expanded_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Lung Segmentation
    lung_mask = lung_segmentation_model.predict(expanded_image)[0]

    # Create a segmented lung image
    lung_mask = np.repeat(lung_mask, 3, axis=-1)
    segmented_image = image.copy()
    segmented_image[lung_mask == 0] = 0

    plt.figure(figsize=(10, 5))

    if predicted_class == 0: #Show if predicted class = COVID
        infection_mask = infection_segmentation_model.predict(expanded_image)[0]

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.xlabel("Original Image", fontsize=16)
        plt.xticks([]), plt.yticks([])

        plt.subplot(1, 3, 2)
        plt.imshow(segmented_image)
        plt.xlabel("Lung Segmentation", fontsize=16)
        plt.xticks([]), plt.yticks([])
        plt.subplot(1, 3, 3)
        plt.imshow(segmented_image)
        plt.imshow(infection_mask, 'Reds', alpha=0.4)
        plt.xlabel("Infection Segmentation", fontsize=16)
        plt.suptitle(f"COVID Detected (Confidence: {np.max(prediction):.2f})", fontsize=20)
        plt.xticks([]), plt.yticks([])
        plt.savefig('Detection_result.png')
        plt.show()
    else:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.xlabel("Original Image", fontsize=16)
        plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2)
        plt.imshow(segmented_image)
        plt.xlabel("Lung Segmentation", fontsize=16)
        if predicted_class == 1: # Show if predicted class = Non-COVID
            plt.suptitle(f"Non-COVID Condition (Confidence: {np.max(prediction):.2f})", fontsize=20)
        else: # Show if predicted class = Normal
            plt.suptitle(f"Normal Condition (Confidence: {np.max(prediction):.2f})", fontsize=20)
        plt.subplots_adjust(top=0.85)
        plt.xticks([]), plt.yticks([])
        plt.savefig('Detection_result.png')
        plt.show()


show_image_prediction(test_image_path)


